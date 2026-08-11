"""Generate an INT8 calibration table via ONNX Runtime (memory-efficient).

This bypasses ``run_calibration.py``'s MLIR reference interpreter, which needs
>28 GB RAM for this 244M-param model (it holds all weights + every intermediate
in a dict without freeing).  ORT frees intermediates between samples, so peak
memory is ~3 GB regardless of sample count, and we can afford many samples.

The calibration table is TPU-MLIR text format — one ``<name> <threshold> <min> <max>``
per line — where ``threshold = max|activation|`` and ``min``/``max`` are the observed
extrema across samples (the ``max`` cali method, symmetric quantization).  MLIR tensor
names are parsed from the MLIR file's ``#locN = loc("...")`` definitions; each maps
from an ONNX node output as ``onnx_output_name + "_" + op_type`` (e.g.
``/model/Mul_output_0`` + ``_Mul`` -> ``/model/Mul_output_0_Mul``).  Tensors whose
MLIR names can't be mapped (auto-named ``_v_NNNN`` by TPU-MLIR, ~11 % of activations)
are omitted and get default thresholds in ``model_deploy.py``.

Usage (in the torch 1.13 export venv)::

    python tools/gen_cali_table_ort.py \
        --onnx models/onnx/tapnext_init.onnx \
        --mlir scripts/tapnext_init.mlir \
        --data-list datasets/cali_data/init_cali.txt \
        --input-num 6 \
        -o scripts/tapnext_init_cali_table
"""
import argparse
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def parse_mlir_loc_names(mlir_path):
    """Extract all location names from ``#locN = loc("name")`` definitions."""
    text = Path(mlir_path).read_text()
    return set(re.findall(r'#loc\d+ = loc\("([^"]+)"\)', text))


def parse_mlir_activation_names(mlir_path):
    """Extract loc names of activation tensors (outputs of non-Weight/None ops).

    The TPU-MLIR lowering pass needs *every* activation tensor to have a
    ``CalibratedQuantizedType`` — if a tensor is missing from the cali table,
    ``getScaleAndZeroPoint`` / ``getQuantInt8Type`` hits an UNREACHABLE and
    crashes.  We parse the MLIR line-by-line to find all op outputs except
    ``top.Weight`` and ``top.None`` (weights / placeholders, not activations)
    and return their loc names.  Line-by-line is required because a single
    regex with ``[^l]*`` to skip attributes breaks on attribute names
    containing ``l`` (``relu_limit``, ``left_transpose``, ``is_lora`` in
    MatMul, etc.), silently dropping those ops' outputs.
    """
    text = Path(mlir_path).read_text()
    loc_defs = dict(re.findall(r'(#loc\d+) = loc\("([^"]+)"\)', text))
    activations = set()
    for line in text.splitlines():
        m = re.search(r'"(top\.\w+)"', line)
        if not m:
            continue
        op_type = m.group(1)
        if op_type in ("top.Weight", "top.None"):
            continue
        m2 = re.search(r'loc\((#loc\d+|unknown)\)\s*$', line)
        if m2 and m2.group(1) in loc_defs:
            activations.add(loc_defs[m2.group(1)])
    return activations


def build_mlir_to_onnx_mapping(model, mlir_names):
    """Map MLIR tensor loc names to ONNX tensor names (reverse mapping).

    TPU-MLIR fuses ONNX op groups into single top-level ops (e.g. multiple ONNX
    ops implementing LayerNorm -> one ``top.LayerNorm``).  The MLIR loc name is
    ``<onnx_tensor_name>_<mlir_op_type>`` where ``onnx_tensor_name`` is an ONNX
    intermediate in the fused group.  We reverse this: for each MLIR name, find
    the longest ONNX tensor name that is its prefix (``mlir_name.startswith(
    onnx_name + "_")``).  This handles fused ops correctly — the forward mapping
    (``onnx_name + "_" + onnx_op_type``) misses them because the ONNX op type
    differs from the MLIR op type.

    Returns ``{mlir_name: onnx_name}`` for MLIR names that match an ONNX tensor.
    """
    vi_names = {vi.name for vi in model.graph.value_info}
    node_outs = set()
    for node in model.graph.node:
        for out in node.output:
            if out:
                node_outs.add(out)
    all_onnx = vi_names | node_outs

    mapping = {}
    for mlir_name in mlir_names:
        if not mlir_name.startswith("/"):
            continue
        best = None
        for onnx_name in all_onnx:
            if mlir_name.startswith(onnx_name + "_"):
                if best is None or len(onnx_name) > len(best):
                    best = onnx_name
        if best is not None:
            mapping[mlir_name] = best
    return mapping


def add_all_intermediate_outputs(model):
    """Add every ``value_info`` tensor as a graph output so ORT returns it."""
    existing = {o.name for o in model.graph.output}
    for vi in model.graph.value_info:
        if vi.name not in existing:
            model.graph.output.append(vi)
            existing.add(vi.name)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mlir", required=True)
    ap.add_argument("--data-list", required=True)
    ap.add_argument("--input-num", type=int, default=0, help="0 = all samples in data_list")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    # --- MLIR tensor names ---
    mlir_names = parse_mlir_loc_names(args.mlir)
    print(f"[mlir] {len(mlir_names)} location names from {args.mlir}")
    activation_names = parse_mlir_activation_names(args.mlir)
    print(f"[mlir] {len(activation_names)} activation tensors (non-Weight/None)")

    # --- ONNX model + name mapping ---
    model = onnx.load(args.onnx)
    mapping = build_mlir_to_onnx_mapping(model, mlir_names)
    print(f"[map] {len(mapping)} MLIR names -> ONNX tensors")

    # --- expose all intermediates as ORT outputs ---
    model = add_all_intermediate_outputs(model)
    n_out = len(model.graph.output)
    print(f"[onnx] {n_out} total outputs (incl. intermediates)")
    tmp_path = "/tmp/_cali_onnx.onnx"
    onnx.save(model, tmp_path)

    sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    print(f"[ort] {len(input_names)} inputs, {len(output_names)} outputs")

    # --- data list ---
    lines = [l.strip() for l in Path(args.data_list).read_text().strip().split("\n") if l.strip()]
    if args.input_num > 0:
        lines = lines[: args.input_num]
    print(f"[data] {len(lines)} samples")

    # --- collect min/max per tensor across samples ---
    # TPU-MLIR cali table format: <name> <threshold> <min> <max>
    # threshold = max(|min|, |max|) for symmetric quantization (the `max` method)
    stats = {}          # onnx tensor name -> [cur_min, cur_max] (intermediates)
    input_stats = {}    # input name -> [cur_min, cur_max]
    for i, path in enumerate(lines):
        data = np.load(path)
        feed = {k: data[k].astype(np.float32) for k in data.files if k in input_names}
        # input thresholds
        for k, arr in feed.items():
            mn, mx = float(arr.min()), float(arr.max())
            if k not in input_stats:
                input_stats[k] = [mn, mx]
            else:
                input_stats[k][0] = min(input_stats[k][0], mn)
                input_stats[k][1] = max(input_stats[k][1], mx)
        # intermediate thresholds
        outs = sess.run(None, feed)
        for name, arr in zip(output_names, outs):
            if arr.size == 0:
                continue
            mn, mx = float(arr.min()), float(arr.max())
            if name not in stats:
                stats[name] = [mn, mx]
            else:
                stats[name][0] = min(stats[name][0], mn)
                stats[name][1] = max(stats[name][1], mx)
        print(f"  [{i + 1}/{len(lines)}] {Path(path).name}")

    # --- write calibration table ---
    # Every MLIR activation tensor must be in the table, or the lowering pass
    # crashes with getQuantInt8Type on an uncalibrated type.  Use ORT values
    # where available; default threshold 1.0 for tensors ORT can't reach.
    DEFAULT_THR = 1.0
    lines_out = []
    covered = set()
    # inputs (frame, query_points, step) — MLIR name == ONNX name, no suffix
    for name in input_names:
        if name in mlir_names and name in input_stats:
            mn, mx = input_stats[name]
            thr = max(abs(mn), abs(mx))
            if thr > 0:
                lines_out.append(f"{name} {thr:.7f} {mn:.7f} {mx:.7f}")
                covered.add(name)
    # intermediates — mapping is {mlir_name: onnx_name}
    n_inter = 0
    n_default = 0
    for mlir_name, onnx_name in mapping.items():
        if onnx_name not in stats:
            continue
        mn, mx = stats[onnx_name]
        thr = max(abs(mn), abs(mx))
        if thr > 0:
            lines_out.append(f"{mlir_name} {thr:.7f} {mn:.7f} {mx:.7f}")
            covered.add(mlir_name)
            n_inter += 1
    # fill defaults for activation tensors not covered by ORT
    for mlir_name in sorted(activation_names):
        if mlir_name not in covered:
            lines_out.append(f"{mlir_name} {DEFAULT_THR:.7f} {-DEFAULT_THR:.7f} {DEFAULT_THR:.7f}")
            n_default += 1

    Path(args.output).write_text("\n".join(lines_out) + "\n")
    print(f"\n[done] {len(lines_out)} entries ({len(lines_out) - n_inter - n_default} inputs + "
          f"{n_inter} ORT intermediates + {n_default} defaults) -> {args.output}")


if __name__ == "__main__":
    main()
