"""Generate a mixed-precision qtable for TAPNext++ INT8 compilation.

The pure-INT8 model suffers quantization degeneration (track outputs collapse to
(1.0, 1.0)).  Keeping the prediction heads + soft-argmax in F16 while the rest is
INT8 prevents this.  This script identifies those ops by ONNX graph traversal and
writes a TPU-MLIR ``--quantize_table`` file.

Identification strategy
-----------------------
1. **Prediction head nodes** — every ONNX node whose name contains
   ``coordinate_head`` or ``visible_head`` (the two MLPs that predict point
   coordinates and visibility logits).
2. **Soft-argmax nodes** — all nodes *downstream* of the prediction heads
   (ArgMax, Softmax, LessOrEqual, Sub, Div, Mul, ReduceSum, …) found by
   forward graph traversal from the head outputs to the model outputs.
3. **Output formatting** — nodes named ``tracks_*`` or ``vis_logits_*``.

These ONNX tensor names are mapped to MLIR loc names via prefix matching (same
logic as ``gen_cali_table_ort.py``).  Auto-named MLIR ops (``_v_NNNN_Gemm`` /
``_v_NNNN_Reshape``) from fused MatMul+Add in the prediction heads are identified
by proximity to prediction-head loc definitions in the MLIR — TPU-MLIR emits loc
definitions in graph order, so a Gemm sandwiched between
``model.coordinate_head.N.bias`` and
``/model/coordinate_head/.../Add_output_0_Reshape`` belongs to the head.

Usage (inside the TPU-MLIR container, after model_transform)::

    python ../tools/gen_mix_qtable.py \
        --onnx ../models/onnx/tapnext_init.onnx \
        --mlir tapnext_init.mlir \
        --chip bm1688 \
        -o tapnext_init_mix_qtable
"""
import argparse
import re
from pathlib import Path

import onnx


def build_consumer_map(model):
    """Map tensor name → list of node indices that consume it."""
    consumers = {}
    for i, node in enumerate(model.graph.node):
        for inp in node.input:
            consumers.setdefault(inp, []).append(i)
    return consumers


def forward_trace(model, seed_node_indices):
    """Return all node indices reachable forward from seed nodes."""
    consumers = build_consumer_map(model)
    visited = set(seed_node_indices)
    queue = list(seed_node_indices)
    while queue:
        ni = queue.pop()
        for out in model.graph.node[ni].output:
            for ci in consumers.get(out, []):
                if ci not in visited:
                    visited.add(ci)
                    queue.append(ci)
    return visited


def find_prediction_head_nodes(model):
    """Find seed nodes: coordinate_head, visible_head, tracks_*, vis_logits_*."""
    seeds = set()
    for i, node in enumerate(model.graph.node):
        n = node.name
        if "coordinate_head" in n or "visible_head" in n:
            seeds.add(i)
        elif n.startswith("tracks_") or n.startswith("vis_logits_"):
            seeds.add(i)
    return seeds


def collect_tensor_names(model, node_indices):
    """Collect all output tensor names of the given nodes."""
    names = set()
    for ni in node_indices:
        for out in model.graph.node[ni].output:
            if out:
                names.add(out)
    return names


def map_onnx_to_mlir(onnx_names, mlir_loc_names):
    """Map ONNX tensor names to MLIR loc names via prefix matching.

    MLIR loc name = <onnx_tensor_name>_<mlir_op_type>.  Reverse: find the
    longest ONNX name that is a prefix of the MLIR name.
    """
    mapping = set()
    for mlir_name in mlir_loc_names:
        if not mlir_name.startswith("/"):
            continue
        best = None
        for onnx_name in onnx_names:
            if mlir_name.startswith(onnx_name + "_"):
                if best is None or len(onnx_name) > len(best):
                    best = onnx_name
        if best is not None:
            mapping.add(mlir_name)
    return mapping


def parse_mlir_loc_defs(mlir_path):
    """Return ordered list of (loc_id, name) from #locN = loc("name") defs."""
    text = Path(mlir_path).read_text()
    return re.findall(r'#loc\d+ = loc\("([^"]+)"\)', text)


def find_adjacent_auto_ops(loc_names, head_patterns):
    """Find _v_NNNN ops adjacent to prediction-head loc defs.

    TPU-MLIR emits loc definitions in graph order.  A ``_v_NNNN_Gemm`` or
    ``_v_NNNN_Reshape`` that sits between a head weight/bias loc and a head
    output loc belongs to the prediction head.
    """
    auto_ops = set()
    for i, name in enumerate(loc_names):
        if not re.match(r"_v_\d+_(Gemm|Reshape)", name):
            continue
        # check neighbors within ±3 lines for head patterns
        window = loc_names[max(0, i - 3):i + 4]
        if any(any(p in w for p in head_patterns) for w in window):
            auto_ops.add(name)
    return auto_ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mlir", required=True)
    ap.add_argument("--chip", default="bm1688")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    # --- ONNX graph traversal ---
    model = onnx.load(args.onnx)
    seeds = find_prediction_head_nodes(model)
    print(f"[onnx] {len(seeds)} prediction-head seed nodes")

    all_nodes = forward_trace(model, seeds)
    print(f"[onnx] {len(all_nodes)} nodes in prediction head + downstream")

    onnx_tensors = collect_tensor_names(model, all_nodes)
    print(f"[onnx] {len(onnx_tensors)} output tensor names")

    # --- MLIR loc names ---
    loc_names = parse_mlir_loc_defs(args.mlir)
    loc_set = set(loc_names)
    print(f"[mlir] {len(loc_set)} loc definitions")

    # --- map ONNX tensors to MLIR loc names ---
    mapped = map_onnx_to_mlir(onnx_tensors, loc_set)
    print(f"[map] {len(mapped)} MLIR loc names from ONNX tensor mapping")

    # --- find auto-named _v_NNNN ops adjacent to prediction heads ---
    head_patterns = ["coordinate_head", "visible_head", "tracks_", "vis_logits_"]
    auto_ops = find_adjacent_auto_ops(loc_names, head_patterns)
    print(f"[auto] {len(auto_ops)} _v_NNNN ops adjacent to prediction heads")

    # --- also include MLIR loc names that directly contain head patterns ---
    direct = {n for n in loc_set
              if any(p in n for p in head_patterns)}
    print(f"[direct] {len(direct)} MLIR loc names with head patterns")

    # --- union ---
    f16_ops = mapped | auto_ops | direct
    # keep only MLIR activation tensor names (ONNX-path "/", auto "_v_", or
    # renamed output ops "tracks_"/"vis_logits_")
    f16_ops = {n for n in f16_ops
               if n.startswith("/") or n.startswith("_v_")
               or n.startswith("tracks_") or n.startswith("vis_logits_")}
    f16_ops = sorted(f16_ops)
    print(f"[total] {len(f16_ops)} F16 ops")

    # --- write qtable ---
    lines = [
        f"# chip: {args.chip}  mix_mode: F16",
        f"# number of F16 layer: {len(f16_ops)}",
        "###",
        "# op_name   quantize_mode",
    ]
    for op in f16_ops:
        lines.append(f"{op} F16")
    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"[done] qtable written to {args.output}")


if __name__ == "__main__":
    main()
