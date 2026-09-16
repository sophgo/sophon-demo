#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
#!/usr/bin/env python3
"""Restore pi05 orbax ckpt, extract PaliGemma/img (SigLIP vision tower) weights,
assemble a PyTorch state_dict, and save npz + printed key/shape report."""
import os as _os

# All inputs and outputs live under PI05_ROOT, the working tree that holds the
# official openpi checkout (openpi-ref/), the official checkpoint (ckpt_official/)
# and the generated artifacts. Set it before running:
#     export PI05_ROOT=/path/to/pi05-work
_PI05_ROOT = _os.environ.get("PI05_ROOT")
if not _PI05_ROOT:
    raise SystemExit("PI05_ROOT is not set; see tools/export/README.md")

import pathlib
import json

import numpy as np
import jax
import orbax.checkpoint as ocp
import flax.traverse_util as traverse_util

CKPT = pathlib.Path(_PI05_ROOT) / "ckpt_official/pi05_libero/params"
OUT = pathlib.Path(_PI05_ROOT) / "siglip"
LOG = pathlib.Path(_PI05_ROOT) / "logs"

DEPTH = 27
WIDTH = 1152
MLP_DIM = 4304
NUM_HEADS = 16
HEAD_DIM = WIDTH // NUM_HEADS  # 72
PROJ_DIM = 2048  # paligemma gemma_2b width
NUM_PATCHES = 256  # 224/14 ^ 2


def make_restore_args(item):
    return jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item)


def restore_params():
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(str(CKPT))
        item = {"params": metadata["params"]}
        params = ckptr.restore(
            str(CKPT),
            ocp.args.PyTreeRestore(item=item, restore_args=make_restore_args(item)),
        )["params"]
    flat = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat):
        flat = {kp[:-1]: v for kp, v in flat.items()}
    return traverse_util.unflatten_dict(flat)


def get(tree, path):
    node = tree
    for p in path:
        node = node[p]
    return np.asarray(node)


def main():
    print("restoring checkpoint ...", flush=True)
    params = restore_params()
    img = params["PaliGemma"]["img"]
    print("restored.", flush=True)

    sd = {}

    def assign(name, arr):
        sd[name] = arr.astype(np.float32)
        print(f"  {name}: {arr.shape} -> {sd[name].shape}", flush=True)

    # patch embedding: JAX NHWC conv kernel [14,14,3,1152]
    assign("patch_embed.weight", get(img, ["embedding", "kernel"]).transpose(3, 2, 0, 1))
    assign("patch_embed.bias", get(img, ["embedding", "bias"]))
    assign("pos_embed", get(img, ["pos_embedding"]))  # [1,256,1152]

    blk = img["Transformer"]["encoderblock"]
    ln0_scale, ln0_bias = get(blk, ["LayerNorm_0", "scale"]), get(blk, ["LayerNorm_0", "bias"])
    ln1_scale, ln1_bias = get(blk, ["LayerNorm_1", "scale"]), get(blk, ["LayerNorm_1", "bias"])
    d0k, d0b = get(blk, ["MlpBlock_0", "Dense_0", "kernel"]), get(blk, ["MlpBlock_0", "Dense_0", "bias"])
    d1k, d1b = get(blk, ["MlpBlock_0", "Dense_1", "kernel"]), get(blk, ["MlpBlock_0", "Dense_1", "bias"])
    for k in ["query", "key", "value"]:
        pass
    atn = blk["MultiHeadDotProductAttention_0"]
    qk, qb = get(atn, ["query", "kernel"]), get(atn, ["query", "bias"])
    kk, kb = get(atn, ["key", "kernel"]), get(atn, ["key", "bias"])
    vk, vb = get(atn, ["value", "kernel"]), get(atn, ["value", "bias"])
    ok, ob = get(atn, ["out", "kernel"]), get(atn, ["out", "bias"])
    enc_scale, enc_bias = get(img, ["Transformer", "encoder_norm", "scale"]), get(
        img, ["Transformer", "encoder_norm", "bias"]
    )
    hk, hb = get(img, ["head", "kernel"]), get(img, ["head", "bias"])

    for i in range(DEPTH):
        pre = f"blocks.{i}"
        assign(f"{pre}.ln1.weight", ln0_scale[i])
        assign(f"{pre}.ln1.bias", ln0_bias[i])
        assign(f"{pre}.ln2.weight", ln1_scale[i])
        assign(f"{pre}.ln2.bias", ln1_bias[i])
        assign(f"{pre}.mlp.fc1.weight", d0k[i].T)
        assign(f"{pre}.mlp.fc1.bias", d0b[i])
        assign(f"{pre}.mlp.fc2.weight", d1k[i].T)
        assign(f"{pre}.mlp.fc2.bias", d1b[i])
        assign(f"{pre}.q.weight", qk[i].reshape(NUM_HEADS * HEAD_DIM, WIDTH).T)
        assign(f"{pre}.q.bias", qb[i].reshape(-1))
        assign(f"{pre}.k.weight", kk[i].reshape(NUM_HEADS * HEAD_DIM, WIDTH).T)
        assign(f"{pre}.k.bias", kb[i].reshape(-1))
        assign(f"{pre}.v.weight", vk[i].reshape(NUM_HEADS * HEAD_DIM, WIDTH).T)
        assign(f"{pre}.v.bias", vb[i].reshape(-1))
        assign(f"{pre}.o.weight", ok[i].reshape(NUM_HEADS * HEAD_DIM, WIDTH).T)
        assign(f"{pre}.o.bias", ob[i].reshape(-1))

    assign("ln.weight", enc_scale)
    assign("ln.bias", enc_bias)
    assign("head.weight", hk.T)
    assign("head.bias", hb)

    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for k, v in sd.items():
        if np.issubdtype(v.dtype, np.floating):
            arrays[k] = v.reshape(-1)
    np.savez_compressed(OUT / "siglip_visual_weights.npz", **{k: v for k, v in sd.items()})

    with open(LOG / "siglip_extract_report.json", "w") as f:
        json.dump({k: [v.shape, v.dtype.name] for k, v in sd.items()}, f, indent=1, default=str)
    print(f"saved {len(sd)} tensors -> {OUT / 'siglip_visual_weights.npz'}")
    print("DONE")


if __name__ == "__main__":
    main()