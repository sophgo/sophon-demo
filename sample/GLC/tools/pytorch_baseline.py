#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""主机侧 PyTorch 基线：跑 GLC 官方 test()，产出精度对比参照。

输出的 JSON 是 tools/eval_accuracy.py 的输入，用于与设备端结果对比。

用法:
    PYTHONPATH=tools/stubs python3 tools/pytorch_baseline.py \
        --ckpt GLC_image.pth.tar --glc-src GLC_src --images images \
        --out baseline.json [--save-recon recon/]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch


def get_padding_size(height, width, p=64):
    pl = (p - width % p) % p
    pt = (p - height % p) % p
    return pl // 2, pl - pl // 2, pt // 2, pt - pt // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--glc-src", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="baseline.json")
    ap.add_argument("--save-recon", default=None)
    ap.add_argument("--q-indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（0=全部）")
    args = ap.parse_args()

    sys.path.insert(0, args.glc_src)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "python"))
    from src.models.image_model import GLC_Image
    from src.utils.test_utils import get_state_dict
    from metrics import psnr, ssim
    from PIL import Image

    net = GLC_Image(inplace=False)
    net.load_state_dict(get_state_dict(args.ckpt), strict=True)
    net.eval()

    files = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    if args.limit:
        files = files[: args.limit]

    if args.save_recon:
        os.makedirs(args.save_recon, exist_ok=True)

    results = {}
    for q in args.q_indexes:
        per_q = {}
        bpps, psnrs, ssims = [], [], []
        for fn in files:
            img = np.asarray(Image.open(os.path.join(args.images, fn)).convert("RGB"))
            h, w = img.shape[:2]
            x = img.astype(np.float32) / 255.0
            x = (x - 0.5) * 2.0
            x = torch.from_numpy(x.transpose(2, 0, 1)[None])

            pl, pr, pt, pb = get_padding_size(h, w, 64)
            if pl or pr or pt or pb:
                x = torch.nn.functional.pad(x, (pl, pr, pt, pb), mode="replicate")

            with torch.no_grad():
                r = net.test(x, q)

            rec = r["x_hat"].clamp(-1, 1)
            rec = torch.nn.functional.pad(rec, (-pl, -pr, -pt, -pb))
            rec = rec[0].numpy().transpose(1, 2, 0)
            rec = np.clip(np.rint(((rec / 2.0) + 0.5) * 255.0), 0, 255).astype(np.uint8)

            bpp = r["bit"] / h / w
            p = psnr(rec, img)
            s = ssim(rec, img)
            per_q[fn] = {"bpp": bpp, "bit_y": r["bit_y"], "bit_z": r["bit_z"],
                         "psnr": p, "ssim": s}
            bpps.append(bpp)
            psnrs.append(p)
            ssims.append(s)
            if args.save_recon:
                Image.fromarray(rec).save(
                    os.path.join(args.save_recon, "q%d_%s" % (q, fn)))
        per_q["_avg"] = {"bpp": float(np.mean(bpps)), "psnr": float(np.mean(psnrs)),
                         "ssim": float(np.mean(ssims))}
        results["q%d" % q] = per_q
        print("q%d: bpp=%.4f  psnr=%.2f  ssim=%.4f  (%d 张)"
              % (q, per_q["_avg"]["bpp"], per_q["_avg"]["psnr"],
                 per_q["_avg"]["ssim"], len(files)))

    meta = {"ckpt": os.path.basename(args.ckpt), "num_images": len(files),
            "images": files, "codebook_size": 16384}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print("基线已写入", args.out)


if __name__ == "__main__":
    main()
