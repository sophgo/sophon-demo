#!/usr/bin/env python3
"""跨芯片互解实测：用 --precision 指定的档案跑完整编解码，落盘码流与 y_hat。

部署形态的真实模拟：
  --mode encode  只写 .glc 码流 + dump 编码端 y_hat（模拟压缩端）
  --mode decode  只读 .glc 码流 + dump 解码端 y_hat（模拟还原端）
两者分处两台机器，中间只传码流文件。

用法：
    cross_chip_run.py <models_dir> <precision> <mode> <image|bitstream> <out_prefix>

输出（out_prefix 加后缀）：
    <prefix>.glc        码流
    <prefix>_yhat.npz   该端的 y_hat
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) != 6:
        print(__doc__)
        return 2
    models, precision, mode, src, prefix = sys.argv[1:6]
    # 兼容两种落地方式：脚本与模块同目录（设备上平铺部署），
    # 或脚本在 tools/、模块在 ../python（仓库布局）。
    for d in (HERE, os.path.dirname(HERE),
              os.path.join(os.path.dirname(HERE), "python")):
        if d not in sys.path:
            sys.path.insert(0, d)

    import glc_codec as gc

    t0 = time.perf_counter()
    if mode == "encode":
        import glc_encode as ge
        from glc_infer import load_image, pad_to_multiple, preprocess

        img = load_image(src)
        h, w = img.shape[:2]
        padded, _ = pad_to_multiple(img)
        x = preprocess(padded)
        enc = ge.Encoder(models, precision)
        try:
            bitstream, y_hat = enc.encode(x, 0, (h, w))
        finally:
            enc.close()
        with open(prefix + ".glc", "wb") as f:
            f.write(bitstream)
    else:
        import glc_decode as gd

        with open(src, "rb") as f:
            bitstream = f.read()
        dec = gd.Decoder(models, precision)
        try:
            x_hat, y_hat = dec.decode(bitstream)
        finally:
            dec.close()
        np.savez(prefix + "_xhat.npz", x_hat=x_hat.astype(np.float32))

    np.savez(prefix + "_yhat.npz", y_hat=np.ascontiguousarray(y_hat, dtype=np.float32))
    ms = (time.perf_counter() - t0) * 1000
    bd = gc.bpp_breakdown(bitstream, 512, 768)
    print("%s %s: 码流 %d 字节  bpp %.6f  y_hat sha=%s  %.1f ms"
          % (mode, precision, len(bitstream), bd["total"],
             __import__("hashlib").sha256(
                 np.ascontiguousarray(y_hat, dtype=np.float32).tobytes()
             ).hexdigest()[:16], ms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
