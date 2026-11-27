#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""
MoGe 精度评测脚本（官方评测数据集，官方 depth_metric 指标口径）。

在 MoGe 官方的 evaluation 协议中，``--mg moge2`` 的 ``depth_metric`` 组对
**度量深度**（metric depth）直接计算两种指标（不经过尺度/位移对齐）：

    rel    = mean(|pred - gt| / (gt + 1e-6))          # 即 AbsRel
    delta1 = mean(max(gt/pred, pred/gt) < 1.25)       # 即 delta 1.25

（见 moge/test/metrics.py 的 depth_metrics_dict / rel_depth / delta_depth。）

本脚本用一个与例程部署路径完全一致、可复现的**方块面板（square-panel）口径**：
对于官方数据集中的每张原始图，不做官方 dataloader 里的相机重投影 warp（那需要
torch + PIL + utils3d 才能在 x86 上跑，与板端 numpy 环境割裂），而是把原图按例程
相同的方式缩放到 bmodel 的静态输入 518x518（aspect=1，直接 squish），推理得到
518x518 的度量深度 + mask，再双线性缩放回原始 GT 分辨率，与原始 ``depth.png``
（经 near/far 元数据解码为米）做掩蔽比较。这样 bmodel(fp32/fp16) 与参考
(fp32 onnx/torch，同样 518x518) 的对比口径完全一致。

与官方 640x480 目标视图 + warp 结果之间的差异，仅来自四点：输入分辨率(518^2 vs
640x480)、方块面板带来的非方形图像畸变、不做目标视角 warp、以及 depth_metric
掩码额外叠加 pred mask（官方仅用 GT 掩码）。README 中已注明。

依赖：``numpy``。后端分三种：
  - ``bmodel``：板端，import sophon.sail + 例程的 MoGe 推理（默认）。
  - ``npy``   ：离线，直接读取例程已保存的 ``results/<name>_depth.npy`` 做评测。
  - ``onnx``  ：x86，onnxruntime 跑官方静态 518x518 fp32 ONNX（参考基线）。
"""
import os
import sys
import json
import argparse
import struct
import zlib

import numpy as np

# 官方 depth_metric 指标（与 moge/test/metrics.py 一致）
EPS = 1e-6


def rel_depth(pred, gt, eps=EPS):
    return float((np.abs(pred - gt) / (gt + eps)).mean())


def delta_depth(pred, gt, threshold=1.25, eps=EPS):
    ratio = np.maximum(gt / (pred + eps), pred / (gt + eps))
    return float((ratio < threshold).mean())


# --------------------------------------------------------------------------- #
# 纯 NumPy 的 PNG 解码器（板端无 PIL）：支持 8/16 bit、灰度/RGB/RGBA、
# 5 种滤波、以及 tEXt 文本块（用于读取 depth.png 的 near/far 元数据）。
# --------------------------------------------------------------------------- #
def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png(path):
    data = open(path, 'rb').read()
    assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a PNG file: %s' % path
    pos, width, height = 8, 0, 0
    bit_depth = color_type = None
    idat = b''
    text = {}
    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        cdata = data[pos + 8:pos + 8 + length]
        if ctype == b'IHDR':
            width, height, bit_depth, color_type = struct.unpack('>IIBBBBB', cdata[:13])[0:4]
            if struct.unpack('>B', cdata[12:13])[0] != 0:
                raise ValueError('interlaced PNG is not supported')
        elif ctype == b'IDAT':
            idat += cdata
        elif ctype == b'tEXt' or ctype == b'iTXt' or ctype == b'zTXt':
            sep = cdata.index(0)
            key = cdata[:sep].decode('latin-1')
            if ctype == b'zTXt':
                val = zlib.decompress(cdata[sep + 2:]).decode('latin-1')
            else:
                val = cdata[sep + 1:].decode('latin-1')
            text[key] = val
        elif ctype == b'IEND':
            break
        pos += 12 + length

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    bpp = channels * bit_depth // 8
    stride = width * bpp
    raw = zlib.decompress(idat)

    out = bytearray()
    prev = bytearray(stride)
    p = 0
    for _ in range(height):
        ftype = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        if ftype == 1:  # Sub
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xff
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xff
        elif ftype == 3:  # Average
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xff
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                c = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(a, prev[i], c)) & 0xff
        out += line
        prev = line

    arr = np.frombuffer(bytes(out), dtype='>u2' if bit_depth == 16 else 'u1')
    if channels > 1:
        arr = arr.reshape(height, width, channels)
    else:
        arr = arr.reshape(height, width)
    return arr.astype(np.int32), text


def read_depth_png(path):
    """读取官方 depth.png（16-bit 对数编码）为 float32 米制深度，等价 moge.utils.io.read_depth。"""
    arr, text = read_png(path)
    near = float(text['near'])
    far = float(text['far'])
    mask_nan, mask_inf = arr == 0, arr == 65535
    depth = (arr.astype(np.float32) - 1.0) / 65533.0
    depth = np.power(near, 1.0 - depth) * np.power(far, depth)
    if 'unit' in text:
        depth = depth * float(text['unit'])
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    return depth


# --------------------------------------------------------------------------- #
# 双线性 resize（等价 torch F.interpolate(align_corners=False)），用于把 518x518
# 的预测图缩回 GT 分辨率。
# --------------------------------------------------------------------------- #
def bilinear_resize(src, height, width):
    src = np.asarray(src, dtype=np.float32)
    h, w = src.shape[-2], src.shape[-1]
    if h == height and w == width:
        return src
    ys = (np.arange(height, dtype=np.float32) + 0.5) * h / height - 0.5
    xs = (np.arange(width, dtype=np.float32) + 0.5) * w / width - 0.5
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)
    top = src[..., y0[:, None], x0[None, :]]
    bot = src[..., y1[:, None], x0[None, :]]
    top = top * (1 - wy)[:, None] + bot * wy[:, None]
    br_0 = src[..., y0[:, None], x1[None, :]]
    br_1 = src[..., y1[:, None], x1[None, :]]
    bot = br_0 * (1 - wy)[:, None] + br_1 * wy[:, None]
    return top * (1 - wx)[None, :] + bot * wx[None, :]


def nearest_resize_bool(mask, height, width):
    """最近邻把 bool mask 缩到 (height, width)（与 torch nearest 一致的下标规则）。"""
    h, w = mask.shape
    ys = np.minimum((np.arange(height) * h // height).astype(np.int32), h - 1)
    xs = np.minimum((np.arange(width) * w // width).astype(np.int32), w - 1)
    return mask[ys[:, None], xs[None, :]]


# --------------------------------------------------------------------------- #
# 各后端
# --------------------------------------------------------------------------- #
class BmodelBackend:
    """板端：复用例程 python/moge_bmcv.py 的 MoGe 推理（sail + 相同预处理/后处理）。"""

    def __init__(self, bmodel, dev_id):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
        import sophon.sail as sail
        from moge_bmcv import MoGe
        self.sail = sail

        class _Args:
            pass
        a = _Args()
        a.bmodel = bmodel
        a.dev_id = dev_id
        self.moge = MoGe(a)
        self.handle = sail.Handle(dev_id)

    def infer(self, image_path):
        decoder = self.sail.Decoder(image_path, True, self.moge.handle.get_device_id())
        bmimg = self.sail.BMImage()
        ret = decoder.read(self.handle, bmimg)
        if ret != 0:
            raise RuntimeError('decode failure: %s' % image_path)
        result = self.moge(bmimg)
        return result['depth'], result['mask']  # 518x518 度量深度 + mask


class NpyBackend:
    """离线：读取例程 results/<name>_depth.npy 与 <name>_mask 比对。"""

    def __init__(self, input_dir):
        self.input_dir = input_dir

    def infer(self, sample_name):
        depth = np.load(os.path.join(self.input_dir, sample_name + '_depth.npy'))
        # 例程不解出 mask 文件时退化为全有效（由 GT mask 兜底）
        mask_path = os.path.join(self.input_dir, sample_name + '_mask.npy')
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(bool)
        else:
            mask = np.ones(depth.shape, dtype=bool)
        return depth.astype(np.float32), mask


class OnnxBackend:
    """x86：onnxruntime 跑官方静态 518x518 fp32 ONNX（参考基线）。

    预处理与例程/bmodel 完全一致：RGB [0,1]，直接正方形 resize 到 518x518（不
    做任何外部归一化——ImageNet 均值/方差已内置于 DINOv2 encoder 中）。
    """

    def __init__(self, onnx_path):
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError('onnxruntime is required for --backend onnx')
        self.ort = ort
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
        from postprocess_numpy import end_to_end
        self.end_to_end = end_to_end

    def infer(self, image_path):
        import cv2
        rgb = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_LINEAR)
        x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # (1,3,518,518)
        outs = self.sess.run(None, {self.input_name: x})
        # onnx 输出顺序: points, normal, mask, metric_scale
        names = [o.name.lower() for o in self.sess.get_outputs()]
        points = outs[names.index('points')][0].astype(np.float32)
        mask = outs[names.index('mask')][0].astype(np.float32)
        metric_scale = np.asarray(outs[names.index('metric_scale')], dtype=np.float32).ravel()
        res = self.end_to_end(points, mask, metric_scale, force_projection=True)
        return res['depth'], res['mask']


# --------------------------------------------------------------------------- #
# 评测主流程
# --------------------------------------------------------------------------- #
def load_dataset(dataset):
    """读取官方数据集目录：.index.txt 内的样本子目录名列表。"""
    index = os.path.join(dataset, '.index.txt')
    names = [l.strip() for l in open(index, encoding='utf-8').read().splitlines() if l.strip()]
    return names


def evaluate(backend, dataset, benchmark='NYUv2', max_samples=None):
    names = load_dataset(dataset)
    if max_samples:
        names = names[:max_samples]

    rels, deltas = [], []
    per_sample = {}
    total = len(names)
    for i, name in enumerate(names):
        sample_dir = os.path.join(dataset, name)
        depth_gt = read_depth_png(os.path.join(sample_dir, 'depth.png'))
        valid = np.isfinite(depth_gt) & (depth_gt > 0)

        pred_depth, pred_mask = backend.infer(os.path.join(sample_dir, 'image.jpg'))

        # 预测图缩回 GT 分辨率；GT mask 之外再加 pred mask 兜底
        pred_depth = bilinear_resize(pred_depth, depth_gt.shape[0], depth_gt.shape[1])
        pred_mask = nearest_resize_bool(pred_mask.astype(bool), depth_gt.shape[0], depth_gt.shape[1])
        final_mask = valid & pred_mask

        if final_mask.sum() < 10:
            print('[%s] %s: skipped (too few valid pixels)' % (benchmark, name))
            continue

        rel = rel_depth(pred_depth[final_mask], depth_gt[final_mask])
        delta1 = delta_depth(pred_depth[final_mask], depth_gt[final_mask], 1.25)
        rels.append(rel)
        deltas.append(delta1)
        per_sample[name] = {'rel': round(rel, 6), 'delta1': round(delta1, 6)}

        if (i + 1) % 20 == 0 or i + 1 == total:
            print('[%s] %d/%d  rel=%.4f delta1=%.4f (running mean)' %
                  (benchmark, i + 1, total, float(np.mean(rels)), float(np.mean(deltas))))

    if not rels:
        print('[%s] no valid samples!' % benchmark)
        return

    result = {
        'benchmark': benchmark,
        'num_samples': len(rels),
        'depth_metric': {
            'rel': float(np.mean(rels)),
            'delta1': float(np.mean(deltas)),
        },
    }
    # 打印一行供 tools/compare_acc.py 解析
    print('[%s] depth_metric rel=%.4f delta1=%.4f' %
          (benchmark, result['depth_metric']['rel'], result['depth_metric']['delta1']))
    out_json = os.path.join(os.getcwd(), '%s_result.json' % benchmark)
    json.dump({'depth_metric': result['depth_metric'], 'per_sample': per_sample},
              open(out_json, 'w'), indent=4)
    print('[%s] result saved in %s' % (benchmark, out_json))
    return result


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--backend', type=str, default='bmodel',
                        choices=['bmodel', 'npy', 'onnx'], help='inference backend')
    parser.add_argument('--dataset', type=str, default='../datasets/NYUv2',
                        help='path of official evaluation dataset (contains .index.txt)')
    parser.add_argument('--bmodel', type=str, default='../models/BM1688/moge-2-vitl-normal_fp16_1b.bmodel',
                        help='path of bmodel (for --backend bmodel)')
    parser.add_argument('--onnx', type=str, default='../models/onnx/moge-2-vitl-normal.onnx',
                        help='path of onnx (for --backend onnx)')
    parser.add_argument('--input', type=str, default='./results',
                        help='path of results dir with *_depth.npy (for --backend npy)')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id (for --backend bmodel)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='only evaluate the first N samples (debug)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argsparser()
    if args.backend == 'bmodel':
        backend = BmodelBackend(args.bmodel, args.dev_id)
    elif args.backend == 'npy':
        backend = NpyBackend(args.input)
    else:
        backend = OnnxBackend(args.onnx)
    evaluate(backend, args.dataset, benchmark=os.path.basename(os.path.normpath(args.dataset)),
             max_samples=args.max_samples)
    print('all done.')