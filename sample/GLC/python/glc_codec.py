#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""GLC 真熵编码端到端参考实现：编码端只出 bytes，解码端只吃 bytes。

两端切分与码流协议：

    header(76B) + z_payload(14-bit 定长) + 四段 y rANS 码流

与现有 glc_infer.py 理论值路径的根本区别：码流长度是真实字节数，不再是
-log2(P) 的求和；两端之间没有任何张量传递，解码端重建 y_hat 只依赖码流。

码流协议逐字段对齐客户脚本 test_image_complement2.py 的 HEADER = ">4sBBH17I"：
    MAGIC(4s) VERSION(B) q_index(B) flags(H) 15 个维度(I) 5 个码流长度(I) 保留(I)
"""

import math
import struct

import numpy as np

import entropy_numpy as en
from rans_coder import RansCoder

# torch 只被主机侧的模型前向路径（encode/decode/gaussian_bits）用到。
# 设备端只用 EntropyContext + compact_masked/scatter_masked，不应被迫安装 torch，
# 所以这里改成惰性导入。

def _torch():
    """惰性导入 torch。

    只有主机侧的模型前向路径需要它；设备端加载 CDF 资产、做 build_indexes、
    跑 rANS 编解码都不该依赖 torch。
    """
    import torch

    return torch


SCALE_MIN = 0.11
SCALE_MAX = 256.0
SCALE_LEVELS = 64
Z_INDEX_BITS = 14
MAGIC = b"GLC2"
VERSION = 2
HEADER = struct.Struct(">4sBBH17I")

# 逐子图精度档案（编解码共用；两端必须取同一个）。
#
# 跨芯片（BM1684X 与 BM1688）部署时不能整体用 BF16：两芯片的 BF16 先验子图输出
# 并不逐位相同，实测约 1.27% 的 scale 落到不同 CDF 档位，rANS 随即失步、
# 码流互不可解。但并非所有子图都需要逐位一致：
#
#   prior_init / stage_1..3  -> 输出 scales（选 CDF 行）、means（重建 y_hat）、
#                               common（喂下一段）、q_dec（缩放 y_hat），且本段
#                               重建的 y_hat 又是下一段的输入，误差逐段累积。
#                               这四个子图必须 FP32。
#   encoder                  -> 只决定写进码流的符号，解码端照着符号重建，
#                               其精度不参与可解性。
#   tail                     -> 只决定最终重建图的样子，同样不影响可解性。
#
# 实测：prior_init / stage_1..3 用 FP32 时两芯片输出 sha256 完全相同；
# encoder / tail 留 BF16 不影响互解，而这两个恰是耗时大头。
MIXED = "mixed"
PRECISION_PROFILES = {
    "fp32":  {"encoder": "fp32", "prior_init": "fp32", "stage": "fp32", "tail": "fp32"},
    "bf16":  {"encoder": "bf16", "prior_init": "bf16", "stage": "bf16", "tail": "bf16"},
    MIXED:   {"encoder": "bf16", "prior_init": "fp32", "stage": "fp32", "tail": "bf16"},
}


def resolve_precision(precision):
    """把 --precision 展开成逐子图精度档案。"""
    if precision not in PRECISION_PROFILES:
        raise ValueError("未知精度 %s，可选：%s"
                         % (precision, ", ".join(sorted(PRECISION_PROFILES))))
    return PRECISION_PROFILES[precision]


class EntropyContext:
    """compressai 预算好的 CDF 资产，设备端只读不重算。

    CDF 只依赖固定 64 档 scale_table、与图像无关，
    所以在主机用 compressai 生成一次、存成 npz 随模型资产下发，
    设备端就彻底绕开了 compressai 与 scipy 依赖。
    """

    def __init__(self, quantized_cdf, cdf_length, offset, scale_table):
        q = np.asarray(quantized_cdf, dtype=np.int32)
        lengths = np.asarray(cdf_length, dtype=np.int32).reshape(-1)
        offsets = np.asarray(offset, dtype=np.int32).reshape(-1)
        # _quantized_cdf 每行宽度是 max_length+2，必须按 _cdf_length 裁掉填充，
        # 否则解码端会在零填充上做无谓比较，也可能越出有效 CDF 区间。
        rows = [q[i, : int(lengths[i])] for i in range(q.shape[0])]
        self.flat = np.concatenate(rows).astype(np.int32)
        starts = np.zeros(len(rows), dtype=np.int32)
        acc = 0
        for i, r in enumerate(rows):
            starts[i] = acc
            acc += r.size
        self.starts = starts
        self.lengths = lengths
        self.offsets = offsets
        self.scale_table = np.asarray(scale_table, dtype=np.float32).reshape(-1)
        if self.scale_table.size != q.shape[0]:
            raise ValueError(
                f"scale_table 档数 {self.scale_table.size} 与 CDF 行数 {q.shape[0]} 不符"
            )

    @classmethod
    def from_compressai(cls, scale_table=None, tail_mass=1e-9):
        from compressai.entropy_models import GaussianConditional

        import torch

        if scale_table is None:
            # 必须与客户 compressai 脚本同源：torch.exp(torch.linspace(...))。
            # numpy 版有 46/64 档末位不同，会让 CDF 与客户不一致、码流无法互解。
            import math as _math

            st = torch.exp(torch.linspace(
                _math.log(en.SCALE_MIN), _math.log(en.SCALE_MAX),
                en.SCALE_LEVELS)).numpy().astype(np.float32)
        else:
            st = np.asarray(scale_table, dtype=np.float32)
        model = GaussianConditional(None, tail_mass=tail_mass)
        model.update_scale_table(torch.from_numpy(np.asarray(st, dtype=np.float32)),
                                 force=True)
        return cls(model._quantized_cdf.numpy(), model._cdf_length.numpy(),
                   model._offset.numpy(), st)

    def save(self, path):
        np.savez(path, flat=self.flat, starts=self.starts, lengths=self.lengths,
                 offsets=self.offsets, scale_table=self.scale_table)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls.__new__(cls)._init_from(z)

    def _init_from(self, z):
        self.flat, self.starts = z["flat"], z["starts"]
        self.lengths, self.offsets = z["lengths"], z["offsets"]
        self.scale_table = z["scale_table"]
        return self


def _indexes_for(scales_np, entropy):
    """clamp_min(SCALE_MIN) 后按 64 档表取 CDF 行号（静态查询表）。"""
    return en.build_indexes(np.maximum(scales_np, SCALE_MIN), entropy.scale_table)


def _encode_stage(coder, entropy, y_scaled, scales, means, mask):
    """编码一段：掩码取 1/4 元素 -> build_indexes -> rANS 编码。

    返回 (payload, 重建值张量)。重建值由「解出的符号 + mean」得到，
    这正是解码端会算出的同一结果，是两端 y_hat 逐位一致的基础。
    """
    y_values = compact_masked(y_scaled, mask)
    stage_scales = compact_masked(scales, mask)
    stage_means = compact_masked(means, mask)

    indexes = _indexes_for(stage_scales.numpy(), entropy)
    symbols = en.quantize(y_values.numpy(), stage_means.numpy())
    payload = coder.encode(symbols.reshape(-1), indexes.reshape(-1), entropy.flat,
                           entropy.lengths, entropy.starts, entropy.offsets)
    return payload, symbols, stage_means, stage_scales


def _decode_stage(coder, entropy, payload, scales, means, mask, count):
    """解码一段：用码流里的符号 + 该段 means 还原出这一段的 y 值。"""
    stage_scales = compact_masked(scales, mask)
    stage_means = compact_masked(means, mask)
    if stage_means.numel() != count:
        raise ValueError(f"该段符号数 {count} 与 means 元素数 {stage_means.numel()} 不符")

    indexes = _indexes_for(stage_scales.numpy(), entropy)
    symbols = coder.decode(payload, count, indexes.reshape(-1), entropy.flat,
                           entropy.lengths, entropy.starts, entropy.offsets)
    values = en.dequantize(symbols, stage_means.numpy().reshape(-1))
    return values.reshape(1, *stage_means.shape[1:])


def compact_masked(tensor, mask):
    """按四段掩码取出该段的 1/4 元素，压成 (1, C, H, W//4)。

    W=48 时每行恰好取一半，压完宽度 24；与客户脚本 compact_masked (:143) 同构。
    """
    batch, channels, height, width = tensor.shape
    if batch != 1:
        raise ValueError("码流编解码目前只支持 batch size 1")
    selected = tensor[mask.expand_as(tensor).bool()]
    expected = channels * height * width // 4
    if selected.numel() != expected:
        raise ValueError("四段掩码选中的元素数不符合预期")
    return selected.reshape(1, channels, height, width // 4)


def scatter_masked(compact, mask, shape):
    """compact_masked 的逆操作：把该段元素放回原张量的对应位置。"""
    output = compact.new_zeros(shape)
    output[mask.expand(shape).bool()] = compact.reshape(-1)
    return output


def _separate_prior(params):
    """与 common_model.separate_prior (:56) 对齐：quant_step 先 clamp_min(0.5)。"""
    quant_step, scales, means = params.chunk(3, 1)
    quant_step = quant_step.clamp_min(0.5)
    return 1.0 / quant_step, quant_step, scales, means


def _prior_params(net, z_hat):
    params = net.y_prior_fusion(net.hyper_dec(z_hat))
    _, q_dec, scales, means = _separate_prior(params)
    common = net.y_spatial_prior_reduction(params)
    return q_dec, scales, means, common


def _stage_params(net, stage, y_hat_so_far, common, scales, means):
    """第 0 段直接用 prior_init 的输出；第 N 段走该段 adaptor（客户脚本 :131）。"""
    if stage == 0:
        return scales, means
    torch = _torch()
    adaptors = (net.y_spatial_prior_adaptor_1, net.y_spatial_prior_adaptor_2,
                net.y_spatial_prior_adaptor_3)
    inputs = torch.cat((y_hat_so_far, common), dim=1)
    return net.y_spatial_prior(adaptors[stage - 1](inputs)).chunk(2, 1)


def encode(coder, entropy, net, z_quantizer, x, q_index, original_size):
    """编码端：图 -> 码流 bytes。只额外返回 y_hat 供往返比对，不传给解码端。"""
    torch = _torch()
    with torch.no_grad():
        return _encode_impl(coder, entropy, net, z_quantizer, x, q_index,
                            original_size)


def _encode_impl(coder, entropy, net, z_quantizer, x, q_index, original_size):
    torch = _torch()
    if not 0 <= q_index < net.q_enc.shape[0]:
        raise ValueError(f"q_index {q_index} 超出已加载模型范围")
    q_enc = net.q_enc[q_index:q_index + 1]

    y = net.enc(net.vqgan.encoder(x), q_enc)
    z = net.hyper_enc(y)

    z_indices = z_quantizer.get_indices(z).reshape(-1).cpu().numpy().astype(np.int32)
    z_payload = en.pack_fixed_width(z_indices.tolist(), Z_INDEX_BITS)
    idx_t = torch.from_numpy(z_indices.astype(np.int64)).reshape(-1, 1)
    z_hat = z_quantizer.get_quan_feat(
        idx_t, (z.shape[0], z.shape[2], z.shape[3], z.shape[1])
    )

    params = net.y_prior_fusion(net.hyper_dec(z_hat))
    q_enc_prior, q_dec_prior, scales, means = _separate_prior(params)
    common = net.y_spatial_prior_reduction(params)
    y_scaled = y * q_enc_prior

    masks = net.get_mask_four_parts(*y.shape, y.dtype, y.device)
    y_hat_so_far = torch.zeros_like(y)
    streams = []
    theory = []          # 与真实码流并排的理论值累计，同一套逐段参数
    for stage, mask in enumerate(masks):
        stage_scales, stage_means = _stage_params(
            net, stage, y_hat_so_far, common, scales, means
        )
        payload, symbols, packed_means, packed_scales = _encode_stage(
            coder, entropy, y_scaled, stage_scales, stage_means, mask
        )
        streams.append(payload)
        theory.append(gaussian_bits(symbols, packed_scales.numpy()))
        recon = en.dequantize(
            symbols.reshape(-1), packed_means.numpy().reshape(-1)
        ).reshape(1, *packed_means.shape[1:])
        y_hat_so_far = y_hat_so_far + scatter_masked(
            torch.from_numpy(recon).to(y.dtype), mask, y.shape
        )

    # 与官方 forward_four_part_prior 同口径（common_model.py:225）：y_hat 乘的是
    # 先验网络输出的 q_dec，不是 net.q_dec 权重；后者只在 dec() 重建时才用。
    y_hat = y_hat_so_far * q_dec_prior

    height, width = original_size
    header = HEADER.pack(
        MAGIC, VERSION, q_index, 0, height, width, x.shape[2], x.shape[3],
        y.shape[1], y.shape[2], y.shape[3], z.shape[1], z.shape[2], z.shape[3],
        len(z_indices), len(z_payload), *(len(s) for s in streams), 0,
    )
    info = {
        "theory_y_bits": float(sum(theory)),
        "codebook_size": net.codebook_size,
        "z_elements": int(z_indices.size),
    }
    return header + z_payload + b"".join(streams), y_hat, info


def gaussian_bits(symbols, scales):
    """现有例程 glc_infer.py:108 / 官方 get_y_gaussian_bits 的理论值口径。

    对**残差**求 -log2(P(y-0.5 < Y < y+0.5))，零均值、sigma = scales，
    对应官方 `get_y_gaussian_bits(y_q, scales_hat)` 里的 mu = zeros_like(sigma)。
    这是信息论下界而非真实码流长度，与真实 bpp 并排只为给出差距归因。
    """
    import torch
    import torch.distributions as td

    sigma = torch.from_numpy(np.asarray(scales, dtype=np.float64).reshape(-1))
    sigma = sigma.clamp(1e-5, 1e10)
    values = torch.from_numpy(np.asarray(symbols, dtype=np.float64).reshape(-1))
    dist = td.Normal(torch.zeros_like(sigma), sigma)
    probs = dist.cdf(values + 0.5) - dist.cdf(values - 0.5)
    bits = (-torch.log(probs + 1e-5) / math.log(2.0)).clamp_min(0)
    return float(bits.sum().item())


def parse_bitstream(payload):
    """解析码流头，返回 (q_index, dimensions, streams)。"""
    if len(payload) < HEADER.size:
        raise ValueError("GLC 码流头被截断")
    fields = HEADER.unpack_from(payload)
    if fields[0] != MAGIC or fields[1] != VERSION:
        raise ValueError("不支持的 GLC 码流")
    if fields[3] != 0 or fields[20] != 0:
        raise ValueError("不支持的 GLC 码流标志位")
    q_index = fields[2]
    dimensions = fields[4:15]
    stream_lengths = fields[15:20]
    if any(length <= 0 for length in stream_lengths):
        raise ValueError("GLC 码流长度字段非法")
    cursor = HEADER.size
    streams = []
    for length in stream_lengths:
        end = cursor + length
        if end > len(payload):
            raise ValueError("GLC 码流载荷被截断")
        streams.append(payload[cursor:end])
        cursor = end
    if cursor != len(payload):
        raise ValueError("GLC 码流尾部有多余数据")
    return q_index, dimensions, streams


def decode(coder, entropy, net, z_quantizer, payload):
    """解码端：码流 bytes -> x_hat, y_hat。全程不接触编码端的任何张量。"""
    torch = _torch()
    with torch.no_grad():
        return _decode_impl(coder, entropy, net, z_quantizer, payload)


def _decode_impl(coder, entropy, net, z_quantizer, payload):
    torch = _torch()
    q_index, dimensions, streams = parse_bitstream(payload)
    (orig_h, orig_w, padded_h, padded_w, y_channels, y_height, y_width,
     z_channels, z_height, z_width, z_count) = dimensions
    z_payload, *y_streams = streams
    if z_count != z_height * z_width:
        raise ValueError("超先验索引数非法")
    if not 0 <= q_index < net.q_dec.shape[0]:
        raise ValueError(f"q_index {q_index} 超出已加载模型范围")

    device = next(net.parameters()).device
    z_indices = en.unpack_fixed_width(z_payload, z_count, Z_INDEX_BITS)
    idx_t = torch.tensor(z_indices, dtype=torch.long, device=device).reshape(-1, 1)
    z_hat = z_quantizer.get_quan_feat(idx_t, (1, z_height, z_width, z_channels))

    q_dec_p, scales, means, common = _prior_params(net, z_hat)

    y_shape = (1, y_channels, y_height, y_width)
    masks = net.get_mask_four_parts(*y_shape, params_dtype(net), params_device(net))
    y_hat_so_far = torch.zeros(y_shape, dtype=params_dtype(net), device=params_device(net))

    for stage, (mask, stream) in enumerate(zip(masks, y_streams)):
        stage_scales, stage_means = _stage_params(
            net, stage, y_hat_so_far, common, scales, means
        )
        count = y_channels * y_height * y_width // 4
        values = _decode_stage(coder, entropy, stream, stage_scales, stage_means,
                               mask, count)
        y_hat_so_far = y_hat_so_far + scatter_masked(
            torch.from_numpy(values).to(y_hat_so_far.dtype), mask, y_shape
        )

    y_hat = y_hat_so_far * q_dec_p
    q_dec = net.q_dec[q_index:q_index + 1]
    x_hat = net.vqgan.generator(net.dec(y_hat, q_dec)).clamp(-1, 1)
    if tuple(x_hat.shape[-2:]) != (padded_h, padded_w):
        raise ValueError(
            f"解码输出尺寸 {tuple(x_hat.shape[-2:])} 与码流记录 {(padded_h, padded_w)} 不符"
        )
    return x_hat[:, :, :orig_h, :orig_w], y_hat


def params_dtype(net):
    return next(net.parameters()).dtype


def params_device(net):
    return next(net.parameters()).device


def bpp_breakdown(payload, height, width):
    """真实码流的分项 bpp：header / z / y 各自占比与字节数。"""
    _, dimensions, streams = parse_bitstream(payload)
    z_payload, *y_streams = streams
    px = height * width
    return {
        "total": len(payload) * 8.0 / px,
        "header": HEADER.size * 8.0 / px,
        "z": len(z_payload) * 8.0 / px,
        "y": sum(len(s) for s in y_streams) * 8.0 / px,
        "header_bytes": HEADER.size,
        "z_bytes": len(z_payload),
        "y_bytes": [len(s) for s in y_streams],
        "y_shape": list(dimensions[4:7]),
        "z_shape": list(dimensions[7:10]),
    }
