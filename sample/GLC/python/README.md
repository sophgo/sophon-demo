# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 SoC平台](#11-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 测试图片](#22-测试图片)
* [3. 精度与性能测试](#3-精度与性能测试)
* [4. 真熵编码：压缩端与还原端分离部署](#4-真熵编码压缩端与还原端分离部署)
    * [4.1 为什么需要它](#41-为什么需要它)
    * [4.2 运行步骤](#42-运行步骤)
    * [4.3 两端口径必须一致（重要）](#43-两端口径必须一致重要)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 | Python例程 | 说明 |
| ---- | ---------- | ---- |
| 1 | glc_infer.py | 单张图片推理，打印bpp/PSNR/SSIM与分阶段耗时（bpp 为理论下界估算） |
| 2 | glc_eval.py | 对整个图片文件夹批量评估，输出JSON |
| 3 | bmrt.py | libbmrt的ctypes绑定，供上面两个例程调用 |
| 4 | metrics.py | PSNR / SSIM实现，主机与设备共用 |
| 5 | glc_codec.py | 真熵编码的编解码分离实现（rANS 码流 + 四段先验） |
| 6 | rans_coder.py | rANS 纯 C ABI 的 ctypes 绑定，设备端只需 numpy |
| 7 | entropy_numpy.py | entropy model 层的 numpy 重写（无 torch / scipy 依赖） |
| 8 | model_loader.py | 加载 GLC 权重，仅供主机侧工具使用 |
| 9 | glc_encode.py | **压缩端**：图 -> 真熵编码码流（.glc） |
| 10 | glc_decode.py | **还原端**：真熵编码码流（.glc）-> 重建图 |

## 1. 环境准备

### 1.1 SoC平台

如果您使用SoC平台（如SE系列边缘设备），刷机后在`/opt/sophon/`下已经预装了libsophon运行库包。

**本例程不需要sophon-sail。** SE7/SE9的OTA出厂镜像只带`libsophon`（含`libbmrt`/`libbmlib`），未预装`sophon-sail`，且它不在公共PyPI上。`bmrt.py`用ctypes直接调用`libbmrt`，提供与`sail.Engine`等价的推理能力，因此设备端只需要两个pip包：

```bash
pip3 install numpy pillow
export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH
```

`bmrt.py`在导入时会校验`bm_device_mem_t`等结构体的尺寸与当前libsophon头文件是否一致，不符会直接报错而不是segfault。

## 2. 推理测试

python例程不需要编译，可以直接运行。
### 2.1 参数说明

glc_infer.py：

```bash
usage: glc_infer.py [--models MODELS] [--precision {fp32,bf16,fp16,int8}]
                    --input INPUT [--q_index {0,1,2,3}] [--dev_id DEV_ID]
                    [--loops LOOPS] [--out OUT]
--models:    模型与资产目录，默认../models/BM1684X（SE9-16 用 ../models/BM1688）；
--precision: 推理精度，默认fp32；
--input:     单张测试图片路径；
--q_index:   质量档位0~3，越大码率越高，默认0；
--dev_id:    用于推理的tpu设备id；
--loops:     重复推理次数（性能测试用），首帧自动预热，默认1；
--out:       重建图输出路径，可选。
```

glc_eval.py：

```bash
usage: glc_eval.py [--models MODELS] [--precision {fp32,bf16,fp16,int8}]
                   --data DATA --out OUT [--q-indexes Q [Q ...]]
                   [--limit LIMIT] [--loops LOOPS] [--dev_id DEV_ID]
--data:       测试图片文件夹路径；
--out:        结果JSON输出路径；
--q-indexes:  要评估的质量档位，默认0 1 2 3；
--limit:      只评估前N张，0表示全部；
--loops:      每张图计时循环次数，默认5。
```

### 2.2 测试图片

单张图片测试：

```bash
python3 python/glc_infer.py --precision bf16 \
    --input datasets/kodak24/kodim01.png --q_index 0 --loops 3
```

测试结束会打印bpp、PSNR/SSIM与分阶段耗时，例如：

```
精度: bf16 | 质量档位: q0 | 输入: kodim01.png 768x512 (pad 后 768x512)
  bpp          : 0.0268  (bit_y=9202, bit_z=1344, 8x12 码本索引)
  PSNR / SSIM  : 18.93 dB / 0.3354
  耗时 (ms)    : encoder 232.2 | vq 77.5 | decoder 398.2 | 合计 708.0
  吞吐         : 1.41 FPS
```

整个文件夹批量测试：

```bash
python3 python/glc_eval.py --precision bf16 \
    --data datasets/kodak24 --out out/eval_bf16.json --loops 3
```

> **注意**：bmodel是静态shape，只支持512x768。Kodak24中有6张是竖构图（768x512），会被自动跳过并在日志中列出。

## 3. 精度与性能测试

参考[主README的第6、7节](../README.md#6-精度测试)查看完整的精度与性能对比表。本节只说明如何复现。

精度测试需要先得到 PyTorch 基线（在主机上跑，需要 torch）：

```bash
PYTHONPATH=tools/stubs python3 tools/pytorch_baseline.py \
    --ckpt models/GLC_image.pth.tar --glc-src <GLC源码> \
    --images datasets/kodak24 --out out/baseline_kodak24.json
```

在设备上跑评估，得到结果 JSON：

```bash
python3 python/glc_eval.py --precision bf16 \
    --data datasets/kodak24 --out out/eval_bf16.json --loops 3
```

再用 `tools/eval_accuracy.py` 把设备结果与基线对比（在主机上跑）：

```bash
python3 tools/eval_accuracy.py --baseline out/baseline_kodak24.json \
    --device out/eval_bf16.json
```

> **注意**：基线是全量 24 张图的结果，而设备评估会自动跳过 6 张尺寸不匹配的竖构图，
> `eval_accuracy.py` 会在两者的公共图集上重算平均值后再比较，避免口径不一致。

性能测试直接看 `glc_infer.py` / `glc_eval.py` 打印的分阶段耗时。


## 4. 真熵编码：压缩端与还原端分离部署

`glc_infer.py` / `glc_eval.py` 走的是「理论值路径」：编解码在同一进程里，中间直接传
`y` 张量，bpp 按高斯熵模型就地估算，是**信息论下界**而非真实码流长度。

真实场景（压缩端与还原端分处两地、码流过无线信道）需要的是：压缩端只产出码流文件，
还原端只读该文件。这就是 `glc_encode.py` 与 `glc_decode.py`。

两个脚本**没有任何共享状态**：编码端写出的 `.glc` 文件是唯一的接口。解码端所需的一切
尺寸信息（原图尺寸、y/z 的通道数与空间尺寸）都从码流头读取，不依赖任何配置文件。

### 4.1 运行步骤

前提：已按第 1 节装好 numpy/pillow，并已按主 README 第 3 节跑过 `scripts/download.sh`。

`download.sh` 会把下列文件拉到 `models/BM1684X/` 与 `models/BM1688/`（无需自行编译；SE7-32 用前者，SE9-16 用后者）：

    glc_encoder_bf16.bmodel      glc_prior_init_bf16.bmodel
    glc_stage_1_bf16.bmodel      glc_stage_2_bf16.bmodel
    glc_stage_3_bf16.bmodel      glc_tail_bf16.bmodel
    codebook.npy  q_enc.npy  q_dec.npy  entropy_assets.npz

真熵编码用的 rANS 是纯 C 实现，**设备和主机第一次使用前都要编译一次**：

```bash
bash tools/rans/build.sh
```

脚本会同时编出 `libglc_rans_x86_64.so` 与 `libglc_rans_aarch64.so`，并按当前
架构挑一个加载；两个架构跑同一份 C 冒烟测试，比对码流逐字节一致。

其中 `entropy_assets.npz` 是熵编码静态资产（CDF 表），由主机侧生成一次
（设备端只读，不需要 torch/compressai）：

```bash
PYTHONPATH=tools/vendor:. python3 tools/make_entropy_assets.py \
    --out models/BM1684X/entropy_assets.npz
```

**压缩端**（只产出码流）：

```bash
python3 python/glc_encode.py --precision bf16 \
    --input datasets/kodak24/kodim01.png --out /tmp/out.glc --q_index 0
```

预期输出：

```
输入 kodim01.png  768x512 (pad 后 768x512)
码流 1416 字节 -> /tmp/out.glc
  真实 bpp 0.028809 = y 0.023844 + z 0.003418 + header 0.001546
  四段 y 字节数 [332, 324, 268, 248]
编码耗时 516.3 ms
```

**还原端**（只吃码流，可以换一台机器跑）：

```bash
python3 python/glc_decode.py --precision bf16 \
    --input /tmp/out.glc --out /tmp/recon.png
```

预期输出：

```
码流 out.glc  1416 字节
  头部记录：q_index=0  原图 512x768  pad 后 512x768
  真实 bpp 0.028809 = y 0.023844 + z 0.003418 + header 0.001546
重建图 (1, 3, 512, 768)  范围 [-1.000, 1.000]
解码耗时 545.8 ms
已写出 /tmp/recon.png
```

两端可以分处两台机器：把 `/tmp/out.glc` 拷过去即可，还原端不需要任何编码端的状态。

### 4.2 两端口径必须一致（重要）

**编码端与还原端必须使用同一精度的模型。** 用 `--precision bf16` 编码的码流，还原端
也必须是 `--precision bf16`；混用 fp32 会失败。

原因是熵模型：scale 要先按 64 档表查到 CDF 行号，而 BF16 与 FP32 算出的 scale 会有
微小差异，实测约 **1.2%** 的 scale 落在不同的档位上。rANS 是算术编码，码流一旦与 CDF
对不上，**解码不会报错**，而是一路解出错误符号，最终产出退化图像。

还原端已内置自检：解出的 `y_hat` 若含 NaN/Inf、量级异常，或重建图退化成常量，会直接
报错并提示检查两端 `--precision`，不会静默交出坏图。

混用精度的实际表现（已实测）：

```
$ python3 python/glc_decode.py --precision bf16 --input host_fp32.glc
ValueError: 解码出的 y_hat 含 NaN/Inf：码流与当前 --precision 模型不匹配
```

### 4.2.1 跨芯片部署用 `--precision mixed`

SE7-32（BM1684X）与 SE9-16（BM1688）**不能用同一份 BF16 码流互解**——两芯片的 BF16
先验子图输出不逐位相同，rANS 会失步。

`--precision mixed` 让「决定 CDF 与 y_hat」的熵模型子图走 FP32、其余留 BF16，
既保证互解又只比纯 BF16 慢约 10%：

| 子图 | mixed 下精度 | 原因 |
|------|-------------|------|
| `glc_prior_init` / `glc_stage_1..3` | FP32 | 输出 scales/means/common/q_dec，决定 CDF 与 y_hat |
| `glc_encoder` / `glc_tail` | BF16 | 不参与可解性，且是耗时大头 |

两芯片双向实测均通过（两端 `y_hat` sha256 相同）。跨芯片部署脚本：

```bash
# 压缩端（SE7-32，在样例根目录执行）
python3 python/glc_encode.py --precision mixed --models models/BM1684X \
    --input datasets/kodak24/kodim01.png --out out.glc
# 还原端（SE9-16，把 out.glc 拷过去，在样例根目录执行）
python3 python/glc_decode.py --precision mixed --models models/BM1688 --input out.glc
```

> `mixed` 所需的 FP32 子图（`glc_prior_init` / `glc_stage_1..3` / `glc_tail`）已随
> 下载包下发，无需自行编译。`--models` 传的是**相对当前工作目录**的路径，上面示例均
> 以样例根目录为工作目录；不传 `--models` 时默认取 `python/../models/BM1684X`
> （按脚本自身位置解析，与工作目录无关），跨芯片部署时会指向错误的芯片目录，
> 报 `bmodel 不存在` 或加载到不匹配的 bmodel，因此跨芯片时两端都要显式传 `--models`。

### 4.3 两端分别需要哪些文件

部署时不需要把整套模型发给两端，各自只带自己要的：

| | 压缩端需要 | 还原端需要 |
|---|---|---|
| bmodel | `glc_encoder`、`glc_prior_init`、`glc_stage_1..3` | `glc_prior_init`、`glc_stage_1..3`、`glc_tail` |
| 资产 | `codebook.npy`、`q_enc.npy`、`entropy_assets.npz` | `codebook.npy`、`q_dec.npy`、`entropy_assets.npz` |
| 依赖 | numpy + pillow | numpy + pillow |

`glc_prior_init` 与 `glc_stage_1..3` 两端都需要（都要逐段算 scales/means），
`glc_encoder` 只有压缩端用，`glc_tail` 只有还原端用。

### 4.4 测试套件

`tools/tests/` 下的用例按依赖分两类——**设备端只跑得动第一个**：

| 套件 | 跑在哪 | 依赖 | 覆盖内容 |
|------|--------|------|----------|
| `test_bitstream_protocol` | 设备或主机 | numpy | 码流头解析、截断/篡改/flags 拒绝 |
| `test_precision_profiles` | 设备或主机 | numpy | 逐子图精度档案（mixed 的切分规则） |
| `test_rans_parity` | 仅主机 | compressai | rANS 与官方字节级一致（含客户 `compress()` 入口） |
| `test_mask_parity` | 仅主机 | torch | 四段掩码与官方 torch 版逐位一致 |
| `test_param_alignment` | 仅主机 | compressai | 按客户样例参数对齐、参数敏感性 |
| `test_prior_stages` | 仅主机 | torch + 权重 | prior 四段 scales/means 与官方逐段一致 |
| `test_split_equivalence` | 仅主机 | torch + 权重 | 边界重切语义不变 |
| `test_roundtrip` | 仅主机 | torch + 权重 | 真实图端到端往返 + 真实/理论 bpp |
| `test_onnx_alignment` | 仅主机 | onnxruntime | ONNX 子图与 PyTorch 逐张量对齐 |
| `test_bmodel_accuracy` | 仅主机 | tpu-mlir cmodel | bmodel 数值与 PyTorch 对齐 |
| `test_e2e_device` | 仅真机（SE7/SE9） | bmodel + numpy | 真机端到端：bmodel 编码 -> 码流 -> bmodel 解码 |

主机侧统一用：

```bash
PYTHONPATH=tools/vendor:python:. python3 tools/tests/<套件>.py
```

（`tools/vendor` 是 compressai 最小子集，见 `docs/GLC_Entropy_Coding.md` 第 5 节。）
设备端只有 numpy，因此只跑得动纯 numpy 的三个：`test_bitstream_protocol`、
`test_precision_profiles` 与 `test_e2e_device`（后者还需要 bmodel）。

`test_e2e_device` 默认按 `GLC_CHIP`（默认 `BM1684X`）到 `models/<CHIP>/` 找模型，
SE9-16 上要传 `GLC_CHIP=BM1688`；也可以直接传图片路径，让它用真实图跑：

```bash
GLC_CHIP=BM1688 python3 tools/tests/test_e2e_device.py datasets/kodak24/kodim01.png
```
