# TAPNext++

## 目录

- [TAPNext++](#tapnext)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 模型编译](#32-模型编译)
  - [4. 例程测试](#4-例程测试)
  - [5. 精度测试](#5-精度测试)
    - [5.1 测试方法](#51-测试方法)
    - [5.2 测试结果](#52-测试结果)
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt\_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  - [8. FAQ](#8-faq)

## 1. 简介
TAPNext++（Tracking Any Point）是 Google DeepMind 提出的新一代密集点跟踪模型，出自 [tapnet](https://github.com/google-deepmind/tapnet) 仓库。模型基于 ViT backbone（12 层 TRecViTBlock，width=768）+ RG-LRU/Conv1D 循环状态空间架构，可在视频中跟踪任意查询点随时间的轨迹并预测其可见性，支持长时序、遮挡场景下的鲁棒点跟踪。

与单目标跟踪（如 FearTracker）不同，TAPNext++ 跟踪的是**任意像素点**而非边界框，且模型内部维护跨帧的循环状态（12 层 × (rg_lru_state, conv1d_state) = 24 个 cache 张量），使其具备时序记忆能力。

本例程对 TAPNext++ 的 PyTorch 实现进行移植，将两图循环架构导出为 ONNX 并编译为 BModel，使之能在 SOPHON BM1688 上进行推理测试。

## 2. 特性

### 2.1 目录结构说明
```bash
├── cpp                   # 存放C++例程及其README
│   └── tapnext_bmcv      # C++例程（BMRT + BMCV）
├── docs                  # 存放本例程专用文档
├── python                # 存放Python例程及其README
│   ├── README.md
│   └── tapnext_infer.py  # Python例程（SAIL）
├── models                # 存放download.sh下载的模型（ONNX / BModel）
├── datasets              # 存放download.sh下载的测试数据
├── scripts               # 存放模型编译、数据下载等shell脚本
├── tools                 # 存放ONNX导出、量化校准等python脚本
└── README.md             # 本例程的中文指南
```

### 2.2 SDK特性
* 支持 BM1688（SoC，如 SE9），支持 2-core 推理
* 支持 FP16 模型编译和推理（**生产精度**）
* 支持基于 SAIL 的 Python 推理和基于 BMRT/BMCV 的 C++ 推理
* 支持视频文件的任意点跟踪，输出逐帧轨迹与可见性
* 支持自定义查询点（`--query` / `--query_file`）

## 3. 数据准备与模型编译

### 3.1 数据准备

本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

```bash
chmod -R +x scripts/
# 下载 BM1688 FP16 BModel（含 2-core 变体）+ 测试视频（推荐，直接可运行）
./scripts/download.sh --BM1688 --dataset
```

`download.sh`参数如下：
```bash
--BM1688   # 下载BM1688的FP16 BModel（含2-core变体）
--onnx     # 下载导出的ONNX（用于重新编译BModel）
--ckpt     # 下载原始PyTorch checkpoint（~2.4 GB，仅从零重新导出ONNX时需要）
--dataset  # 下载测试视频
--all      # 下载全部
```

下载的模型包括：
```bash
models/
├── onnx
│   ├── tapnext_init.onnx                     # 导出的 init 图 ONNX（2 输入）
│   └── tapnext_step.onnx                     # 导出的 step 图 ONNX（27 输入）
└── BM1688
    ├── tapnext_init_fp16_1b.bmodel           # BM1688 FP16 init BModel，1-core
    ├── tapnext_init_fp16_1b_2core.bmodel     # BM1688 FP16 init BModel，2-core
    ├── tapnext_step_fp16_1b.bmodel           # BM1688 FP16 step BModel，1-core
    └── tapnext_step_fp16_1b_2core.bmodel     # BM1688 FP16 step BModel，2-core
```

下载的数据包括：
```bash
./datasets
├── test.mp4                                  # 测试用视频文件
└── test_se9.mp4                              # SE9 适配的短测试视频
```

> **注:** bmodel 和测试视频体积较大，未纳入 git 管理，由 `scripts/download.sh` 或 [3.2 模型编译](#32-模型编译) 生成。

### 3.2 模型编译

**如果您不编译模型，只想直接使用下载的数据和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，编译前要先将 PyTorch 模型导出为 ONNX。TAPNext++ 采用**两图循环架构**，需导出两个静态形状的 ONNX 图：

| 图 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| init | frame `[1,3,256,256]` + query_points `[1,Q,3]` | tracks + vis + 24 caches | 处理第 0 帧，初始化循环状态 |
| step | frame + step + query_points + 24 caches（共 27 输入） | tracks + vis + 24 caches | 处理后续帧，cache 回馈 |

- 导出 ONNX

本例程在 `tools` 目录下提供了模型导出脚本 `export_onnx.py` 和校准数据生成脚本 `gen_cali_data.py`。在 TPU-MLIR 环境中执行：

```bash
cd tools
python3 export_onnx.py --ckpt ../models/tapnextpp_ckpt.pt --outdir ../models/onnx
```

导出的 ONNX 文件位于 `models/onnx/`。POC 静态形状为 256×256、Q=1。

- 编译 BModel

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel，目标平台为 BM1688。

生成 FP16 BModel（**生产精度**）：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1688     # 生成 BM1688 FP16 BModel（含 2-core 变体）
```

执行上述命令会在 `models/BM1688/` 下生成 `tapnext_init_fp16_1b.bmodel` 和 `tapnext_step_fp16_1b.bmodel`，以及 `_2core` 变体用于双核推理。

生成 INT8 BModel（可选，**不推荐**）：

```bash
./scripts/gen_int8bmodel_mlir.sh bm1688
```

> **警告:** INT8 在本模型上不可用，详见 [5.2 测试结果](#52-测试结果)。生产环境请使用 FP16。

## 4. 例程测试
- [Python 例程](./python/README.md)
- [C++ 例程](./cpp/tapnext_bmcv/README.md)

Python 例程使用 SAIL 推理，无需编译直接运行；C++ 例程使用 BMRT + BMCV，端到端性能更优。两者实现相同的两图循环推理流程，测试参数和运行方式详见各自 README。

## 5. 精度测试

### 5.1 测试方法

在 SE9（BM1688）上对 FP16 模型进行精度测试，测试视频 10 帧、Q=1 查询点（原图像素 (y=50, x=100)），以第 0 帧跟踪结果作为精度判据（与 PyTorch CPU 参考值比对）：

```bash
python3 python/tapnext_infer.py \
    --input datasets/test.mp4 \
    --init_bmodel models/BM1688/tapnext_init_fp16_1b.bmodel \
    --step_bmodel models/BM1688/tapnext_step_fp16_1b.bmodel \
    --query 50,100 --max_frames 10 \
    --output_dir results/fp16
```

### 5.2 测试结果

FP16 生产精度的第 0 帧跟踪结果（查询点 (50,100)，参考值约为 (50, 100)）：

| 测试平台 | 测试程序 | 测试模型 | 第0帧跟踪(y,x) |
| -------- | -------- | -------- | ------------- |
| SE9 | tapnext_infer.py | tapnext_init/step_fp16_1b.bmodel | (50.06, 99.75) |
| SE9 | tapnext_bmcv.soc | tapnext_init/step_fp16_1b.bmodel | (49.88, 100.00) |

> **测试说明**：
> 1. FP16 跟踪结果与 PyTorch 参考值一致（亚像素级误差），为生产精度；
> 2. Python（SAIL）与 C++（BMCV 预处理）结果存在 <0.2 像素的差异，属 BMCV 硬件缩放与 cv2 软解的舍入差异，正常。

- INT8 量化分析（**结论：INT8 不可用，生产用 FP16**）

在 SE9 上对 6 种 INT8 变体的 init 图进行复测（`tapnext_bmcv.soc`，`test.mp4`，查询点 (y=50, x=100)，`--max_frames=1`，仅运行 init 图）。FP16 参考值为 (49.88, 100.00)，所有 INT8 变体均无法满足精度要求：

| 精度 | 第 0 帧跟踪 (y,x) | init 图耗时 | init 模型大小 | 结论 |
| --- | --- | --- | --- | --- |
| **FP16** | **(49.88, 100.00)** | **612 ms** | **542 MB** | **✅ 正确，生产精度** |
| 纯 INT8 | (1.00, 1.00) | 522 ms | 363 MB | ❌ 输出崩塌为 (1,1) |
| INT8-mix v1（预测头 F16） | (186.00, 158.38) | 522 ms | 364 MB | ❌ 散射 garbage |
| INT8-mix v2（非 MatMul 全 F16） | (144.25, 179.75) | 1198 ms | 381 MB | ❌ 散射 garbage（init 反慢 2×） |
| INT8 + per-channel | (1.00, 1.00) | 528 ms | 365 MB | ❌ 输出崩塌为 (1,1) |
| INT8-mix + per-channel | (195.50, 251.25) | 528 ms | 365 MB | ❌ 散射 garbage |
| W8F16 | — | — | — | ❌ 编译期固件 assert，无可用 bmodel |

> **根因分析:** TAPNext++ 的核心是 ViT self-attention + RG-LRU/Conv1D 状态空间模型，对 MatMul 的 INT8 量化极度敏感。即使将大量算子保留 F16（mix v2，仅 MatMul 量化），attention 模式仍被破坏，输出散射为 garbage——且 mix 变体的 garbage 在多次运行中**完全稳定可复现**，说明是确定性的量化误差放大，而非随机噪声。per-channel 量化（transformer INT8 的标准修复手段）亦无法缓解，纯 INT8 与 per-channel 均崩塌为 (1,1)。
>
> **速度亦无优势:** INT8 init 图仅比 FP16 快约 15%（522 vs 612 ms），但 init 图为一次性开销，逐帧推理由 step 图主导（FP16 约 637 ms/帧，见 6.2 节）；init 图既已崩塌，step 图的 INT8 化已无意义（故未保留 INT8 step bmodel）。更突出的是 INT8-mix v2 的 init 图反而比 FP16 慢 2×（1198 ms），说明本架构中大量 element-wise RG-LRU 门控/Conv1D 算子不受益于 INT8，量化/反量化开销甚至抵消 MatMul 加速。综上 FP16 在精度上完胜、速度上无明显劣势、模型仅大 1.4×，选为生产精度。

## 6. 性能测试

在 SE9（BM1688）上对 FP16 生产精度进行性能测试，测试视频 661 帧、Q=1 查询点。

### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1688/tapnext_step_fp16_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间。两个图的 BModel 分别测试，结果如下：

| 测试平台 | 测试模型 | calculate time(ms) |
| -------- | -------- | ----------------- |
| SE9 | BM1688/tapnext_init_fp16_1b.bmodel | 590.4 |
| SE9 | BM1688/tapnext_init_fp16_1b_2core.bmodel | 376.0 |
| SE9 | BM1688/tapnext_step_fp16_1b.bmodel | 615.7 |
| SE9 | BM1688/tapnext_step_fp16_1b_2core.bmodel | 398.5 |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性，已取 loopnum=10 的均值；
> 2. init 图仅在首帧运行一次，step 图为逐帧稳态推理；
> 3. 2-core 对 init/step 的 TPU 推理分别加速约 1.57×/1.55×。

### 6.2 程序运行性能
参考[C++例程](cpp/tapnext_bmcv/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。下表为逐帧稳态（step 图）的平均耗时：

| 测试平台 | 测试程序 | 测试模型 | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | -------- | -------- | ----------- | --------------- | -------------- | ---------------- |
| SE9 | tapnext_infer.py | tapnext_step_fp16_1b.bmodel | — | 2.42 | 911.32 | 0.03 |
| SE9 | tapnext_infer.py | tapnext_step_fp16_1b_2core.bmodel | — | 2.50 | 685.00 | 0.03 |
| SE9 | tapnext_bmcv.soc | tapnext_step_fp16_1b.bmodel | 1.31 | 0.49 | 636.65 | 0.40 |

> **测试说明**：
> 1. 时间单位均为毫秒(ms)，为逐帧稳态（step 图）的平均耗时；
> 2. `inference_time` 为 step 图稳态推理时间。**init 图仅在首帧运行一次**，为一次性开销：Python 约 4400 ms（含 SAIL 引擎创建 + 542 MB bmodel 加载 + 24 个 cache 提取），C++ 约 611 ms（BMRT 直接加载更高效）；
> 3. Python 例程一次性预读全部帧（cv2.VideoCapture），解码时间未单独统计，记为 "—"；
> 4. C++ 例程采用零拷贝 cache 回馈（见 [C++ README 3.3 节](./cpp/tapnext_bmcv/README.md#33-架构说明)），24 个循环 cache（约 131 MB）留在 device memory 无 D2S 回读，`postprocess_time` 仅约 0.4 ms（tracks/vis 回读）；Python 例程该开销包含在 SAIL 推理调用内，故 postprocess 趋近于 0；
> 5. 端到端稳态吞吐：Python 约 1.10 FPS（1-core）/ 1.46 FPS（2-core），C++ 约 1.56 FPS（1-core）；
> 6. 该模型 244M 参数、12 层 Transformer+SSM 循环架构，逐帧串行推理，上述性能符合此类模型在边缘 SoC 上的预期。

## 8. FAQ

**Q1: SE9 运行时报 OOM（exit 137）？**
A: SE9 默认 CPU 内存仅 ~850 MB，加载 step 图时可能 OOM。请按 [Python README](./python/README.md) 调整内存布局（减小 npu、把内存让给 CPU），改完需 reboot。

**Q2: C++ 例程交叉编译链接时报 `undefined reference to ...@GLIBC_2.34` 等符号错误？**
A: 这是交叉编译 sysroot 的 glibc 版本低于 SDK 运行库所需导致（x86 交叉 sysroot 常为 glibc 2.30，而 SDK 的 .so 需要 2.34/2.35）。可用一次性构建参数绕过：`cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" ..`。生成的二进制中直接的 libc 引用向后兼容，SDK 库所需的高版本符号在 SoC（glibc 2.35）运行时解析，可正常运行。

**Q3: 为什么不用 INT8？**
A: INT8 经 6 种变体充分验证均不可用（输出崩塌或散射 garbage），且无速度优势。详见 [5.2 测试结果](#52-测试结果)。

**Q4: 模型 track 输出的坐标系是什么？**
A: 模型 track 输出为 `[y, x]`（行、列），范围为 [0, 255.5] 模型像素。查询点格式为 `[t, y, x]`。`--query` 给的是原图像素坐标，程序会按视频分辨率自动缩放到 256×256 模型像素。

**Q5: C++ 例程支持哪些视频格式？**
A: C++ 例程使用 `cv::VideoCapture`（sophon-opencv）软件解码，对 FFmpeg 支持的任意编码格式/参数均兼容（无 VPU 硬解的 profile 限制），解码耗时约 1.3 ms/帧，相对逐帧推理可忽略。若遇个别视频无法打开，建议优先使用 `datasets/` 下随例程提供的测试视频。
