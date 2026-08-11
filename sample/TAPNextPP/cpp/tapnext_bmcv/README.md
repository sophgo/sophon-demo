# C++例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 编译](#2-编译)
    * [2.1 x86/arm/riscv PCIe平台](#21-x86armriscv-pcie平台)
    * [2.2 SoC平台](#22-soc平台)
* [3. 推理测试](#3-推理测试)
    * [3.1 参数说明](#31-参数说明)
    * [3.2 视频跟踪](#32-视频跟踪)

cpp目录下提供 C++ 版本的 TAPNext++ 推理例程 `tapnext_bmcv`，使用 BMRT + BMCV 对两图循环架构进行逐帧点跟踪。与 Python 例程功能一致，但通过 C++ BMRT 直接推理（绕过 SAIL SYSIO 开销），端到端性能更优。

| 序号 | C++例程        | 说明                                              |
| ---- | -------------- | ------------------------------------------------- |
| 1    | tapnext_bmcv   | OpenCV 解码 + BMCV 前处理 + BMRT 推理 init/step 双图 |

## 1. 环境准备

### 1.1 x86/arm/riscv PCIe平台

如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接作为运行环境。通常还需要一台 x86 主机作为开发环境，用于交叉编译 C++ 程序（SoC 仅支持交叉编译，不支持在 SoC 上本地编译）。

> **注:** 运行前需设置环境变量：
> ```bash
> export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH
> ```
>
> SE9 SoC 内存布局调整请参考 [Python README 1.3 节](../python/README.md#13-se9-内存布局调整)。

## 2. 编译

### 2.1 x86/arm/riscv PCIe平台

```bash
mkdir build && cd build
cmake .. && make
```

编译完成后可执行文件在 `tapnext_bmcv/tapnext_bmcv.pcie`。

### 2.2 SoC平台

在 x86 主机上交叉编译，需先使用 SOPHON SDK 搭建交叉编译环境，将程序依赖的头文件和库文件打包至 `soc-sdk` 目录，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程依赖 libsophon、sophon-opencv 和 sophon-ffmpeg 运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/tapnext_bmcv
mkdir build_soc && cd build_soc
# 请根据实际情况修改 -DSDK 路径，需使用绝对路径
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make
```

编译完成后可执行文件在 `tapnext_bmcv/tapnext_bmcv.soc`，拷贝到 SoC 设备上运行。

> **注:** 本例程 vendored `json.hpp` 为 nlohmann/json 3.11.2，需要 C++17，因此 SoC 交叉编译使用 `-std=c++17`（多数其它例程使用 `-std=c++11`）。

## 3. 推理测试

PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 3.1 参数说明

```bash
usage: tapnext_bmcv.{pcie,soc} [--input INPUT] [--init_bmodel INIT_BMODEL] [--step_bmodel STEP_BMODEL]
                              [--dev_id DEV_ID] [--query QUERY] [--query_file QUERY_FILE]
                              [--max_frames MAX_FRAMES] [--output_dir OUTPUT_DIR]
--input:       输入视频文件或图像目录路径；
--init_bmodel: init 图 bmodel 路径（第 0 帧）；
--step_bmodel: step 图 bmodel 路径（后续帧）；
--dev_id:      TPU 设备 id；
--query:       查询点 "y1,x1;y2,x2"，原图像素坐标（t=0，自动缩放到模型尺寸）；
--query_file:  查询点 JSON 文件 [[t,y,x],...]，模型像素坐标（不缩放）；
--max_frames:  最大处理帧数，0 = 全部；
--output_dir:  输出目录。
```

### 3.2 视频跟踪

```bash
# FP16：跟踪原图 (y=50, x=100) 处的点，处理 10 帧
# PCIe 平台：
./tapnext_bmcv.pcie \
    --input=../../datasets/test.mp4 \
    --init_bmodel=../../models/BM1688/tapnext_init_fp16_1b.bmodel \
    --step_bmodel=../../models/BM1688/tapnext_step_fp16_1b.bmodel \
    --query=50,100 --max_frames=10 \
    --output_dir=results/fp16

# SoC 平台（运行前需设置库路径）：
export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:/opt/sophon/sophon-opencv-latest/lib:/opt/sophon/sophon-ffmpeg-latest/lib:$LD_LIBRARY_PATH
./tapnext_bmcv.soc \
    --input=../../datasets/test.mp4 \
    --init_bmodel=../../models/BM1688/tapnext_init_fp16_1b.bmodel \
    --step_bmodel=../../models/BM1688/tapnext_step_fp16_1b.bmodel \
    --query=50,100 --max_frames=10 \
    --output_dir=results/fp16
```

> **参数格式说明:** 请使用 `--key=value` 形式传参（而非 `--key value` 空格形式）。本例程依赖的 sophon-opencv 4.8.0 中 `cv::CommandLineParser` 对空格形式传值的解析存在缺陷，会把值误判为字符串 `"true"`，导致 `cannot find init bmodel: true` 等错误。该问题影响 PCIe 与 SoC 两种构建。

测试结束后，跟踪结果保存在 `output_dir/tracks.json`（逐帧 `{y, x, visible}`），同时打印各阶段耗时。

> **坐标系说明:** 模型 track 输出为 `[y, x]`（行、列），范围为 [0, 255.5] 模型像素。查询点格式为 `[t, y, x]`。`--query` 给的是原图像素坐标，程序会按视频分辨率自动缩放到 256×256 模型像素。
>
> **精度说明:** 生产精度使用 FP16，详见 [Python README 精度说明](../python/README.md#22-视频跟踪)。

### 3.3 架构说明

C++ 例程与 Python 例程实现相同的两图循环推理流程，但使用 BMRT C API 替代 SAIL：

1. **前处理 (BMCV):** `bmcv_image_vpp_convert` 完成 resize + 色彩转换（任意输入格式 → RGB_PLANAR 256×256），`bmcv_image_convert_to` 完成归一化（`x / 127.5 - 1.0` → `[-1, 1]` float32），全程在 TPU 侧 device memory 上完成，无需 CPU 介入。
2. **推理 (BMRT):** `bmrt_launch_tensor` + `bm_thread_sync` 直接驱动 TPU。输入张量的 `bm_tensor_t::device_mem` 绑定 device memory（帧图像来自 BMCV，query/step/cache 来自 host 拷贝）。输出中 tracks/vis 经 D2S 拷回 CPU 用于结果保存；24 个循环 cache 则零拷贝留在 device memory 上直接回馈下一帧（见下条），无 host 往返。
3. **循环状态管理（零拷贝）:** 24 个 cache 张量（12 层 × (rg_lru, conv1d)，约 131 MB）是 `bmrt_launch_tensor` 分配、归调用者所有的 device memory，因此 init 图输出可直接作为 step 图输入、step 图每帧输出的 cache 也直接作为下一帧输入，全程在 device memory 上回馈，无 D2S/S2D host 往返。bmodel 顺序加载（init 释放后再加载 step）以适应 SE9 有限 CPU 内存。
4. **视频解码:** `cv::VideoCapture`（sophon-opencv）软件解码 + `bm_image_create` / `bm_image_copy_host_to_device` 上传 device memory。软解对编码格式/参数无特殊要求（VPU 硬解对部分 h264 profile 不兼容），且解码耗时（数 ms/帧）相对逐帧推理可忽略。

### 3.4 性能

SE9（BM1688）FP16 单点跟踪性能（`test_se9.mp4`，661 帧，Q=1）：

| 阶段         | 耗时            |
| ------------ | --------------- |
| 解码         | ~1.3 ms/帧      |
| 前处理       | ~0.5 ms/帧      |
| init 图      | ~612 ms（一次性）|
| step 图      | ~637 ms/帧      |
| 后处理       | ~0.4 ms/帧      |
| **端到端**   | **~641 ms/帧（1.56 FPS）** |

> step 图耗时较高是因为 TAPNext++ 含 12 层 Transformer + LRU 循环状态，模型本身计算量大。
> init 图仅第 0 帧执行一次，后续帧仅执行 step 图。
> 后处理仅是 tracks/vis 回读；24 个循环 cache（约 131 MB）零拷贝留在 device memory 上回馈，无 D2S host 往返（见 3.3 节）。
