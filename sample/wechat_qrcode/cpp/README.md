[简体中文](./README.md) | [English](./README_EN.md)

# C++例程

## 目录

- [C++例程](#c例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 程序编译](#2-程序编译)
    - [2.1 x86/arm/riscv PCIe平台](#21-x86armriscv-pcie平台)
    - [2.2 SoC平台](#22-soc平台)
  - [3. 推理测试](#3-推理测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 测试图片](#32-测试图片)

`cpp`目录下提供了两个C++例程以供参考使用，具体情况如下：

| 序号 | C++例程            | 说明                                                          |
| ---- | ------------------ | ------------------------------------------------------------- |
| 1    | wechat_qrcode_bmcv | 纯 bmrt + bmcv 的完整算法移植（不依赖 sail），vendor 了 zxing      |
| 2    | wechat_qrcode_sail | 薄封装，调用 `sail::wechat_qrcode` 交付接口（依赖 libsail.so）     |

两者共用同一套 detect/sr 模型与同一算法，输出文本 + 四角点，性能一致；精度在同一平台上基本一致、存在个位数以内的实测差异，见主 [`README.md#5.2 测试结果`](../README.md#52-测试结果)。

## 1. 环境准备

### 1.1 x86/arm/riscv PCIe平台

如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv（以及 `wechat_qrcode_sail` 所需的sophon-sail），具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

## 2. 程序编译

C++程序运行前需要编译可执行文件。

### 2.1 x86/arm/riscv PCIe平台

可以直接在PCIe平台上编译程序：

```bash
# bmcv 例程（不依赖 sail）
cd cpp/wechat_qrcode_bmcv
mkdir build && cd build
cmake ..
make
cd ..

# sail 例程（依赖 /opt/sophon/sophon-sail）
cd cpp/wechat_qrcode_sail
mkdir build && cd build
cmake ..
make
cd ..
```

编译完成后，会分别在对应目录下生成 `wechat_qrcode_bmcv.pcie` / `wechat_qrcode_sail.pcie`。

### 2.2 SoC平台

通常在x86主机上交叉编译程序（需先搭建交叉编译环境，参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)）：

```bash
# bmcv 例程
cd cpp/wechat_qrcode_bmcv
mkdir build && cd build
# 请根据实际情况修改 -DSDK 的路径，需使用绝对路径
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make

# sail 例程（额外需要 -DSAIL_PATH 指向 libsail.so 所在目录）
cd ../../wechat_qrcode_sail
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk -DSAIL_PATH=/path_to_sail ..
make
```

编译完成后，会在对应目录下生成 `wechat_qrcode_bmcv.soc` / `wechat_qrcode_sail.soc`。

> **注意（sail 读图路径）：** `wechat_qrcode_sail` 默认 `USE_OPENCV_DECODE=0`，即 bench 模式（设备读图路径）用 `sail::Decoder`（ffmpeg 硬件解码）。如需改用 sophon-opencv 的三参 `cv::imread()`（VPU 解码到设备），可在上面的 cmake 命令中追加 `-DUSE_OPENCV_DECODE=1`。该开关只影响 bench 读图路径，目录模式（精度读图）仍用主机 `cv::imread`，不受影响。

## 3. 推理测试

对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以SoC模式进行介绍。

> **注意：** `wechat_qrcode_sail` 依赖完整的 `libsail.so`（内含 `sail::wechat_qrcode` 符号），运行前需保证 `LD_LIBRARY_PATH` 包含 libsail 所在目录；`wechat_qrcode_bmcv` 无此依赖。

### 3.1 参数说明

两个例程均使用位置参数（与 Python 不同，不使用 `--key=value`），依次为：

```bash
./cpp/wechat_qrcode_bmcv/wechat_qrcode_bmcv.soc detect.bmodel sr.bmodel input_path dev_id iters
./cpp/wechat_qrcode_sail/wechat_qrcode_sail.soc detect.bmodel sr.bmodel input_path dev_id iters [core_id]
```

| 参数         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| detect.bmodel| detect 检测模型路径（默认 `../models/BM1684X/detect_f32_fused.bmodel`） |
| sr.bmodel    | sr 超分模型路径（默认 `../models/BM1684X/sr_f16_fused.bmodel`）       |
| input_path   | 输入图片路径或目录（默认 `../images/qr_small.png`）                 |
| dev_id       | TPU device id（默认 0）                                        |
| iters        | >0 为 bench 模式迭代次数；<=0 为单图/目录模式（默认 0）               |
| core_id      | BM1688 双核 NPU 定核（0/1），-1 为自动（默认 -1；仅 sail 例程）      |

`iters <= 0`（单图/目录模式）：对每张图打印每个二维码的 `text` 与四个角点，并在 `results/wechat_qrcode_results.json` 落盘结果 JSON（供 `../tools/eval_qrcode.py` 直接消费）。

`iters > 0`（bench 模式）：预热 3 次后跑 `iters` 次，打印端到端与 detect/sr/zxing 分段耗时（`ms/img` 与 `fps`）。

### 3.2 测试图片

图片测试实例如下，支持对整个图片目录（递归）进行测试：

```bash
# 单图模式
./cpp/wechat_qrcode_bmcv/wechat_qrcode_bmcv.soc models/BM1688/detect_f32_fused.bmodel models/BM1688/sr_f16_fused.bmodel images/qr_small.png 0 0

# 目录模式（递归扫描所有 jpg/png/bmp）
./cpp/wechat_qrcode_bmcv/wechat_qrcode_bmcv.soc models/BM1688/detect_f32_fused.bmodel models/BM1688/sr_f16_fused.bmodel datasets/BoofCV_qrcode_v4/qrcodes 0 0

# bench 模式
./cpp/wechat_qrcode_sail/wechat_qrcode_sail.soc models/BM1688/detect_f32_fused.bmodel models/BM1688/sr_f16_fused.bmodel images/qr_small.png 0 50
```

预期对 `images/qr_small.png` 输出：

```text
[images/qr_small.png]
  decoded 1 code(s)
  [0] text=sail wechat qrcode small
     corners(4x2 float32): (24.0,24.0) (174.0,24.0) (174.0,174.0) (24.0,174.0)
```

测试结束后，结果 JSON 保存在 `results/` 下。