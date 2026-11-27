# C++例程

- [C++例程](#c例程)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 程序编译](#2-程序编译)
    - [2.1 x86/arm/riscv PCIe平台](#21-x86armriscv-pcie平台)
    - [2.2 SoC平台](#22-soc平台)
  - [3. 推理测试](#3-推理测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 测试图片](#32-测试图片)
  - [4. 实现说明](#4-实现说明)

cpp 目录下提供了一系列 C++ 例程，具体情况如下：

| 序号 |   C++例程       | 说明                                      |
| ---- | --------------- | ---------------------------------------- |
| 1    | moge_bmcv       | 使用 sophon-opencv 解码、BMCV 前处理、BMRT 推理 |

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台

如果您在 x86/arm/riscv 平台安装了 PCIe 加速卡（如 SC 系列加速卡），并使用它测试本例程，您需要安装 libsophon、sophon-opencv、sophon-ffmpeg 和 sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)、[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台

如果您使用 SoC 平台（如 SE、SM 系列边缘设备），刷机后在 `/opt/sophon/` 下已经预装了相应的 libsophon、sophon-opencv 和 sophon-ffmpeg 运行库包。您还需要交叉编译好 BMRT 所需的交叉编译工具链与 libsophon，具体可参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#4-tdl-sdk交叉编译环境搭建)。

## 2. 程序编译

### 2.1 x86/arm/riscv PCIe平台
可以直接在 PCIe 平台上编译程序：
```bash
cd cpp/moge_bmcv
mkdir build && cd build
cmake .. && make
cd ..
```
编译完成后，会在 `cpp/moge_bmcv` 下生成 `moge_bmcv.pcie`。

### 2.2 SoC平台
通常在 x86 主机上交叉编译程序，您需要在 x86 主机上使用 SOPHON SDK 搭建交叉编译环境，将程序所依赖的头文件和库文件打包至 SDK 中。交叉编译步骤如下：
```bash
cd cpp/moge_bmcv
mkdir build && cd build
cmake .. -DTARGET_ARCH=soc -DSDK=/path_to_soc_sdk && make
cd ..
```
编译完成后，会在 `cpp/moge_bmcv` 下生成 `moge_bmcv.soc`。

## 3. 推理测试
### 3.1 参数说明
```bash
usage: moge_bmcv.pcie [--input INPUT_PATH] [--bmodel BMODEL] [--dev_id DEV_ID] [--help]
--input:  测试数据路径，输入整个图片文件夹路径或单个图片路径；
--bmodel: 用于推理的 bmodel 路径；
--dev_id: 用于推理的 tpu 设备 id；
--help:   打印帮助信息
```

### 3.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
./moge_bmcv.soc --input=../../datasets/test --bmodel=../../models/BM1688/moge-2-vitl-normal_fp16_1b.bmodel --dev_id=0
```
测试结束后，会将结果保存在 `results` 目录下：

```bash
results/
├── images/
│   ├── <name>_depth.png     # 深度着色图
│   └── <name>_normal.png    # 法向着色图
├── <name>_depth.npy         # 度量深度（518x518 float32）
├── <name>_points.npy        # 点云（518x518x3 float32）
├── <name>_normal.npy        # 法向图（3x518x518 float32，CHW）
└── <bmodel>_cpp_result.json # 每张图内参/视场角/深度范围汇总
```

同时会打印 `decode / preprocess / inference / postprocess` 四段平均耗时（TimeStamp 汇总）。

## 4. 实现说明

本例程 C++ 代码严格复刻 Python 例程（`python/moge_bmcv.py` + `python/postprocess_numpy.py`）的口径，保证跨语言结果一致：

- **前处理**：sophon-opencv 硬解 JPEG → `cv::bmcv::toBMI` 转 bm_image → `convert_format` 转 RGB 平面 → BMCV 双线性 resize 到 518×518 → `/255` 归一化（BMCV 定点缩放会差到 /256，故 `/255` 在浮点阶段完成，保证与 Python 完全一致）。
- **推理**：`bm_wrapper.hpp` 的 `bmrt` 推理（仅用 bm_wrapper，不使用 bmnn_utils，参考 yolov8_plus_det 的 C++ 实现）。
- **后处理**：与 `postprocess_numpy.py` 相同的 focal/shift 恢复、pixel-center 投影、HFoV/VFoV 计算，以及相同的深度/法向着色。
- **.npy 写入**：按 NumPy `.npy` v1.0 格式写出，与 Python 例程的 `.npy` 逐字节对齐（含 shape 顺序：depth/pixels 为 (H,W)、points 为 (H,W,3)、normal 为 (3,H,W)）。