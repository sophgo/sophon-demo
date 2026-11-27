# Python例程

- [Python例程](#python例程)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 测试图片](#22-测试图片)
    - [2.3 输出说明](#23-输出说明)

python 目录下提供了一系列 Python 例程，具体情况如下：

| 序号 |  Python例程      | 说明                                  |
| ---- | ---------------- | ------------------------------------ |
| 1    | moge_bmcv.py     | 使用 SAIL 解码、BMCV 前处理、SAIL 推理 |

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台

如果您在 x86/arm/riscv 平台安装了 PCIe 加速卡（如 SC 系列加速卡），并使用它测试本例程，您需要安装 libsophon、sophon-opencv、sophon-ffmpeg 和 sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)、[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

本例程**不依赖**公版 opencv/pillow/scipy，仅需 numpy 与 sophon-sail：

```bash
pip3 install numpy
```

### 1.2 SoC平台

如果您使用 SoC 平台（如 SE、SM 系列边缘设备），刷机后在 `/opt/sophon/` 下已经预装了相应的 libsophon、sophon-opencv 和 sophon-ffmpeg 运行库包。您还需要交叉编译安装 sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

SoC 平台仅需确保 `numpy` 与 `sophon.sail` 可用（板端默认已预装）。

## 2. 推理测试
python 例程不需要编译，可以直接运行，PCIe 平台和 SoC 平台的测试参数和运行方式是相同的。
### 2.1 参数说明
```bash
usage: moge_bmcv.py [--input INPUT_PATH] [--bmodel BMODEL] [--dev_id DEV_ID]
--input: 测试数据路径，可输入单个图片路径或整个图片文件夹路径；
--bmodel: 用于推理的 bmodel 路径；
--dev_id: 用于推理的 tpu 设备 id；
```

### 2.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
python3 python/moge_bmcv.py --input datasets/test --bmodel models/BM1688/moge-2-vitl-normal_fp16_1b.bmodel --dev_id 0
```
测试结束后，会将结果保存在 `results` 目录下：

```bash
results/
├── <name>_depth.npy     # 度量深度（518x518 float32）
├── <name>_points.npy    # 点云（518x518x3 float32）
├── <name>_normal.npy    # 法向图（3x518x518 float32，CHW）
├── <name>_depth.png     # 深度着色图
├── <name>_normal.png    # 法向着色图
└── <bmodel>_python_result.json   # 每张图的内参/视场角/深度范围等汇总
```

同时会打印 `decode_time/preprocess_time/inference_time/postprocess_time`（单位 ms）。

### 2.3 输出说明

`<bmodel>_python_result.json` 中每张图的字段含义：

| 字段 | 说明 |
| ---- | ---- |
| `intrinsics` | 3×3 相机内参矩阵（fx, fy, cx=cy=0.5，pixel-center 定义） |
| `fov_x_deg` / `fov_y_deg` | 水平 / 垂直视场角（度） |
| `focal` | 恢复出的焦距（归一化坐标，等于 fx/2） |
| `shift` | 恢复出的 z 位移 |
| `depth_min` / `depth_max` | 有效区域内的深度范围（米） |