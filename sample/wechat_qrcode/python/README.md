[简体中文](./README.md) | [English](./README_EN.md)

# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 测试图片](#22-测试图片)

`python`目录下提供了一系列Python例程，具体情况如下：

| 序号 | Python例程               | 说明                                        | 图像加载      |
| ---- | ------------------------ | ------------------------------------------- | ------------- |
| 1    | wechat_qrcode_opencv.py  | OpenCV 读图（cv2.imdecode）→ `sail.wechat_qrcode` | CPU 解码 ndarray |
| 2    | wechat_qrcode_bmcv.py    | bmcv 读图（`sail.Decoder`/`sail.BMImage`）→ `sail.wechat_qrcode` | 硬件解码 BMImage |

两个例程都调用 `sophon-sail` 交付的 `sail.wechat_qrcode.WeChatQRCode` 接口，仅图像加载路径不同（`cv2.imdecode` vs `sail.Decoder`）；输出文本 + 四角点。精度上在同一平台因读图解码器不同存在实测差异（见主 [`README.md#5.2 测试结果`](../README.md#52-测试结果)）；性能上 `bmcv.py` 的 `sail.Decoder` 把解码与图像传输下放到 TPU/VPU、`opencv.py` 的 `cv2.imdecode` 走 CPU 且多一次主机→设备拷贝，故 `opencv.py` 的 FPS 略低于 `bmcv.py`（具体数值见主 README §6.2，两例程图像解码均在计时循环外完成、口径可比）。

## 1. 环境准备

### 1.1 x86/arm/riscv PCIe平台

如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

需要执行以下命令安装所需的python库：

```bash
pip3 install -r requirements.txt
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade  # 如需下载模型/数据集
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。还需安装 pysail wheel（`sophon_arm-*.whl`，随部署脚本推送到盒子上）：

```bash
pip3 install --user sophon_arm-*.whl
pip3 install -r requirements.txt
```

> **注:** `requirements.txt` 中的 `opencv-python` 用于 `wechat_qrcode_opencv.py` 的 `cv2.imdecode`；`wechat_qrcode_bmcv.py` 走 `sail.Decoder`，不需要 opencv-python。

## 2. 推理测试

Python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

两个例程参数一致（`--key value` 形式），如下：

```bash
usage: wechat_qrcode_opencv.py [-h] [--detect DETECT] [--sr SR] [--input INPUT]
                               [--dev_id DEV_ID] [--core_id CORE_ID] [--iters ITERS]

  --detect DETECT   detect 检测模型路径（默认 ../models/BM1684X/detect_f32_fused.bmodel）
  --sr SR           sr 超分模型路径（默认 ../models/BM1684X/sr_f16_fused.bmodel）
  --input INPUT     输入图片路径或目录（默认 ../images/qr_small.png）
  --dev_id DEV_ID   TPU device id（默认 0）
  --core_id CORE_ID BM1688 双核定核 0/1，-1 自动（默认 -1）
  --iters ITERS     >0 为 bench 模式迭代次数；<=0 为单图/目录模式（默认 0）
```

`--iters <= 0`（单图/目录模式）：对每张图打印每个二维码的 `text` 与四角点，并在 `results/{detect名}_{输入名}_{tag}_python_result.json` 落盘结果 JSON（供 `../tools/eval_qrcode.py` 直接消费）。

`--iters > 0`（bench 模式）：预热 3 次后跑 `iters` 次，打印端到端与 detect/sr/zxing 分段耗时。

### 2.2 测试图片

测试实例如下（支持目录，递归扫描）：

```bash
# OpenCV 读图
python3 wechat_qrcode_opencv.py --detect ../models/BM1688/detect_f32_fused.bmodel \
    --sr ../models/BM1688/sr_f16_fused.bmodel --input ../images/qr_small.png

# bmcv 读图
python3 wechat_qrcode_bmcv.py --detect ../models/BM1688/detect_f32_fused.bmodel \
    --sr ../models/BM1688/sr_f16_fused.bmodel --input ../images/qr_small.png

# 目录模式（评测集）
python3 wechat_qrcode_opencv.py --detect ../models/BM1688/detect_f32_fused.bmodel \
    --sr ../models/BM1688/sr_f16_fused.bmodel --input ../datasets/BoofCV_qrcode_v4/qrcodes

# bench 模式
python3 wechat_qrcode_opencv.py --detect ../models/BM1688/detect_f32_fused.bmodel \
    --sr ../models/BM1688/sr_f16_fused.bmodel --input ../images/qr_small.png --iters 50
```

预期对 `../images/qr_small.png` 输出：

```text
INFO: [../images/qr_small.png] decoded 1 code(s)
INFO:   [0] text=sail wechat qrcode small corners=(24.0,24.0) (174.0,24.0) (174.0,174.0) (24.0,174.0)
INFO: result saved in results/detect_f32_fused_qr_small_opencv_python_result.json (1/1 images decoded)
```

测试结束后，结果 JSON 保存在 `results/` 下。