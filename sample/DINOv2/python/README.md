# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 测试图片](#22-测试图片)

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | dinov2_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |
| 2    | dinov2_bmcv.py   | 使用SAIL解码、BMCV前处理、SAIL推理     |

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台

如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install -r requirements.txt
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install -r requirements.txt
```

> **注：`requirements.txt` 里没有 `sophon`。** SoC 上的 `sophon.sail` 必须用上面链接的方法交叉编译安装（产物在 `~/.local/lib/python3.*/site-packages/sophon/`）；PyPI 上的 `sophon` 是 3.5KB 的占位包，`pip install` 会把占位文件写进同一个 `sophon/` 目录，可能破坏已装好的 sail。

> **注:**
>
> 上述命令安装的opencv是公版opencv，如果您希望使用sophon-opencv，可以设置如下环境变量：
> ```bash
> export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
> ```
> **若使用sophon-opencv需要保证python版本小于等于3.8。**

## 2. 推理测试
python例程不需要编译，可以直接运行。SoC 平台的测试参数和运行方式与 PCIe 相同；本版本的数据均在 SoC 上实测，PCIe 路径未实测。

> Python 例程的 `--input` 只接受**图片目录**（脚本会遍历目录下的图片）。需要单张推理时，请把图片放进一个目录；C++ 例程同时支持单文件和目录。

模型输出的是官方线性探针特征 `concat([CLS], mean(patch_tokens))`（维度 1536），例程在CPU侧再接官方线性头 `dinov2_vitb14_reg4_linear_head`（`weight[1000,1536]` + `bias[1000]`）得到 1000 类 logits，最后 softmax 取 Top-1。

前处理对齐官方 `dinov2/data/transforms.py: make_classification_eval_transform()` 的默认参数（官方 ImageNet-1k linear-probe 精度口径）：`Resize(短边=256, BICUBIC)` -> `CenterCrop(224)` -> `(x/255 - mean) / std`，其中 `mean=(0.485,0.456,0.406)`、`std=(0.229,0.224,0.225)`。`dinov2_bmcv.py` 因 VPP 不支持 BICUBIC，用 `BMCV_INTER_LINEAR` 替代。

### 2.1 参数说明
dinov2_opencv.py和dinov2_bmcv.py的命令参数相同，以dinov2_opencv.py的推理为例，参数说明如下：

```bash
usage:dinov2_opencv.py [--input IMG_PATH] [--bmodel BMODEL] [--head HEAD] [--classnames CLASSNAMES] [--topk TOPK] [--dev_id DEV_ID]
--input: 推理图片目录路径（不支持单文件；单张图片请放入目录后传入）；
--bmodel: 用于推理的bmodel路径，默认为../models/BM1684X/dinov2_vitb14_reg_fp32_1b.bmodel；
--head: 官方线性头权重文件，两个 Python 例程均使用 npz 格式（`dinov2_vitb14_reg4_linear_head.npz`；C++ 例程使用 `.bin`）；
--classnames: ImageNet-1k类别名文件；
--topk: 每张图片输出top-k个类别及概率，默认5；
--dev_id: 用于推理的tpu设备id。
```

### 2.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试。示例按 BM1684X 编写，在 SE9-16 / SE9-8 上请把 `models/BM1684X/` 换成 `models/BM1688/` 或 `models/CV186X/`（文件名相同）。
```bash
# 测试整个文件夹
python3 python/dinov2_opencv.py --input datasets/imagenet_val_1k/img --bmodel models/BM1684X/dinov2_vitb14_reg_fp32_1b.bmodel --head datasets/dinov2_vitb14_reg4_linear_head.npz --classnames datasets/imagenet_classes.txt --dev_id 0
```
测试结束后，会将预测结果保存在当前工作目录的`results/dinov2_vitb14_reg_fp32_1b.bmodel_img_opencv_python_result.json`下，同时会打印预测结果、推理时间等信息。
