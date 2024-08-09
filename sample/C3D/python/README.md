<font size=5>Python例程</font>

<font size=4> 目录</font>
- [1. 环境准备](#1-环境准备)
  - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
  - [1.2 SoC平台](#12-soc平台)
- [2. 推理测试](#2-推理测试)
  - [2.1 参数说明](#21-参数说明)
  - [2.2 测试视频理解数据集](#22-测试视频理解数据集)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | c3d_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install opencv-python-headless
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install opencv-python-headless
```

> **注:**
>
> 上述命令安装的是公版opencv，如果您希望使用sophon-opencv，可以设置如下环境变量：
> ```bash
> export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
> ```
> **若使用sophon-opencv需要保证python版本小于等于3.8。**

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明
```bash
usage: c3d_opencv.py [--input INPUT_PATH] [--bmodel BMODEL] [--dev_id DEV_ID]
--input: 测试数据路径，格式：文件夹/类别（二级目录）/视频；
--bmodel: 用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--dev_id: 用于推理的tpu设备id；
--classnames: 数据集类别文件。
```
### 2.2 测试视频理解数据集
测试实例如下，对UCF101视频数据集的一个子集进行测试。
```bash
python3 python/c3d_opencv.py --input datasets/UCF_test_01 --bmodel models/BM1684X/c3d_fp32_1b.bmodel --dev_id 0 --classnames datasets/ucf_names.txt
```
测试结束后，会打印预测结果、推理时间等信息，并将预测结果保存在`./results/c3d_fp32_1b.bmodel_opencv_python.json`中。

```
INFO:root:result saved in ./results/c3d_fp32_1b.bmodel_opencv_python.json
INFO:root:decode_time(ms): 67.19
INFO:root:preprocess_time(ms): 31.01
INFO:root:inference_time(ms): 88.76
INFO:root:postprocess_time(ms): 0.10
all done.
```
