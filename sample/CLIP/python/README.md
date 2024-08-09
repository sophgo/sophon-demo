# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 测试图片](#22-测试图片)


python目录下提供了一系列Python例程，具体情况如下：

| 序号 | Python例程          | 说明         |
| ---- | ------------------- | ------------ |
| 1    | zeroshot_predict.py | 使用SAIL推理 |


## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：

```bash
pip3 install ftfy
pip3 install regex
pip3 install torch
pip3 install torchvision
```
### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。
此外您可能还需要安装其他第三方库：
```bash
pip3 install ftfy
pip3 install regex
pip3 install torch
pip3 install torchvision
```

若您使用sophon-opencv，需要设置环境变量，**使用sophon-opencv需要保证python版本小于等于3.8。**
```bash
export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv_<x.x.x>/opencv-python
```

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

```bash
usage: zeroshot_predict.py  [--image_path IMAGE_PATH] [--text TEXT [TEXT ...]] [--image_model IMAGE_MODEL]         
                            [--text_model TEXT_MODEL] [--dev_id DEV_ID]
--image_path: 测试图片路径，也可输入整个图片文件夹的路径；
--text: 输入多段文本；
--image_model 图片编码bmodel；
--text_model 文本编码bmodel；
--dev_id: 用于推理的tpu设备id；
```

### 2.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
python3 python/zeroshot_predict.py --image_path datasets --text "a diagram" "a dog" "a cat" --image_model models/BM1684X/clip_image_vitb32_bm1684x_f16_1b.bmodel --text_model models/BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel --dev_id 0
```
程序运行结束后，会在命令行中打印信息，输出图片和文本的匹配度。

```
INFO:root:Text: a diagram, Similarity: 0.9986683130264282
INFO:root:Text: a cat, Similarity: 0.0008334724698215723
INFO:root:Text: a dog, Similarity: 0.0004982181708328426
INFO:root:-------------------Image num 1, Preprocess average time ------------------------
INFO:root:preprocess(ms): 11.55
INFO:root:------------------ Image num 1,Image Encoding average time ----------------------
INFO:root:image_encode(ms): 8.09
INFO:root:------------------ Image num 1, Text Encoding average time ----------------------
INFO:root:text_encode(ms): 26.06
```

