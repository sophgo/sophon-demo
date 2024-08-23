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

| 序号 | Python例程           | 说明                                                                               |
| ---- | ------------------- | ---------------------------------------------------------------------------------- |
| 1    | blip_cap.py         | 图像字幕例程，使用opencv预处理，SAIL推理                                              |
| 2    | blip_server_cap.py  | 图像字幕例程，使用opencv预处理，SAIL推理，基于线程和队列的思想为http前后端接口提供服务    |
| 3    | blip_itm.py         | 图文匹配例程，使用opencv预处理，SAIL推理                                               |
| 4    | blip_server_itm.py  | 图文匹配例程，使用opencv预处理，SAIL推理，基于线程和队列的思想为http前后端接口提供服务    |
| 5    | blip_vqa.py         | 图文问答例程，使用opencv预处理，SAIL推理                                    |
| 6    | blip_server_vqa.py  | 图文问答例程，使用opencv预处理，SAIL推理，基于线程和队列的思想为http前后端接口提供服务    |

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：

```bash
pip3 install transformers
```
### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。
此外您可能还需要安装其他第三方库：
```bash
pip3 install transformers
```

若您使用sophon-opencv，需要设置环境变量，**使用sophon-opencv需要保证python版本小于等于3.8。**
```bash
export PYTHONPATH=/opt/sophon/sophon-opencv_<x.x.x>/opencv-python:$PYTHONPATH
```

## 2. 推理测试

blip_cap.py, blip_itm.py, blip_vqa.py 不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
blip_server_cap.py, blip_server_itm.py, blip_server_vqa.py 是封装好的class，为网页后端提供接口调用服务，在web_ui文件夹中调用，详情见[web_ui](../web_ui/README.md)。


### 2.1 参数说明

blip_cap.py
```bash
usage: blip_cap.py  [--image_path IMAGE_PATH] [--bmodel_path MODEL] [--tokenizer_path TOKENIZER_PATH] [--dev_id DEV_ID]
--image_path 测试图片路径，也可输入整个图片文件夹的路径；
--bmodel_path bmodel路径；
--tokenizer_path 分词器路径；
--dev_id: 用于推理的tpu设备id；
```

blip_itm.py
```bash
usage: blip_cap.py  [--image_path IMAGE_PATH] [--text [text ...]] [--bmodel_path MODEL] [--tokenizer_path TOKENIZER_PATH] [--dev_id DEV_ID]
--image_path 测试图片路径，也可输入整个图片文件夹的路径；
--text 用于与图片测试匹配度的文字;
--bmodel_path bmodel路径；
--tokenizer_path 分词器路径；
--dev_id: 用于推理的tpu设备id；
```

blip_vqa.py
```bash
usage: blip_cap.py  [--image_path IMAGE_PATH] [--venc_bmodel_path VENC_MODEL] [--tenc_bmodel_path TENC_MODEL] [--tdec_bmodel_path TDEC_MODEL] [--tokenizer_path TOKENIZER_PATH] [--dev_id DEV_ID]
--image_path 测试图片路径；
--venc_bmodel_path 图像编码器bmodel路径；
--tenc_bmodel_path 多模态编码器bmodel路径；
--tdec_bmodel_path 解码器bmodel路径；
--tokenizer_path 分词器路径；
--dev_id: 用于推理的tpu设备id；
```

### 2.2 测试图片
图文字幕测试实例如下，支持对整个图片文件夹进行测试。
```bash
python3 python/blip_cap.py --image_path datasets/test --bmodel_path models/BM1684X/blip_cap_bm1684x_f32_1b.bmodel --tokenizer_path models/bert-base-uncased --dev_id 0
```
程序运行结束后，会在命令行中打印信息，输出图片对应的字幕。

图文匹配测试实例如下
```bash
python3 python/blip_itm.py --image_path datasets/test/demo.jpg --text "a woman sitting on the beach with a dog" "a woman sitting on the beach with a cat" --bmodel_path models/BM1684X/blip_itm_bm1684x_f32_1b.bmodel --tokenizer_path models/bert-base-uncased --dev_id 0
```
程序运行结束后，会在命令行中打印信息，输出图片和文本的匹配度。

图文问答测试实例如下
```bash
python3 python/blip_vqa.py --image_path datasets/test/demo.jpg --venc_bmodel_path models/BM1684X/blip_vqa_venc_bm1684x_f32_1b.bmodel --tenc_bmodel_path models/BM1684X/blip_vqa_tenc_bm1684x_f32_1b.bmodel --tdec_bmodel_path models/BM1684X/blip_vqa_tdec_bm1684x_f32_1b.bmodel --tokenizer_path models/bert-base-uncased --dev_id 0
```
程序运行后，会提示输入问题，可以输入"where is the woman", "what are they doing", "what's the color of the clothes"来测试，输入exit退出，程序运行结束后，会在命令行中打印信息，输出图片预处理以及推理的时间
