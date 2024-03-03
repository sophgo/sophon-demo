[简体中文](./README.md)

# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 测试图片](#22-测试图片)
    * [2.3 测试视频](#23-测试视频)

python目录下提供了一系列Python例程，具体情况如下：

| 序号  |  Python例程          | 说明                              |
| ---- | -------------------- | ---------------------------------|
| 1    | groundingdino_pil.py | 使用PIL解码、PIL前处理、SAIL推理     |

## 1. 环境准备
考虑到GroundingDINO的demo需要在最新的sophon-sail下才能推理，因此需要当前使用的sail版本满足>=3.8.0.
用户可以通过执行
```bash
pip show sophon
```
后查看`Version`属性来确定版本号。您可以参考[编译可被Python3接口调用的Wheel文件](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#python3wheel)

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您还需要配置其他第三方库：
```bash
pip3 install -r requirements.txt
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您还需要配置其他第三方库：
```bash
pip3 install -r requirements.txt
```
## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明
以groundingdino_pil.py为例：
```bash
usage: groundingdino_pil.py [--image_path INPUT_PATH] [--bmodel BMODEL] [--dev_id DEV_ID]
                        [--text_threshold TEXT_THRESHOLD] [--box_threshold BOX_THRESHOLD] 
                        [--text_prompt TEXT_PROMPT][--output_dir OUTPUT_DIR] 
                        [--tokenizer_path TOKENIZER_PATH] [--token_spans TOKEN_SPANS]

--image_path: 测试数据路径，目前支持输入图片路径；
--bmodel: 用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--text_prompt: 用于检测的目标名称；
--dev_id: 用于推理的tpu设备id；
--text_threshold: 后处理中匹配每个token的置信度阈值；
--box_threshold: 后处理中用于筛选掉框的置信度；
--output_dir: 生成图片的保存位置；
--tokenizer_path: 分词器的地址；
--token_spans: 感兴趣的token的位置。
```

> **注意：** 默认token_spans是关闭的，可以添加参数`--token_spans`来开启该接口, 使用方式为:
`For example, a caption is 'a cat and a dog', if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. ,if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'.`

### 2.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
python3 groundingdino_pil.py --image_path ../datasets/test/zidane.jpg --bmodel ../models/BM1684X/groundingdino_bm1684x_fp16.bmodel --dev_id 0 --box_threshold 0.3 --text_threshold 0.2 --text_prompt "person" --tokenizer_path ../models/BM1684X/bert-base-uncased --output_dir ./results
```
测试结束后，会将预测的图片保存在`results`下.

