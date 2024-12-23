# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 使用方式](#22-使用方式)

Qwen2-vl能够输入单一图片/多张图/视频进行连续对话，python目录下提供了例程，具体情况如下：

| 序号  |  Python例程       |            说明                 |
| ---- | ---------------- | ------------------------------ |
|   1  | qwen2_vl.py      | 使用SAIL推理                     |

## 1. 环境准备

### 1.1 x86/arm PCIe平台

- 需要**SDK v24.04.01及其以上版本**

- 如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

- 此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# x86_64平台使用如下命令
pip3 install decord==0.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# arm平台使用如下命令
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen2-VL/decord-0.6.0-cp38-cp38-linux_aarch64.whl
pip3 install decord-0.6.0-cp38-cp38-linux_aarch64.whl
rm -f decord-0.6.0-cp38-cp38-linux_aarch64.whl
```

*其中decord在ARM平台没有官方安装包，若需要自行编译安装，可参考[decord从源码安装指南](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source)，该程序需要设置-DFFMPEG_DIR到FFMPEG路径。*

- 您还需要安装sophon-sail，由于本例程需要的sophon-sail版本较新，可以用如下命令下载sophon-sail源码，并参考[sophon-sail python3接口编译安装指南](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#python3wheel)自行编译sophon-sail。

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen2-VL/sophon-sail_24_12_18.zip
unzip sophon-sail_24_12_18.zip
```

- 需要下载运行配置文件，执行如下命令

```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen2-VL/configs.zip
unzip configs.zip
rm configs.zip
```

### 1.2 SoC平台

- 需要**SDK v24.04.01及其以上版本**

- 如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，请使用**SDK V24.04.01及其以上版本**对应的刷机包进行刷机，刷机成功后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

- 此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen2-VL/decord-0.6.0-cp38-cp38-linux_aarch64.whl
pip3 install decord-0.6.0-cp38-cp38-linux_aarch64.whl
rm -f decord-0.6.0-cp38-cp38-linux_aarch64.whl
``` 

*其中decord在ARM平台没有官方安装包，若需要自行编译安装，可参考[decord从源码安装指南](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source)和[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)，该程序需要设置-DFFMPEG_DIR到FFMPEG路径。在交叉编译环境编译得到.so文件后，需要拷贝代码到ARM SOC机器，执行python bindings相关操作。*

- 本例程依赖sophon-sail，可直接安装编译好的sophon-sail包，执行如下命令：

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen2-VL/sophon_arm-3.9.2-py3-none-any.whl
pip3 install sophon_arm-3.9.2-py3-none-any.whl --force-reinstall
```

- 您还可以自行交叉编译安装sophon-sail，由于本例程需要的sophon-sail版本较新，可以先用如下命令下载sophon-sail源码，接着可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。 

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen2-VL/sophon-sail_24_12_18.zip
unzip sophon-sail_24_12_18.zip
```

- 需要下载运行配置文件，执行如下命令

```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen2-VL/configs.zip
unzip configs.zip
rm configs.zip
```

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

```bash
usage: qwen2_vl.py [-h] [-m BMODEL_PATH] [-t TOKENIZER_PATH] [-p PROCESSOR_PATH] [-c CONFIG] [-d DEV_ID] [-g {greedy,penalty_sample}] [-i INPUT_PATHS [INPUT_PATHS ...]] [-ity {image,video}]
                   [-vc VISION_PREPROCESS_CONFIG] [-fsn FRAME_SAMPLE_NUM]
--bmodel_path: 用于推理的bmodel路径；
--tokenizer_path: tokenizer目录路径；
--processor_path: 预处理参数文件路径；
--config: 模型配置文件路径；
--dev_id: 用于推理的tpu设备id；
--input_paths: 输入图片、视频文件路径，可接受多个图片输入；
--input_type: 输入的类型，仅支持"image"、"video"两种；
--vision_preprocess_config: 用于生成prompt的额外预处理参数，例如可设置"resized_height"、"resized_width"、"min_pixels"、"max_pixels"等，与官方支持的输入一致，在内存不足时，可适当设置这些参数；
--frame_sample_num: 视频帧采样间隔，仅对输入为单一视频文件有用，对于输入一系列视频帧图片和图片输入无效；
```

### 2.2 使用方式

- 为了测试`../datasets/videos/carvana_video.mp4`输入，设置`resized_height`、`resized_width`参数到较小值，并设置`--frame_sample_num`参数每隔两帧抽一帧，可以使用如下命令
```bash
python3 qwen2_vl.py --frame_sample_num=2 --vision_preprocess_config=\{\"resized_height\":140,\"resized_width\":210\}
```

- 为了测试图片，可以参考执行如下命令
```bash
python3 qwen2_vl.py --input_paths ../datasets/images/panda.jpg --input_type image --vision_preprocess_config=\{\"resized_height\":280,\"resized_width\":420\}
```

在Question: 处进行提问，例如：Describe this video。
终端将打印FTL、TPS性能数据，并输出回答结果，接着可进一步对视频进行提问，输入q即可退出。
