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

- 此外您可能还需要安装其他库：

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

python3 -m dfss --url=open@sophgo.com:tools/silk2/silk2.tools.logger-1.0.2-py3-none-any.whl
pip3 install silk2.tools.logger-1.0.2-py3-none-any.whl --force-reinstall
rm -f silk2.tools.logger-1.0.2-py3-none-any.whl

pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

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

- 此外您可能还需要安装其他库：

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

python3 -m dfss --url=open@sophgo.com:tools/silk2/silk2.tools.logger-1.0.2-py3-none-any.whl
pip3 install silk2.tools.logger-1.0.2-py3-none-any.whl --force-reinstall
rm -f silk2.tools.logger-1.0.2-py3-none-any.whl

pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
``` 

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
                   [-vc VISION_PREPROCESS_CONFIG]
--bmodel_path: 用于推理的bmodel路径；
--tokenizer_path: tokenizer目录路径；
--processor_path: 预处理参数文件路径；
--config: 模型配置文件路径；
--dev_id: 用于推理的tpu设备id；
--vision_inputs: json格式，输入图片、视频文件路径、视觉预处理参数，可接受多个图片输入，格式：[{"type":"video","video":path-to-video/list(path-to-frame image), ...},{"type":"video","video":path-to-video/list(path-to-frame image), ...},{"type":"image","image":path-to-image, ...},...]。其中字典里面的其他参数用于生成prompt的额外视觉预处理参数，例如可设置"resized_height"、"resized_width"、"min_pixels"、"max_pixels"等，与官方支持的输入一致，在内存不足时，可适当设置这些参数，支持的常用参数说明如下：
    * --resized_height: resize后的固定高度，会进行28对齐，即28的倍数；
    * --resized_width: resize后的固定宽度，会进行28对齐，即28的倍数；
    * --min_pixels: resize后的最小像素点数量，像素点数量是图片高度乘以宽度，若小于该像素数量会重新计算高宽的缩放比例，最终高度宽度也会进行28对齐，即28的倍数，若--resized_height和--resized_width设置，则该参数无效；
    * --max_pixels: resize后的最大像素点数量，像素点数量是图片高度乘以宽度，若大于该像素数量会重新计算高宽的缩放比例，最终高度宽度也会进行28对齐，即28的倍数，若--resized_height和--resized_width设置，则该参数无效；
    * --video_sample_num: 输入到模型推理的视频帧数量，仅对输入为单一视频文件有用，对于输入一系列视频帧图片和图片输入无效；
    * --nframes: 均匀采样得到的视频帧数量，可以通过该参数实现抽帧；
    * --video_start: 设置需要处理的视频起始帧；
    * --video_end: 设置需要处理的视频结束帧；
--log_level: log等级，支持DEBUG、INFO、WARNING、ERROR，默认为INFO。
```

### 2.2 使用方式

- 为了测试`../datasets/videos/carvana_video.mp4`输入，设置`resized_height`、`resized_width`参数到较小值，并设置`nframes`参数为处理2帧，可以使用如下命令
```bash
python3 qwen2_vl.py --vision_inputs="[{\"type\":\"video\",\"video\":\"../datasets/videos/carvana_video.mp4\",\"resized_height\":420,\"resized_width\":630,\"nframes\":2}]"
```

- 为了测试图片，可以参考执行如下命令
```bash
python3 qwen2_vl.py --vision_inputs="[{\"type\":\"image\",\"image\":\"../datasets/images/panda.jpg\", \"resized_height\":280,\"resized_width\":420}]"
```

- 为了同时对图片和视频提问，可以参考执行如下命令
```bash
python3 qwen2_vl.py --vision_inputs="[{\"type\":\"video\",\"video\":\"../datasets/videos/carvana_video.mp4\",\"resized_height\":420,\"resized_width\":630,\"nframes\":2},{\"type\":\"image\",\"image\":\"../datasets/images/panda.jpg\", \"resized_height\":280,\"resized_width\":420}]"
```

- 为了纯文本对话，可以参考执行如下命令
```bash
python3 qwen2_vl.py --vision_inputs=""
```

在Question: 处进行提问，例如：Describe this video。
终端将打印FTL、TPS性能数据，并输出回答结果，接着可进一步对视频进行提问，输入q即可退出。

> **测试说明**：  
> 1. 图片或者视频尺寸越大，一般精度越高，直到达到一定尺寸，较大输入需要上下文较长的模型；