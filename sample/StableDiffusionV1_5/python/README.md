# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 测试文生图](#22-测试文生图)
    * [2.3 测试controlnet](#23-测试controlnet)

## 1. 环境准备

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

python/run.py脚本文件参数说明：

```bash
usage: run.py [--model_path BMODELS_PATH] [--stage SINGLIZE/MULTILIZE] [--controlnet_name CONTROLNET_NAME] [--processor_name PROCESSOR_NAME] [--controlnet_img IMG_FOR_CONTROLNET] [--tokenizer TOKENIZER_FILE] [--prompt PROMPT] [--neg_prompt NEGATIVE_PROMPT] [--num_inference_steps ITERATION_NUMS] [--dev_id DEV_ID]
--model_path: 各类bmodel文件的总目录;
--stage: singlize或multilize，controlnet必须选择multilize;
--controlnet_name controlnet bmodel文件名，需配合multilize使用;
--processor_name processor bmodel文件名，需配合multilize和controlnet使用;
--controlnet_img controlnet所用的参考图像;
--tokenizer tokenizer files路径;
--prompt 用于图像生成的提示词;
--neg_prompt 用于图像生成的负面提示词;
--num_inference_steps Stable Diffusion的迭代次数;
--dev_id: 用于推理的tpu设备id;
```

### 2.2 测试文生图

文生图测试实例如下:

```bash
cd python

python3 run.py --model_path ../models/BM1684X --stage singlize --prompt "a rabbit driking at the bar" --neg_prompt "worst quality" --num_inference_steps 20 --dev_id 0
```

运行结束后，生成的的图片保存为`result.png`。

### 2.3 测试controlnet

controlnet测试实例如下:

```bash
python3 run.py --model_path ../models/BM1684X --stage multilize --controlnet_name scribble_controlnet_fp16.bmodel --processor_name scribble_processor_fp16.bmodel --controlnet_img ../pics/scribble.png --prompt "royal chamber with fancy bed" --neg_prompt "worst quality" --num_inference_steps 20 --dev_id 0
```

运行结束后，生成的的图片保存为`results.png`。controlnet更多说明可参考[controlnet](../docs/Export_Controlnet.md)