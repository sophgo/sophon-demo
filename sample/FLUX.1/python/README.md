# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 提示词参考](#22-提示词参考)
    * [2.3 测试文生图](#23-测试文生图)

## 1. 环境准备

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon(>=V24.04.01)和sophon-sail[指定版本](../README.md#3-运行环境准备)，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，sophon-sail包已在[运行环境准备](../README.md#3-运行环境准备)中提供，具体编译方法可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库（**安装pytorch提示python3.8.2版本过低时，可选择安装pytorch 1.13.1版**）：

```bash
pip3 install -r requirements.txt
# python3.8.2可安装pytorch==1.31.1版
pip3 install torch==1.13.1
```

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

`python/run.py`脚本文件参数说明：

```bash
--flux_type: flux.1的类型，dev或schnell;
--model_path: bmodel文件的总目录;
--chip_type: 芯片类型，目前仅支持bm1684X和bm1688;
--quant_type: transformer主体结构的量化方式，单芯运行选W4BF16，三芯运行选BF16;
--prompt: clip的提示词;
--prompt_2: t5的提示词，若不给提示词则和prompt保持一致;
--num_inference_steps: 迭代/去噪 次数;
--guidance_scale: cfg参数，仅flux.1-dev支持;
--dev_ids: 用于推理的tpu设备id;单芯输入设备号，如 0;三芯输入3个设备号，如 0 1 2;
--tiny_vae: 是否使用tiny_vae，单芯模式下使用，可减少显存占用;
--seed: 随机种子，0 ~ 2^32 - 1
```

`python/web.py`脚本文件参数说明：

```shell
--flux_type: flux.1的类型，dev或schnell;
--model_path: bmodel文件的总目录;
--chip_type: 芯片类型，目前仅支持bm1684X和bm1688;
--quant_type: transformer主体结构的量化方式，单芯运行选W4BF16，三芯运行选BF16;
--dev_ids: 用于推理的tpu设备id;单芯输入设备号，如 0;三芯输入3个设备号，如 0 1 2;
--tiny_vae: 是否使用tiny_vae，单芯模式下使用，可减少显存占用;
```

### 2.2 提示词参考

文生图的图像质量与提示词(prompt)高度相关，好的提示词可以生成更好的图像，提示词的构建可考虑如下几个角度：

- 内容主体：对象(cat; painting; a pair of lovers; boy; sorceress; rocket; doctor)，状态(angry; drinking; wearing jacket; sitting on the roof; playing basketball)，地点(in an empty square; at the bar; in forest)等。
- 画风：风格(digital painting; oil painting; photography; sketch; impressionist; hyperrealistic; modernist)，质量(HDR; high quality; masterpiece;)等。
- 色调：色彩(vivid color; black and white; iridescent gold)，光线(cinematic lighting; dark; rim light)等。

### 2.3 测试文生图

`run.py`文生图若干测试实例如下:

```bash
cd python
# 单芯dev版
python3 run.py --prompt "a rabbit drinking at the bar" --num_inference_steps 10 --quant_type W4BF16 --dev_ids 0 --tiny_vae --chip_type bm1684x
python3 run.py --prompt "a rabbit drinking at the bar" --num_inference_steps 10 --quant_type W4BF16 --dev_ids 0 --tiny_vae --chip_type bm1688

# 单芯schnell版
python3 run.py --prompt "a rabbit drinking at the bar" --num_inference_steps 4 --quant_type W4BF16 --dev_ids 0 --tiny_vae --flux_type schnell --chip_type bm1684x
python3 run.py --prompt "a rabbit drinking at the bar" --num_inference_steps 4 --quant_type W4BF16 --dev_ids 0 --tiny_vae --flux_type schnell --chip_type bm1688

# 3芯dev版
python3 run.py --prompt "a powerful mysterious sorceress, casting lightning magic, detailed clothing, digital painting, hyperrealistic, fantasy, Surrealist, upper body, artstation, highly detailed, sharp focus, stunningly beautiful, dystopian" --num_inference_steps 10 --dev_ids 0 1 2 --quant_type BF16 --chip_type bm1684x

# 3芯schnell版
python3 run.py --prompt "a powerful mysterious sorceress, casting lightning magic, detailed clothing, digital painting, hyperrealistic, fantasy, Surrealist, upper body, artstation, highly detailed, sharp focus, stunningly beautiful, dystopian" --num_inference_steps 10 --dev_ids 0 1 2 --quant_type BF16  --flux_type schnell --chip_type bm1684x
```

代码运行结束后，生成的的图像保存为`result.png`。

`web.py`文生图测试如下，根据终端提示网址打开网页即可访问，默认为http://0.0.0.0:7860：

```shell
cd python
# dev
python3 web.py --quant_type W4BF16 --flux_type dev --dev_ids 0 --tiny_vae --chip_type bm1684x
python3 web.py --quant_type W4BF16 --flux_type dev --dev_ids 0 --tiny_vae --chip_type bm1688
python3 web.py --quant_type BF16 --flux_type dev --dev_ids 0 1 2 --chip_type bm1684x

#schnell
python3 web.py --quant_type W4BF16 --flux_type schnell --dev_ids 0 --tiny_vae --chip_type bm1684x
python3 web.py --quant_type W4BF16 --flux_type schnell --dev_ids 0 --tiny_vae --chip_type bm1688
python3 web.py --quant_type BF16 --flux_type schnell --dev_ids 0 1 2 --chip_type bm1684x
```

代码运行结束后，生成的图像保存为`generated_image_{%Y%m%d_%H%M%S}.png`。
