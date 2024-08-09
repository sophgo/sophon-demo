# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 提示词参考](#22-提示词参考)
    * [2.3 测试文生图](#23-测试文生图)
    * [2.4 测试图生图](#24-测试图生图)

## 1. 环境准备

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt
```

> **注:**
>
> 上述命令安装的opencv是公版opencv，如果您希望使用sophon-opencv，可以设置如下环境变量：
> ```bash
> export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
> ```
> **若使用sophon-opencv需要保证python版本小于等于3.8。**

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

python/sdxl_t2i.py脚本文件参数说明：

```bash
usage: run.py [--model_path BMODELS_PATH] [--tokenizer TOKENIZER_FILE] [--tokenizer_2 TOKENIZER_2_FILE] [--prompt PROMPT] [--neg_prompt NEGATIVE_PROMPT] [--num_inference_steps ITERATION_NUMS] [--guidance_scale CFG parameter] [--dev_id DEV_ID]
--model_path: 各类bmodel文件的总目录;
--tokenizer tokenizer files路径;
--tokenizer_2 tokenizer_2 files路径;
--prompt 用于图像生成的提示词，词汇越靠前，权重越大;
--neg_prompt 用于图像生成的负面提示词，不希望图像中出现的内容;
--num_inference_steps Stable Diffusion的迭代次数;
--guidance_scale cfg参数;
--dev_id: 用于推理的tpu设备id;
```

python/sdxl_i2i.py脚本文件参数说明：

```bash
usage: run.py [--model_path BMODELS_PATH] [--tokenizer TOKENIZER_FILE] [--tokenizer_2 TOKENIZER_2_FILE] [--prompt PROMPT] [--neg_prompt NEGATIVE_PROMPT] [--init_img REFERENCED IMAGE] [--num_inference_steps ITERATION_NUMS] [--guidance_scale CFG parameter] [--strength INFLUENCE OF REFERENCED IMAGE] [--dev_id DEV_ID]
--model_path: 各类bmodel文件的总目录;
--tokenizer tokenizer files路径;
--tokenizer_2 tokenizer_2 files路径;
--prompt 用于图像生成的提示词，词汇越靠前，权重越大;
--neg_prompt 用于图像生成的负面提示词，不希望图像中出现的内容;
--init_img 参考图像;
--num_inference_steps Stable Diffusion的迭代次数;
--guidance_scale cfg参数;
--strength 参考图像的权重，越小则越接近参考图像，[0,1];
--dev_id: 用于推理的tpu设备id;
```

### 2.2 提示词参考

文生图的图像质量与提示词(prompt)高度相关，好的提示词可以生成更好的图像，提示词的构建可考虑如下几个角度：

**1. 正向提示词:**

- 内容主体：对象(cat; painting; a pair of lovers; boy; sorceress; rocket; doctor)，状态(angry; drinking; wearing jacket; sitting on the roof; playing basketball)，地点(in an empty square; at the bar; in forest)等。
- 画风：风格(digital painting; oil painting; photography; sketch; impressionist; hyperrealistic; modernist)，质量(HDR; high quality; masterpiece;)等。
- 色调：色彩(vivid color; black and white; iridescent gold)，光线(cinematic lighting; dark; rim light)等。

**2. 负向提示词:**

- 内容主体：对象(hand; limbs; mustache; poorly drawn feet)，环境(windy; light)等。
- 画风：风格(cartoon; pop-art; art nouveau)，质量(blurry; sharp; worst quality; deformed)等。

**3. 关键词权重**:

- ()和[]：()表示增加关键词权重，如(keyword)表示将keyword键词的权重提升为1.1倍，也可用(keyword: ratio)，表示keyword的权重为ratio倍。[]表示降低关键词权重，如[other keyword]表示将other keyword的权重降低1.1倍，也可用[other keyword: ratio]，表示other keyword的权重为1/ratio倍。

### 2.3 测试文生图

文生图若干测试实例如下:

```bash
cd python

python3 sdxl_t2i.py --model_path ../models/BM1684X --prompt "a rabbit driking at the bar" --neg_prompt "worst quality" --num_inference_steps 20 --dev_id 0

python3 sdxl_t2i.py --model_path ../models/BM1684X --prompt "a powerful mysterious sorceress, casting lightning magic, detailed clothing, digital painting, hyperrealistic, fantasy, Surrealist, upper body, artstation, highly detailed, sharp focus, stunningly beautiful, dystopian" --neg_prompt "worst quality" --num_inference_steps 50 --dev_id 0

python3 sdxl_t2i.py --model_path ../models/BM1684X --prompt "best quality, photography, vivid color, young boy, wearing jacket, short hair, sitting on the roof, the background are several tall buildings" --neg_prompt "worst quality" --num_inference_steps 50 --dev_id 0
```

代码运行结束后，生成的的图像保存为`t2i_result.png`。

### 2.4 测试图生图

图生图测试实例如下:

```bash
python3 sdxl_i2i.py --model_path ../models/BM1684X --prompt "A magician riding a grey donkey" --neg_prompt "worst quality" --init_img "../pics/astronaut.png" --num_inference_steps 50 --dev_id 0
```

运行结束后，生成的的图片保存为`i2i_results.png`。