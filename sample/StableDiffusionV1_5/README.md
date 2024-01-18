# StableDiffusionV1.5

## 目录
* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型](#3-准备模型)
  ​    [3.1 自己下载并且编译模型](#3.1-自己下载并且编译模型)
  ​    [3.2 使用准备好的模型文件](#3.2-使用准备好的模型文件)
* [4. 例程测试](#4-例程测试)
* [5. 运行性能测试](#5-运行性能测试)
* [6. FAQ](#6-FAQ)

## 1. 简介
StableDiffusion V1.5 是开源AIGC模型(Huggingface官网：https://huggingface.co/runwayml/stable-diffusion-v1-5)，可以依据文本提示生成相应的图像内容。

目前提供了python版的文本生成图像、controlnet插件辅助控制生成图像；

## 2. 特性

- 支持BM1684X(x86 PCIe、SoC)
- 支持FP32(BM1684X)、FP16(BM1684X)
- 基于sophon-sail的python推理，文生图和controlnet辅助生图两种模式

## 3. 准备模型

StableDiffusion V1.5暂时只支持在BM1684X上运行，模型来自于开源的Huggingface。本demo提供了singlize和multilize两种模型，基本的文生图模式使用singlize模型，controlnet插件需要使用multilize模型。

### 3.1 自己下载并且编译模型
用户若自己下载和编译singlize模型，请安装所需的第三方库（下载官方模型需要用户可以正常连接HuggingFace网站）：

```bash
pip3 install -r requirements.txt
pip3 install onnx==1.15.0
```

在script路径下，运行 export_pt_from_Huggingface.py 即可将Huggingface上pipeline中的模型以pt/onnx的格式保存在models文件夹下:

```bash
cd script
python3 export_pt_from_Huggingface.py
```

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

最后参考TPU-MLIR工具的使用方式激活对应的环境，并在script路径下执行四个bmodel导出脚本文件（get_text_encoder_bmodel.sh, get_vae_encoder_bmodel.sh, get_vae_decoder_bmodel.sh, get_unet_bmodel.sh），会将当前路径下的onnx/pt文件转换为bmodel，并将bmodel移入models文件夹下。

```bash
./get_text_encoder_bmodel.sh
./get_vae_encoder_bmodel.sh
./get_vae_decoder_bmodel.sh
./get_unet_bmodel.sh
```

由于multilize模型的转换相对复杂，本例程暂时只提供了转换好的multilize模型，还没有提供如何转换multilize模型的教程，用户可以参考[章节3.2](#32-使用准备好的模型文件)直接下载转换为bmodel的multilize模型，并配合controlnet插件使用；若用户想使用或者编译自己的controlnet，请参考[controlnet导出说明](./docs/Export_Controlnet.md)。

### 3.2 使用准备好的模型文件
在script路径下，可以执行download_singlize_bmodel.sh下载转换好的singlize模型，运行结束后会在 ../models/BM1684X/singlize路径下保存Stable Diffusion V1.5所需要的所有bmodel，并将下载好的pt/onnx文件保存到../models/onnx_pt/singlize中，用户可以使用准备好的bmodel，也可以用MLIR工具自行编译onnx_pt模型。

```bash
cd script
./download_singlize_bmodel.sh
```

用户还可以选择执行script路径下的download_multilize_bmodel.sh和download_controlnets_bmodel.sh，脚本执行完毕后，会在../models/BM1684X/multilize/路径下保存模型需要的bmodel，在../models/BM1684X/controlnets/路径下保存controlnet bmodel，在../models/BM1684X/processors/下保存配合controlnet所需的processor net bmodel，controlnet插件只能配合multilize模型使用。 

```bash
./download_multilize_bmodel.sh
./download_controlnets_bmodel.sh
```

在script目录下执行上述download脚本后，当前目录下的文件结构如下：

```
./models
├── BM1684X
│   ├── controlnets
│   │   ├── canny_controlnet_fp16.bmodel        # 使用TPU-MLIR编译，用于BM1684X的FP16 canny controlnet
│   │   ├── depth_controlnet_fp16.bmodel        # 使用TPU-MLIR编译，用于BM1684X的FP16 depth controlnet
│   │   ├── hed_controlnet_fp16.bmodel          # 使用TPU-MLIR编译，用于BM1684X的FP16 hed controlnet
│   │   ├── openpose_controlnet_fp16.bmodel     # 使用TPU-MLIR编译，用于BM1684X的FP16 openpose controlnet
│   │   ├── scribble_controlnet_fp16.bmodel     # 使用TPU-MLIR编译，用于BM1684X的FP16 scribble controlnet
│   │   └── segmentation_controlnet_fp16.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 segmentation controlnet
│   ├── multilize
│   │   ├── text_encoder_1684x_f32.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 text encoder BModel，最大编码长度为77
│   │   ├── unet_multize.bmodel                 # 使用TPU-MLIR编译，用于BM1684X的FP16 多尺度unet，可配合controlnet使用
│   │   ├── vae_decoder_multize.bmodel          # 使用TPU-MLIR编译，用于BM1684X的FP16 多尺度vae decoder
│   │   └── vae_encoder_multize.bmodel          # 使用TPU-MLIR编译，用于BM1684X的FP16 多尺度vae encoder
│   ├── processors
│   │   ├── depth_processor_fp16.bmodel         # 使用TPU-MLIR编译，用于BM1684X的FP16 depth processor
│   │   ├── hed_processor_fp16.bmodel           # 使用TPU-MLIR编译，用于BM1684X的FP16 hed processor
│   │   ├── openpose_body_fp16.bmodel           # 使用TPU-MLIR编译，用于BM1684X的FP16 openpose body processor
│   │   ├── openpose_face_fp16.bmodel           # 使用TPU-MLIR编译，用于BM1684X的FP16 openpose face processor
│   │   ├── openpose_hand_fp16.bmodel           # 使用TPU-MLIR编译，用于BM1684X的FP16 openpose hand processor
│   │   ├── scribble_processor_fp16.bmodel      # 使用TPU-MLIR编译，用于BM1684X的FP16 scribble processor
│   │   └── segmentation_processor_fp16.bmodel  # 使用TPU-MLIR编译，用于BM1684X的FP16 segmentation processor
│   └── singlize
│       ├── text_encoder_1684x_f32.bmodel       # 使用TPU-MLIR编译，用于BM1684X的FP32 text encoder BModel，最大编码长度为77
│       ├── unet_1684x_f16.bmodel               # 使用TPU-MLIR编译，用于BM1684X的FP16 单尺度unet，只能生成512*512图像
│       ├── vae_decoder_1684x_f16.bmodel        # 使用TPU-MLIR编译，用于BM1684X的FP16 单尺度vae decoder
│       └── vae_encoder_1684x_f16.bmodel        # 使用TPU-MLIR编译，用于BM1684X的FP16 单尺度vae encoder
├── onnx_pt
│   ├── text_encoder_1684x_f32.onnx             # 导出的text encoder的onnx模型，用户自行使用
│   ├── unet_fp32.pt                            # 单尺度unet，用户自行使用
│   ├── vae_decoder_singlize.pt                 # 单尺度vae decoder，用户自行使用
│   └── vae_encoder_singlize.pt                 # 单尺度vae encoder，用户自行使用
└── tokenizer_path
    ├── merges.txt                              # CLIPTokenizer参考的token合并文件
    ├── special_tokens_map.json                 # 特殊token映射
    ├── tokenizer_config.json                   # CLIPTokenizer配置文件
    ├── tokenizer.json                          # tokenizer文件
    └── vocab.json                              # 字典文件
```

## 4. 例程测试
- [Python例程](./python/README.md)

## 5. 运行性能测试

图像生成的总体时间与设定的迭代次数相关，此处设定迭代20次，性能如下:

|   测试平台    |    测试模式    | 模型格式 | text_encoder_time | inference_time | vae_decoder time |
| -----------  | ------------- | -------- | --------------- | -------------  | ---------------- |
| BM1684X SoC  |    text2img   |   fp32   |      51.81      |    5808.30     |     473.05       |
| BM1684X SoC  |   controlnet  |   fp32   |      51.79      |    10462.45    |     471.20       |

## 6. FAQ
[常见问题解答](../../docs/FAQ.md)