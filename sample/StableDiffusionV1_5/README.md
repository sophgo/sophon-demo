# StableDiffusionV1.5

## 目录
* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型](#3-准备模型)
  * [3.1 自己下载并且编译模型](#31-自己下载并且编译模型)
  * [3.2 使用准备好的模型文件](#32-使用准备好的模型文件)
* [4. 例程测试](#4-例程测试)
* [5. 运行性能测试](#5-运行性能测试)
* [6. FAQ](#6-FAQ)

## 1. 简介
StableDiffusion V1.5 是开源AIGC模型:[Huggingface官网stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)，可以依据文本提示生成相应的图像内容。

目前提供了python版的文本生成图像、controlnet插件辅助控制生成图像；

## 2. 特性

- 支持BM1684X(x86 PCIe、SoC)
- 支持FP32(BM1684X)、FP16(BM1684X)
- 基于sophon-sail的python推理，文生图和controlnet辅助生图两种模式

## 3. 准备模型

StableDiffusion V1.5暂时只支持在BM1684X上运行，模型来自于开源的Huggingface。本demo提供了singlize和multilize两种模型，基本的文生图模式使用singlize模型，可生成``512*512``大小的图像；multilize模型可使用controlnet插件控制图像生成内容，并支持如下46种不同的图像尺度（高，宽），尺度最大的(512,896)在用cpu导出时，运行内存占用约20G，外存占用约8G，用户内存资源不足时，请删除export*.py脚本中img_size列表里不需要的尺度，仅保留一个尺度进行导出：

```
(128, 384), (128, 448), (128, 512), (192, 384), (192, 448), (192, 512), (256, 384), 
(256, 448), (256, 512), (320, 384), (320, 448), (320, 512), (384, 384), (384, 448), 
(384, 512), (448, 448), (448, 512), (512, 512), (512, 576), (512, 640), (512, 704),
(512, 768), (512, 832), (512, 896), (768, 768), (384, 128), (448, 128), (512, 128),
(384, 192), (448, 192), (512, 192), (384, 256), (448, 256), (512, 256), (384, 320),
(448, 320), (512, 320), (448, 384), (512, 384), (512, 448), (576, 512), (640, 512), 
(704, 512), (768, 512), (832, 512), (896, 512)
```

### 3.1 自己下载并且编译模型
用户若自己下载和编译singlize模型，请安装所需的第三方库（下载官方模型需要用户可以正常连接HuggingFace网站）：

```bash
pip3 install -r requirements.txt
pip3 install onnx==1.15.0
```

在scripts路径下，运行export_singlize_pt_from_Huggingface.py 即可将Huggingface上pipeline中的singlize模型以pt/onnx的格式保存在models文件夹下:

```bash
cd scripts
python3 export_singlize_pt_from_Huggingface.py
```

**注意：**若执行上述导出脚本时，出现无法连接Huggingface的情况，可使用如下指令从镜像站下载模型（仅在当前终端生效）：

```bash
pip3 install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
python3 export_singlize_pt_from_Huggingface.py
```

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)创建并进入docker环境，**注意：**请在docker中使用如下指令安装mlir:

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:/aigc/tpu_mlir-1.6.404-py3-none-any.whl
```

安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

最后参考TPU-MLIR工具的使用方式激活对应的环境，并在scripts路径下执行四个bmodel的导出脚本文件（get_text_encoder_bmodel.sh, get_vae_encoder_bmodel.sh, get_vae_decoder_bmodel.sh, get_unet_bmodel.sh），会将models/onnx_pt/下的pt文件转换为bmodel，并将bmodel移入models/BM1684X/singlize文件夹下。

```bash
./get_text_encoder_bmodel.sh
./get_vae_encoder_bmodel.sh
./get_vae_decoder_bmodel.sh
./get_unet_bmodel.sh
```

用户若需要multilize模型，在scripts路径下，运行export_multilize_pt_from_Huggingface.py即可将Huggingface上pipeline中的multilize模型以pt/onnx的格式保存在models/onnx_pt/multilize/文件夹下:

```bash
cd scripts
python3 export_multilize_pt_from_Huggingface.py
```

准备并激活TPU-MLIR环境，在scripts路径下执行四个bmodel的导出脚本文件（get_text_encoder_bmodel.sh, get_mul_vae_encoder_bmodel.sh, get_mul_vae_decoder_bmodel.sh, get_mul_unet_bmodel.sh），会将生成的bmodel移入models/BM1684X/multilize文件夹下。

```bash
./get_text_encoder_bmodel.sh
./get_mul_vae_encoder_bmodel.sh
./get_mul_vae_decoder_bmodel.sh
./get_mul_unet_bmodel.sh
```

multilize模型可配合controlnet使用，控制图像生成的内容，结构。若用户想使用或者编译自己的controlnet，请参考[controlnet导出说明](./docs/Export_Controlnet.md)。

### 3.2 使用准备好的模型文件
在scripts路径下，可以执行download_singlize_bmodel.sh下载转换好的singlize模型，运行结束后会在 ../models/BM1684X/singlize路径下保存Stable Diffusion V1.5所需要的所有bmodel，并将下载好的pt/onnx文件保存到../models/onnx_pt/singlize中，用户可以使用准备好的bmodel，也可以用MLIR工具自行编译onnx_pt模型。

```bash
cd scripts
./download_singlize_bmodel.sh
```

用户还可以选择执行scripts路径下的download_multilize_bmodel.sh和download_controlnets_bmodel.sh，脚本执行完毕后，会在../models/BM1684X/multilize/路径下保存模型需要的bmodel，在../models/BM1684X/controlnets/路径下保存controlnet bmodel，在../models/BM1684X/processors/下保存配合controlnet所需的processor net bmodel，controlnet插件只能配合multilize模型使用。 

```bash
./download_multilize_bmodel.sh
./download_controlnets_bmodel.sh
```

在scripts目录下执行上述download脚本后，当前目录下的文件结构如下：

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
│   │   ├── openpose_body_processor_fp16.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 openpose body processor
│   │   ├── openpose_face_processor_fp16.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 openpose face processor
│   │   ├── openpose_hand_processor_fp16.bmodel # 使用TPU-MLIR编译，用于BM1684X的FP16 openpose hand processor
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

图像生成的总体时间与设定的迭代次数相关，此处设定迭代20次，图像大小为(512,512)，性能如下:

|   测试平台    |    测试模式     |                        模型格式                         | text_encoder_time | inference_time | vae_decoder time |
| -----------  | ------------- | ------------------------------------------------------ | --------------- | -------------  | ---------------- |
| BM1684X SoC  |    text2img   |   text_encoder fp32 + singlize unet/vae_decoder fp16   |      50.61      |    4808.42     |     493.20       |
| BM1684X SoC  |   controlnet  |   text_encoder fp32 + multilize unet/vae_decoder fp16  |      50.69      |    9223.65     |     493.20       |

## 6. FAQ
[常见问题解答](../../docs/FAQ.md)
