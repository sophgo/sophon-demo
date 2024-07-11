# StableDiffusionXL

## 目录
* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型](#3-准备模型)
  * [3.1 自己下载并且编译模型](#31-自己下载并且编译模型)
  * [3.2 使用准备好的模型文件](#32-使用准备好的模型文件)
* [4. 运行环境准备](#4-运行环境准备)
* [5. 例程测试](#5-例程测试)
* [6. 运行性能测试](#6-运行性能测试)
* [7. FAQ](#7-FAQ)

## 1. 简介
StableDiffusionXL 是开源AIGC模型:[Huggingface官网stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)，可以依据文本提示生成相应的图像内容，目前支持了python版的文本生成图像和图像生成图像。该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，若bmodel推理出现NAN，或生成图像全黑的情况，请注意驱动版本是否达到要求。

## 2. 特性

- 支持BM1684X(x86 PCIe、SoC)
- 支持FP32(BM1684X)、FP16(BM1684X)
- 基于sophon-sail的python推理，文生图和图生图两种模式

## 3. 准备模型

StableDiffusionXL暂时只支持在BM1684X上运行，模型来自于开源的Huggingface，可生成``1024*1024``大小的图像。用户在用cpu导出`onnx/pt`模型时，运行内存占用约64G。

### 3.1 自己下载并且编译模型
用户若自己下载和编译模型，请安装所需的第三方库（下载官方模型需要用户可以正常连接HuggingFace网站）：

```bash
pip3 install -r requirements.txt
```

在scripts路径下，运行export_models_from_Huggingface.py 即可将Huggingface上pipeline中的部件模型以pt/onnx的格式保存在models文件夹下:

```bash
cd scripts
python3 export_models_from_Huggingface.py
```

**注意：**若执行上述导出脚本时，出现无法连接Huggingface的情况，可使用如下指令从镜像站下载模型（仅在当前终端生效）：

```bash
pip3 install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
python3 export_models_from_Huggingface.py
```

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)创建并进入docker环境，**注意：**请在docker中使用如下指令安装mlir:

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:/aigc/tpu_mlir-1.6.502-py3-none-any.whl
```

安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)相应版本的SDK中获取)。

最后参考TPU-MLIR工具的使用方式激活对应的环境，并在scripts路径下执行四个bmodel的导出脚本文件（get_text_encoder_bmodel.sh, get_vae_encoder_bmodel.sh, get_vae_decoder_bmodel.sh, get_unet_bmodel.sh），会将models/onnx_pt/下的pt文件转换为bmodel，并将bmodel移入models/BM1684X文件夹下。

```bash
./get_text_encoder_bmodel.sh
./get_unet_bmodel.sh
./get_vae_encoder_bmodel.sh
./get_vae_decoder_bmodel.sh
```

### 3.2 使用准备好的模型文件
在scripts路径下，可以执行download.sh下载转换好的bmodel模型，运行结束后会在 ../models/BM1684X路径下保存Stable DiffusionXL所需要的所有bmodel，并将下载好的pt/onnx文件保存到../models/onnx_pt/中，用户可以使用准备好的bmodel，也可以用MLIR工具自行编译onnx_pt模型。

```bash
cd scripts
./download.sh
```

在scripts目录下执行上述download脚本后，当前目录下的文件结构如下：

```
./models
├── BM1684X
│   ├── text_encoder_1_1684x_f32.bmodel		# 使用TPU-MLIR编译，用于BM1684X的FP32 text encoder 1 BModel，最大编码长度为77
│   ├── text_encoder_2_1684x_f16.bmodel		# 使用TPU-MLIR编译，用于BM1684X的FP16 text encoder 2 BModel，最大编码长度为77
│   ├── unet_base_1684x_bf16.bmodel			  # 使用TPU-MLIR编译，用于BM1684X的BF16 unet base，对应图像尺寸为1024*1024
│   ├── vae_decoder_1684x_bf16.bmodel		  # 使用TPU-MLIR编译，用于BM1684X的BF16 vae decoder
│   └── vae_encoder_1684x_bf16.bmodel		  # 使用TPU-MLIR编译，用于BM1684X的BF16 vae encoder
└── onnx_pt
    ├── text_encoder_1
    │   └── text_encoder_1.onnx				    # 导出的text encoder 1的onnx模型，用户自行使用
    ├── text_encoder_2
    │   └── text_encoder_2.onnx				    # 导出的text encoder 2的onnx模型，用户自行使用
    ├── unet
    │   └── unet_base.pt					        # 导出的unet base的pt模型，用户自行使用
    ├── vae_decoder
    │   └── vae_decoder.pt					      # 导出的vae decoder的pt模型，用户自行使用
    └── vae_encoder
        └── vae_encoder.pt					      # 导出的vae encoder的pt模型，用户自行使用
```

## 4. 运行环境准备

在PCIe上无需修改内存，以下为soc模式相关：

对于1684X系列设备（如SE7/SM7），都可以通过这种方式完成环境准备，使得满足StableDiffusionXL的运行条件。首先，在1684x SoC环境上，参考如下命令修改设备内存:

```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/09/11/13/DeviceMemoryModificationKit.tgz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_{vx.x}.tar.xz #vx.x是版本号
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息
./memory_edit.sh -c -npu 7615 -vpu 3072 -vpp 3072 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot
```


## 5. 例程测试
- [Python例程](./python/README.md)

## 6. 运行性能测试

图像生成的总体时间与设定的迭代次数相关，此处设定迭代20次，图像大小为(1024, 1024)，性能如下（单位ms）:

|   测试平台    |    测试模式     | text_encoder_time | inference_time | vae_encoder_time | vae_decoder_time |
| -----------  | ------------- | ---------------   | -------------  | ---------------- | ---------------- |
| BM1684X SoC  |    text2img   |      178.96      |    37450.7    |    null          |     2099.22     |
| BM1684X SoC  |    img2img    |      178.96      |    37450.7    |    1195.56    |     2099.22      |

## 7. FAQ
[常见问题解答](../../docs/FAQ.md)
