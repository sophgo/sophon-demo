# FLUX.1

## 目录

	- [1. 简介](#1-简介)
	- [2. 特性](#2-特性)
	- [3. 运行环境准备](#3-运行环境准备)
	- [4. 准备模型](#4-准备模型)
	- [5. 例程测试](#5-例程测试)
	- [6. 程序性能测试](#6-程序性能测试)

## 1. 简介

FLUX.1-dev/schnell是black-forest开源的文本生成图像模型，关于它的特性，可前往HuggingFace源repo查看：[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)和[FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)。 本例程对FLUX.1-dev和FLUX.1-schnell进行移植，使之能在SOPHON BM1684X上进行推理测试。该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，且需要安装较新的sophon-sail，可参考[运行环境准备](#3-运行环境准备)来安装本例程需要的sophon-sail版本。

## 2. 特性

* 支持BM1684X(x86 PCIe、SoC)
* 支持BF16(3芯运行)、W4BF16(单芯运行)模型编译和推理
* 支持基于SAIL推理的Python例程


## 3. 运行环境准备

**安装sophon-sail**

本项目使用的sophon-sail版本较新，可按照如下指令下载sophon-sail源码：

```shell
pip3 install dfss --upgrade #安装dfss依赖
# sophon-sail
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/sophon-sail.tar.gz
tar xvf sophon-sail.tar.gz
```

参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#)编译不包含bmcv,sophon-ffmpeg,sophon-opencv的可被Python3接口调用的Wheel文件。

若在**soc模式**运行本项目，且使用刷机后默认的`python3.8.2`运行本项目，可通过whl包方式直接安装sophon-sail:

```shell
pip3 install dfss --upgrade #安装dfss依赖
python3 -m dfss --url=open@sophgo.com:sophon-demo/FLUX_1/sophon_arm-3.9.0-py3-none-any.whl # for SE7 py38
pip3 install sophon_arm-3.9.0-py3-none-any.whl --force-reinstall
```

**修改soc模式下的内存**

在PCIe上无需修改内存，以下为soc模式相关：
对于1684X系列设备（如SE7/SM7）参考如下命令修改设备内存。

```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/09/11/13/DeviceMemoryModificationKit.tgz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_{vx.x}.tar.xz #vx.x是版本号
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息

#如果是1684x系列设备，执行以下命令
./memory_edit.sh -c -npu 7615 -vpu 2360 -vpp 2360 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot
```

> **注意：**
>
> 1. tpu总内存为npu/vpu/vpp三者之和。
> 2. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)

## 4. 准备模型

已提供编译好的bmodel。

### 4.1 使用提供的模型

本例程在`scripts`目录下提供了下载脚本`download.sh`

```bash
# 可选BF16或W4BF16版本，对应单芯运行和三芯运行
# soc模式下必选W4BF16且解码使用tiny-vae
./scripts/download.sh
# 参数可选如下，分别对应：transformer主体结构的量化方式，flux的版本，是否使用tiny-vae(若在soc模式运行必选W4BF16和tiny-vae)
# --quantize BF16/W4BF16 
# --flux_type dev/schnell
# --use_taef1 1/0
```

执行下载脚本下载**所有模型**后，./models目录下的文件如下：

```bash
models
├── BM1684X
│   ├── clip.bmodel					# 使用TPU-MLIR编译，用于BM1684X的BF16 clip编码器，最大编码长度为77
│   ├── dev_bf16_transformer_on_device0.bmodel		# 使用TPU-MLIR编译，用于BM1684X的BF16 FLUX.1-dev，加载到三芯时逻辑上的第一颗
│   ├── dev_bf16_transformer_on_device1.bmodel		# 使用TPU-MLIR编译，用于BM1684X的BF16 FLUX.1-dev，加载到三芯时逻辑上的第二颗
│   ├── dev_bf16_transformer_on_device2.bmodel		# 使用TPU-MLIR编译，用于BM1684X的BF16 FLUX.1-dev，加载到三芯时逻辑上的第三颗
│   ├── dev_w4bf16_transformer.bmodel		        # 使用TPU-MLIR编译，用于BM1684X的W4BF16 FLUX.1-dev，单芯运行时使用
│   ├── schnell_bf16_transformer_on_device0.bmodel	# 使用TPU-MLIR编译，用于BM1684X的BF16 FLUX.1-dev，加载到三芯时，逻辑上的第一颗
│   ├── schnell_bf16_transformer_on_device1.bmodel	# 使用TPU-MLIR编译，用于BM1684X的BF16 FLUX.1-dev，加载到三芯时，逻辑上的第二颗
│   ├── schnell_bf16_transformer_on_device2.bmodel	# 使用TPU-MLIR编译，用于BM1684X的BF16 FLUX.1-dev，加载到三芯时，逻辑上的第三颗
│   ├── schnell_w4bf16_transformer.bmodel		# 使用TPU-MLIR编译，用于BM1684X的W4BF16 FLUX.1-schnell，单芯运行时使用
│   ├── tiny_vae_decoder_bf16.bmodel		        # 使用TPU-MLIR编译，用于BM1684X的BF16 tiny-vae，pcie/soc模式下使用
│   ├── vae_decoder_bf16.bmodel				# 使用TPU-MLIR编译，用于BM1684X的BF16 vae，pcie模式下使用
│   └── w4bf16_t5.bmodel				# 使用TPU-MLIR编译，用于BM1684X的W4BF16 t5编码器，最大编码长度为512
├── ids_emb.pt						# 1024*1024图像的空间位置编码结果，常量值
├── tokenizer						# clip的提词器文件
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
└── tokenize_2						# t5的提词器文件
		├── special_tokens_map.json
		├── spiece.model
		├── tokenizer_config.json
		└── tokenizer.json
```

### 4.2 自行编译模型

用户若自己下载和编译模型，请安装所需的第三方库以导出onnx模型（下载官方模型需要用户可以正常连接HuggingFace网站）：

```bash
pip3 install -r requirements.txt
```

**替换diffusers-0.30.0的文件：**

导出onnx/pt文件前，请将`tools/diffusers-0.30.0/attention_processor.py`文件替换到**当前python环境对应的diffusers安装目录**下，具体为`/path-to-site-packages/diffusers/models/attention_processor.py`。

```shell
cp tools/diffusers-0.30.0/attention_processor.py /path-to-site-packages/diffusers/models/attention_processor.py
```

在scripts路径下，运行export_models_from_HF.py 即可将Huggingface上pipeline中的部件模型以pt/onnx的格式保存在models文件夹下（FLUX全模型较大，建议在内存>=96G的环境导出）:

```bash
cd scripts
python3 export_models_from_HF.py
```

若执行上述导出脚本时，出现无法连接Huggingface的情况，可参考[从镜像站下载模型](https://hf-mirror.com/)，建议使用hfd工具下载，然后将导出脚本中的`from_pretained`接口改为本地路径。

模型编译前需要安装TPU-MLIR(注意，若在docker环境导出onnx模型，安装TPU-MLIR时请另起一个新docker环境，避免依赖冲突)，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)创建并进入docker环境，注意：请在docker中使用如下指令安装mlir:

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/tpu_mlir-1.10b0-py3-none-any.whl
# 安装tpu-mlir
pip3 install tpu_mlir-1.10b0-py3-none-any.whl
# 升级pytorch>=2.1.0(docker内的2.0.1+cpu版本，在导出时会报错)
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

安装好后需在TPU-MLIR环境中进入本例程所在目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)相应版本的SDK中获取)。

最后参考TPU-MLIR工具的使用方式激活对应的环境（若为pip安装tpu-mlir方式可不用再进行激活操作），并在scripts路径下执行bmodel的导出脚本文件`gen_bmodels.sh`，会将models/onnx_pt/下的pt文件转换为bmodel，并将bmodel移入models/BM1684X文件夹下。

```bash
./gen_bmodels.sh
# 参数可选如下，分别对应：transformer主体结构的量化方式，flux的版本，是否使用tiny-vae(若在soc模式运行必选W4BF16和tiny-vae)
# --quantize BF16/W4BF16 
# --flux_type dev/schnell
# --use_taef1 1/0
```

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

soc模式下性能数据：

| 测试平台  | 测试程序  |   测试模型     |  模型加载耗时 | 10步生图耗时  | bm-smi显存占用 |
| -------- | -------- | -------------- | ------------ | ------------ | -------------- |
| SE7-32   | run.py   | schnell_w4bf16 | 224.65 s     | 163.34 s     | 11419MB        |
| SE7-32   | run.py   | dev_w4bf16     | 224.37 s     | 163.27 s     | 11438MB        |


> **测试说明**：  
>
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 3. 这里使用的SDK版本是BM1684X V24.04.01；
