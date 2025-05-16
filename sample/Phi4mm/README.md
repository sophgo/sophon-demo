# Phi4mm

## 目录
- [Phi4mm](#phi4mm)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 准备模型](#3-准备模型)
    - [3.1 使用提供的模型](#31-使用提供的模型)
    - [4.2 自行编译模型](#42-自行编译模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 程序性能测试](#6-程序性能测试)

## 1. 简介
Phi4mm是microsoft推出的全模态大模型，可以融合文本、视觉、语音三种模态的数据，原始仓库见[Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)。

本例程对Phi-4-multimodal-instruct模型进行移植，使其可在Sophon BM1684X芯片上运行。PCIE模式下，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行。在1684X SoC设备（如SE7、SM7、Airbox等）上，支持在V24.04.01(libsophon_0.5.1)SDK上运行。

## 2. 特性

### 2.1 目录结构说明

```bash
├── python                # 存放Python例程及其README
|   ├──README.md 
|   ├──phi4mm.py          # 使用SAIL推理的Python例程
|   ├──requirements.txt   # 运行环境上需要安装的第三方依赖
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译等shell脚本
└── tools                 # 存放onnx导出等python脚本
```

### 2.2 SDK特性

* 支持BM1684X(x86 PCIe、SoC)
* LLM语言部分支持INT4，视觉部分支持BF16
* 支持基于SAIL推理的Python例程
  
## 3. 准备模型

### 3.1 使用提供的模型

可以通过以下命令下载我们编译好的模型和对应的processor，以及测试图片和测试音频。

```bash
./scripts/download.sh #文件会直接放在python/目录下
```

### 4.2 自行编译模型

Phi4mm模型导出onnx需要依赖[Phi4mm官方仓库](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)，目前只支持在x86主机进行模型编译。  

**注意:** 用cpu转模型需要保证运行内存至少64G以上，需要存储空间60G以上，请确有足够的内存完成对应的操作。  

```bash
pip3 install -r tools/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple #安装导出模型所需依赖
git lfs install
git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct
```

如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。  

将源模型仓库完整拉取后，找到`vision_siglip_navit.py`这个文件，将第558行的Conv2d算子参数`padding`修改为`0`：
```python
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0 #"valid", 
        )
```

并替换config文件：
```bash
cp tools/files/config.json /path/to/Phi-4-multimodal-instruct/ 
```

在`tools`文件夹下，运行`model_export.py`脚本即可导出onnx模型，并存放在`models/onnx_seq${seq_length}`文件夹下，指令如下：

```bash
cd tools
python3 model_export.py --model_path /path/to/Phi-4-multimodal-instruct/ --seq_length 512 --out_dir ./tmp
```

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

在tpu-mlir环境中执行编译脚本，会将`models/onnx/`下的文件转换为bmodel。

```bash
cd scripts
./gen_bmodel.sh --mode int4 --name phi4mm --chip bm1684x --onnx_dir ../tools/tmp/onnx_seq512 #会在当前目录下生成phi4mm_bm1684x_int4_1core.bmodel
```

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

测试输入为："What is shown in this image?"，测试图片为`australia.jpg`，测试音频为空：
|    测试平台   |               测试模型                   |first token latency(s)|token per second(tokens/s)| 
| -----------  | -------------------------------------- | --------------------- | ----------------------- | 
| SE7-32       | phi4mm_bm1684x_int4_1core.bmodel       |   0.878               | 9.163 |
 
测试输入为空，测试图片为`australia.jpg`，测试音频为`what_is_shown_in_this_image.wav`：
|    测试平台   |               测试模型                   |first token latency(s)|token per second(tokens/s)| 
| -----------  | -------------------------------------- | --------------------- | ----------------------- | 
| SE7-32       | phi4mm_bm1684x_int4_1core.bmodel       |   0.901               | 8.083 |

测试输入为"Based on the attached audio, generate a comprehensive text transcription of the spoken content."，测试图片为空，测试音频为`what_is_shown_in_this_image.wav`：
|    测试平台   |               测试模型                   |first token latency(s)|token per second(tokens/s)| 
| -----------  | -------------------------------------- | --------------------- | ----------------------- | 
| SE7-32       | phi4mm_bm1684x_int4_1core.bmodel       |  0.693                | 9.187 |
> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. 加载模型和第一次问答可能会比较慢，后续几次就会恢复正常。