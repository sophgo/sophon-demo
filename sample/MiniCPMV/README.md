# MiniCPMV

## 目录
- [MiniCPMV](#minicpmV)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型](#4-准备模型)
    - [4.1 使用提供的模型](#41-使用提供的模型)
    - [4.2 自行编译模型](#42-自行编译模型)
      - [4.2.1 从Huggingface下载MiniCPMV](#421-从huggingface下载MiniCPM-V-4)
      - [4.2.2 下载TPU-MLIR docker，并配置TPU-MLIR编译环境](#422-下载tpu-mlir-docker并配置tpu-mlir编译环境)
      - [4.2.3 编译模型生成bmodel](#423-编译模型生成bmodel)
  - [5. 例程测试](#5-例程测试)
  - [6. 程序性能测试](#6-程序性能测试)

## 1. 简介
MiniCPM-V 4.0 是MiniCPM-V系列中的最新高效模型。该模型基于SigLIP2-400M和MiniCPM4-3B构建，总共有4.1B个参数。它继承了MiniCPM-V 2.6强大的单图、多图及视频理解性能，并大幅提高了效率。关于它的特性，请前往源repo查看：https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4-AWQ。 本例程对MiniCPM-V-4进行移植，使之能在SOPHON BM1684X、SOPHON BM1688以及SOPHON CV186AH上进行推理测试。

对于BM1684X，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在BM1684X SoC设备（如SE7、SM7、Airbox等）上运行。对于BM1688、CV186AH，支持在V1.8及以上版本的SoC设备（SE9、SM9）运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持BM1688&CV186AH(SoC)
* 支持W4A16模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持多轮对话


## 3. 运行环境准备
在PCIe上无需修改内存，以下为soc模式相关：
对于SoC产品，如SE7、SE9等，可以通过这种方式完成环境准备，使得满足MiniCPMV运行条件。参考如下命令修改设备内存。
```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://github.com/sophgo/sophon-tools/releases/download/v24.09.21/memory_edit_v2.10.tar.xz
tar xvf memory_edit_v2.10.tar.gz
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息

#如果是BM1684x系列设备，执行以下命令。BM1688、SV186AH系列产品可不用更改。
./memory_edit.sh -c -npu 7615 -vpu 3072 -vpp 3072 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot
```


> **注意：**
> 1. tpu总内存为npu/vpu/vpp三者之和。
> 2. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)

## 4. 准备模型
已提供编译好的bmodel。
### 4.1 使用提供的模型

​本例程在`scripts`目录下提供了下载脚本`download.sh`

```bash
# minicpmv 1684x
./scripts/download.sh bm1684x

# minicpmv BM1688
./scripts/download.sh bm1688

# minicpmv cv186ah
./scripts/download.sh cv186ah
```

执行下载脚本后，当前目录下的文件如下：
```bash
├── models
│   └── BM1684X
│       └── minicpm-v-4-awq_w4bf16_seq2048_bm1684x_1dev_20250915_204204.bmodel
│   └── BM1688
│       └── minicpm-v-4-awq_w4bf16_seq2048_bm1688_2core_20251011_141218.bmodel
├── pics
│   └── demo.png
├── python
│   ├── config
│   │   └── minicpmv.yaml
│   ├── minicpmv.py
│   ├── README.md
│   ├── requirements.txt
│   └── token_config
│       ├── config.json
│       ├── configuration_minicpm.py
│       ├── generation_config.json
│       ├── image_processing_minicpmv.py
│       ├── modeling_minicpmv.py
│       ├── modeling_navit_siglip.py
│       ├── preprocessor_config.json
│       ├── processing_minicpmv.py
│       ├── resampler.py
│       ├── safetensors_tensor_info.csv
│       ├── special_tokens_map.json
│       ├── tokenization_minicpmv_fast.py
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── tokenizer.model
├── README.md
└── scripts
    └── download.sh

```

### 4.2 自行编译模型

#### 4.2.1 从Huggingface下载MiniCPMV

(比较大，会花费较长时间)

``` shell
# 下载模型
git lfs install
git clone https://www.modelscope.cn/OpenBMB/MiniCPM-V-4-AWQ.git
```

#### 4.2.2 下载TPU-MLIR docker，并配置TPU-MLIR编译环境

TPU-MLIR编译环境参考 [TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md/#1-tpu-mlir环境搭建)

#### 4.2.3 编译模型生成bmodel

``` shell
# 如果有提示transformers版本问题，pip3 install transformers --upgrade
llm_convert.py -m /workspace/MiniCPM-V-4-AWQ -s 2048 --quantize w4bf16 -c bm1684x --out_dir minicpmv --max_pixels 980,980
```
编译完成后，在指定目录`minicpmv`生成`minicpm-v-4-xxx.bmodel`和`config`

另外如果指定的seqlen比较长的话，比如8K，可以指定`--dynamic`编译，首token延时会根据实际长度变化，如下：
``` shell
# 如果有提示transformers版本问题，pip3 install transformers --upgrade
llm_convert.py -m /workspace/MiniCPM-V-4-AWQ -s 8192 --quantize w4bf16 -c bm1684x --dynamic --out_dir minicpmv --max_pixels 980,980
```

编译BM1688的模型，2核。
``` shell
llm_convert.py -m /workspace/MiniCPM-V-4-AWQ -s 2048 --quantize w4bf16 -c bm1688 --num_core 2 --out_dir minicpmv --max_pixels 980,980 
```


## 5. 例程测试
- [Python例程](./python/README.md)

## 6. 程序性能测试

这里的测试输入为："图片里面有什么东西？"。
请注意，BM1684X测试的是8b量级的模型。其中，minicpm-v-4-awq_w4bf16_seq2048_bm1684x_1dev_20250915_204204.bmodel是静态模型，seq输入长度是固定的2048，FTL是相对固定的数值。


|   测试平台   |     测试程序       |           测试模型                                                    |first token latency(s) |token per second(tokens/s)| 
| ----------- | ----------------  | --------------------------------------------------------------------- | --------------------- | ------------------------ | 
| SE7-32      | minicpmv.py           |  minicpm-v-4-awq_w4bf16_seq2048_bm1684x_1dev_20250915_204204.bmodel    |     10.272             |       15.499           | 
| SE9-16      | minicpmv.py           |  minicpmv-0.5b-gptq_w4bf16_seq512_bm1688_2core_20250616_122001.bmodel  |   31.078          |        7.747           |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，SE9-16的主控处理器为8核ARM A53 @1.6GHz，SE9-8的主控处理器为6核ARM A53 @1.6GHz；
> 3. 这里使用的BM1684XSDK版本是 V24.04.01，BM1688以及CV186AH的版本是 V1.8；
