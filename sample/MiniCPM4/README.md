# MiniCPM4

## 目录
- [MiniCPM4](#minicpm4)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型](#4-准备模型)
    - [4.1 使用提供的模型](#41-使用提供的模型)
    - [4.2 自行编译模型](#42-自行编译模型)
      - [4.2.1 从Huggingface下载MiniCPM4](#421-从huggingface下载minicpm4)
      - [4.2.2 下载TPU-MLIR docker，并配置TPU-MLIR编译环境](#422-下载tpu-mlir-docker并配置tpu-mlir编译环境)
      - [4.2.3 编译模型生成bmodel](#423-编译模型生成bmodel)
  - [5. 例程测试](#5-例程测试)
  - [6. 程序性能测试](#6-程序性能测试)

## 1. 简介
MiniCPM4是开源中英双语对话模型，关于它的特性，请前往源repo查看：https://github.com/OpenBMB/MiniCPM。 本例程对MiniCPM4进行移植，使之能在SOPHON BM1684X、SOPHON BM1688以及SOPHON CV186AH上进行推理测试。

对于BM1684X，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在BM1684X SoC设备（如SE7、SM7、Airbox等）上运行。对于BM1688、CV186AH，支持在V1.8及以上版本的SoC设备（SE9、SM9）运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持BM1688&CV186AH(SoC)
* 支持W4A16模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持多轮对话


## 3. 运行环境准备
在PCIe上无需修改内存，以下为soc模式相关：
对于SoC产品，如SE7、SE9等，可以通过这种方式完成环境准备，使得满足MiniCPM4运行条件。参考如下命令修改设备内存。
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
# minicpm4 1684x
./scripts/download.sh bm1684x

# minicpm4 BM1688
./scripts/download.sh bm1688

# minicpm4 cv186ah
./scripts/download.sh cv186ah
```

执行下载脚本后，当前目录下的文件如下：
```bash
├── models
│   └── BM1684X
│       ├── minicpm4-8b_w4bf16_seq512_bm1684x_1dev_20250613_175044.bmodel
│       └── minicpm4-8b_w4bf16_seq8192_bm1684x_1dev_20250613_182940.bmodel
├── pics
│   └── demo.png
├── python
│   ├── config
│   │   └── minicpm4.yaml
│   ├── minicpm4.py
│   ├── README.md
│   ├── requirements.txt
│   └── token_config
│       ├── added_tokens.json
│       ├── config.json
│       ├── configuration.json
│       ├── generation_config.json
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── tokenizer.model
├── README.md
└── scripts
    └── download.sh

```

### 4.2 自行编译模型

#### 4.2.1 从Huggingface下载MiniCPM4

(比较大，会花费较长时间)

``` shell
# 下载模型
git lfs install
git clone git@hf.co:openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format
# 如果是8B，则如下：
git clone git@hf.co:openbmb/MiniCPM4-8B
```

#### 4.2.2 下载TPU-MLIR docker，并配置TPU-MLIR编译环境

TPU-MLIR编译环境参考 [TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md/#1-tpu-mlir环境搭建)

#### 4.2.3 编译模型生成bmodel

``` shell
# 如果有提示transformers版本问题，pip3 install transformers --upgrade
llm_convert.py -m /workspace/MiniCPM4-0.5B-QAT-Int4-GPTQ-format -s 512 --quantize w4bf16 -c bm1684x --out_dir minicpm4_0.5b
```
编译完成后，在指定目录`minicpm4_0.5b`生成`minicpm4-xxx.bmodel`和`config`

另外如果指定的seqlen比较长的话，比如8K，可以指定`--dynamic`编译，首token延时会根据实际长度变化，如下：
``` shell
# 如果有提示transformers版本问题，pip3 install transformers --upgrade
llm_convert.py -m /workspace/MiniCPM4-0.5B-QAT-Int4-GPTQ-format -s 8192 --quantize w4bf16 -c bm1684x --dynamic --out_dir minicpm4_0.5b
```

## 5. 例程测试
- [Python例程](./python/README.md)

## 6. 程序性能测试

这里的测试输入为："请使用C++写一段冒泡排序算法。"。

请注意，BM1684X测试的是8b量级的模型。其中，minicpm4-8b_w4bf16_seq512_bm1684x_1dev_20250613_175044.bmode是静态模型，seq输入长度是固定的512，FTL是相对固定的数值。inicpm4-8b_w4bf16_seq8192_bm1684x_1dev_20250613_182940.bmodel是动态模型，seq输入最长支持8192，FTL取决于实际的输入。

BM1688、CV186AH测试的是0.5b量级的模型，暂不支持动态模型，seq输入长度是固定的512。

|   测试平台   |     测试程序       |           测试模型                                                    |first token latency(s) |token per second(tokens/s)| 
| ----------- | ----------------  | --------------------------------------------------------------------- | --------------------- | ------------------------ | 
| SE7-32      | minicpm4.py           |  minicpm4-8b_w4bf16_seq512_bm1684x_1dev_20250613_175044.bmodel    |     0.787             |        8.538           | 
| SE7-32      | minicpm4.py           |  minicpm4-8b_w4bf16_seq8192_bm1684x_1dev_20250613_182940.bmodel    |    0.703             |        5.926           | 
| SE9-16      | minicpm4.py           |  minicpm4-0.5b-gptq_w4bf16_seq512_bm1688_2core_20250616_122001.bmodel  |   0.261          |        38.408           |
| SE9-8       | minicpm4.py           |  minicpm4-0.5b-gptq_w4bf16_seq512_cv186x_1core_20250616_122126.bmodel  |   0.397          |        34.280           |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，SE9-16的主控处理器为8核ARM A53 @1.6GHz，SE9-8的主控处理器为6核ARM A53 @1.6GHz；
> 3. 这里使用的BM1684XSDK版本是 V24.04.01，BM1688以及CV186AH的版本是 V1.8；
