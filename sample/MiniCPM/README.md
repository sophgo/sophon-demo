# MiniCPM

## 目录

- [MiniCPM](#minicpm)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型和链接库](#4-准备模型和链接库)
    - [4.1 使用提供的模型](#41-使用提供的模型)
    - [4.2 自行编译模型](#42-自行编译模型)
    - [4.3 编译模型](#43-编译模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 程序性能测试](#6-程序性能测试)

## 1. 简介

MiniCPM 是面壁与清华大学自然语言处理实验室共同开源的系列端侧语言大模型，主体语言模型 MiniCPM-2B 仅有 24亿（2.4B）的非词嵌入参数量。关于该模型的其他特性，请前往源repo查看：https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16。本例程对MiniCPM-2B进行移植，使之能在SOPHON BM1684X/BM1688 上进行推理测试。


该例程支持在V23.09LTS SP2及以上的BM1684X SOPHONSDK, 或在v1.5.1及以上的BM1688 及CV186X SOPHONSDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在1684X SoC设备（如SE7、SM7、Airbox等）上运行，也支持在BM1688 Soc设备（如SE9-16）上运行， 同时还支持在 CV186X Soc设备（如SE9-8）上运行。

在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2.特性

* 支持BM1684X(x86 PCIe、SoC)，BM1688(SoC), CV186X(Soc)
* 支持INT8(BM1684X)、INT4(BM1684X/BM1688/CV186X)模型编译和推理
* 支持基于BMRT的C++例程
* 支持多轮对话

## 3. 运行环境准备
在PCIe上无需修改内存，以下为soc模式相关：

因为编译的最小的INT4模型加载后，最少要使用2253MB的TPU内存，所以您需要保证您的SE9的内存至少要大于2300MB为好，正常的SE9-16设备，在出厂的时候，其TPU内存为5632，您无需修改，如果您需要修改内存，请参考下面的教程操作：

对于1688系列设备（如SE9/SM9），都可以通过这种方式完成环境准备，使得满足MiniCPM-2B运行条件。首先，在1688 SoC环境上，参考如下命令修改设备内存。

```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/09/11/13/DeviceMemoryModificationKit.tgz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_{vx.x}.tar.xz #vx.x是版本号
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息
./memory_edit.sh -c -npu 1536 -vpu 0 -vpp 4096 #npu也可以访问vpu和vpp的内存
# 根据上一句命令输出最后一行的提示替换itb文件，对于1684X系列设备，执行
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
# 对于1688系列设备，执行
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/boot.itb /boot/boot.itb && sync
sudo reboot
```

> **注意：**
> 1. tpu总内存为npu/vpu/vpp三者之和，int4应满足tpu内存 >= 2300MB。
> 2. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)

## 4. 准备模型和链接库

该模型目前支持在CV186X、BM1688和BM1684X上运行，已提供编译好的bmodel。

### 4.1 使用提供的模型

​本例程在`scripts`目录下提供了下载脚本`download.sh`

**注意：**在运行前，应该保证存储空间大于20GB。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt-get update
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行程序后，当前目录下的文件如下：

```shell
.
├── cpp
│   ├── CMakeLists.txt
│   ├── demo.cpp                            #主程序
│   ├── include_bm1684x                     #1684x编译所需头文件
│   ├── include_bm1688                      #1688 编译所需头文件
│   ├── lib_pcie                            #1684X pcie编译所需链接库
│   ├── lib_soc_bm1684x                     #1684x编译所需链接库
│   ├── lib_soc_bm1688                      #1688 编译所链接库
│   ├── README.md                           #例程说明
│   ├── requirements.txt                    #需求库
│   └── token_config                        #tokenizer文件及模型
├── docs
│   └── FAQ.md                              #问题汇总
│   └── MiniCPM-2B_Export_Guide.md          #模型导出及编译指南
├── models
│   ├── BM1684X                                     #使用TPU-MLIR编译，用于BM1684X的模型
│   │   ├── minicpm-2b_bm1684x_int4.bmodel
│   ├── CV186X                                      #使用TPU-MLIR编译，用于CV186X的模型
│   │   ├── minicpm-2b_cv186x_int4_1core.bmodel
│   └── BM1688                                      #使用TPU-MLIR编译，用于BM1688的模型
│       ├── minicpm-2b_bm1688_int4_1core.bmodel
│       └── minicpm-2b_bm1688_int4_2core.bmodel
├── pics                                    #图片文件
│   ├── image.png
│   ├── Show_Results.png
│   └── sophgo_chip.png
├── README.md
├── scripts                                #下载及模型编译脚本等
│   ├── gen_bmodel.sh                      #编译bmodel的脚本
│   └── download.sh
└── tools
    ├── export_onnx.py                     #导出onnx模型脚本
    └── MiniCPM-2B
        └── modeling_minicpm.py            #模型文件
```

### 4.2 自行编译模型
此部分请参考[MiniCPM-2B模型导出与编译](./docs/MiniCPM-2B_Export_Guide.md)


## 5. 例程测试

C++例程的详细编译请参考[C++例程](./cpp/README.md)


## 6. 程序性能测试

这里的测试输入为："山东省最高的山是哪座山？"
根据测试，我们得到了如下表的模型性能表：
|    测试平台   |     测试程序       |           测试模型          |first token latency(s)|token per second(tokens/s)|
| -----------  | ---------------- | ---------------------------  | --------------------- | ----------------------- |
|   SE7-32     | demo.cpp  | minicpm-2b_bm1684x_int4       | 0.355 s |   26 token/s  |
|   SE9-16     | demo.cpp  | minicpm-2b_bm1688_int4_1core  | 2.039 s |   11 token/s  |
|   SE9-16     | demo.cpp  | minicpm-2b_bm1688_int4_2core  | 1.206 s |   13 token/s  |
|   SE9-8      | demo.cpp  | minicpm-2b_cv186x_int4_1core  | 2.018 s |   11 token/s  |


> **测试说明**：
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，SE9-16为8核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 3. 这里SE7-32使用的SDK版本是V23.09LTS SP2，SE9-16使用的SDK版本是v1.5.1；
