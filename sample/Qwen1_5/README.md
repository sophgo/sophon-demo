# Qwen1.5

## 目录
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型](#4-准备模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 程序性能测试](#6-程序性能测试)

## 1. 简介
Qwen1.5 是Qwen的第二代版本，它是开源中英双语对话模型，关于它的特性，请前往源repo查看：https://huggingface.co/Qwen。本例程对Qwen1.5进行移植，使之能在SOPHON BM1684X上进行推理测试。

该例程支持在V23.07.01(libsophon_0.4.9)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在1684X SoC设备（如SE7、SM7、Airbox等）上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持FP16、INT8、INT4模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持多轮对话


## 3. 运行环境准备
在PCIe上无需修改内存，以下为soc模式相关：
对于1684X系列设备（如SE7/SM7），都可以通过这种方式完成环境准备，使得满足Qwen1.5运行条件。首先，在1684x SoC环境上，参考如下命令修改设备内存。
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
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/boot.itb /boot/boot.itb && sync
sudo reboot
```
> **注意：**
> 1. tpu总内存为npu/vpu/vpp三者之和，fp16模型应满足tpu内存 >= 12800 MB，int8应满足tpu内存 >= 7168MB，int4应满足tpu内存 >= 4608MB。
> 2. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)

## 4. 准备模型
该模型目前只支持在1684X上运行，已提供编译好的bmodel。
### 4.1 使用提供的模型

​本例程在`scripts`目录下提供了下载脚本`download.sh`

**注意：**在运行前，应该保证存储空间大于10G (Qwen1.5-1.8B), 50GB(Qwen1.5-7B)

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行下载脚本后，当前目录下的文件如下：
```bash
├── docs
│   └── Qwen1.5_Export_Guide.md        #Qwen1.5 onnx导出和bmodel编译指南
├── models
│   └── BM1684X                     #download.sh下载的bmodel
│       ├── qwen1.5-1.8b_int4_1dev.bmodel
│       └── qwen1.5-1.8b_int8_1dev.bmodel
│       └── qwen1.5-4b_int4_1dev.bmodel
│       └── qwen1.5-7b_int4_1dev.bmodel
├── python
│   ├── web_demo.py                 #Qwen1.5 web-demo
│   ├── qwen1_5.py                  #Qwen1.5 python推理脚本
│   ├── README.md                   #python例程执行指南
│   ├── requirements.txt            #python例程的依赖模块
│   └── token_config                #download.sh下载的tokenizer
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── vocab.json
├── README.md                       #Qwen1.5例程指南
├── scripts                         
│   ├── download.sh                 #下载脚本
│   └── gen_bmodel.sh               #模型编译脚本
└── tools
    ├── Qwen1.5-0.5B-Chat            #修改过的Qwen-0.5B源码
    │   ├── config.json
    │   └── modeling_qwen.py
    ├── Qwen1.5-1.8B-Chat            #修改过的Qwen-1.8B源码
    │   ├── config.json
    │   └── modeling_qwen.py
        ├── Qwen1.5-4B-Chat        #修改过的Qwen-4B源码
    │   ├── config.json
    │   └── modeling_qwen.py
        ├── Qwen1.5-7B-Chat        #修改过的Qwen-7B源码
    │   ├── config.json
    │   └── modeling_qwen.py
    └── export_onnx.py               #Qwen导出onnx脚本。
```

### 4.2 自行编译模型

此部分请参考[Qwen1.5模型导出与编译](./docs/Qwen1_5_Export_Guide.md)

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

这里的测试输入为："请使用python写一段冒泡排序算法。"
|    测试平台   |     测试程序       |           测试模型               |first token latency(s)|token per second(tokens/s)| 
| -----------  | ---------------- | ---------------------------     | --------------------- | ----------------------- | 
| SE7-32  | qwen1_5.py      | qwen1.5-1.8b_int4_1dev.bmodel         |    0.191              |        31.863           | 
| SE7-32  | qwen1_5.py      | qwen1.5-1.8b_int8_1dev.bmodel         |    0.177              |        22.117           |
| SE7-32  | qwen1_5.py      | qwen1.5-4b_int4_1dev.bmodel           |    0.379              |        15.526           | 
| SE7-32  | qwen1_5.py      | qwen1.5-7b_int4_1dev.bmodel           |    0.728              |        10.151           | 
 
> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
