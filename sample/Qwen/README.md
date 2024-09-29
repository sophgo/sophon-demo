# Qwen

## 目录
- [Qwen](#qwen)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型](#4-准备模型)
    - [4.1 使用提供的模型](#41-使用提供的模型)
    - [4.2 自行编译模型](#42-自行编译模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 程序性能测试](#6-程序性能测试)

## 1. 简介
Qwen / Qwen1.5/ Qwen2是开源中英双语对话模型，关于它的特性，请前往源repo查看：https://huggingface.co/Qwen。 本例程对Qwen / Qwen1.5/ Qwen2进行移植，使之能在SOPHON BM1684X、BM1688/CV186X（仅限Qwen1.5 1.8b）上进行推理测试。

对于BM1684X，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在1684X SoC设备（如SE7、SM7、Airbox等）上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

对于BM1688/CV186X，该例程支持在V1.7.0及以上的SDK上运行，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* Qwen1.5 1.8b支持BM1688/CV186X(SoC)
* 支持INT8、INT4模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持多轮对话


## 3. 运行环境准备
在PCIe上无需修改内存，以下为soc模式相关：
对于1684X系列设备（如SE7/SM7）和1688/cv186系列设备（SE9-16的8G/16G版本和SE9-8的8G版本）都可以通过这种方式完成环境准备，使得满足Qwen运行条件。参考如下命令修改设备内存。
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
./memory_edit.sh -c -npu 7615 -vpu 3072 -vpp 3072 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot

#如果是se9-16设备或se9-8 8G版本设备，执行以下命令
./memory_edit.sh -c -npu 6800 -vpu 0 -vpp 40 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/boot.itb /boot/boot.itb && sync
sudo reboot

#如果是se9-8 4G版本设备，执行以下命令
./memory_edit.sh -c -npu 2300 -vpu 0 -vpp 0 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/boot.itb /boot/boot.itb && sync
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
# qwen 1684x
./scripts/download.sh qwen

# qwen1.5 1684x
./scripts/download.sh qwen1.5

# qwen2 1684x
./scripts/download.sh qwen2

# bm1688
./scripts/download.sh bm1688

# cv186x
./scripts/download.sh cv186x

```

执行下载脚本后，当前目录下的文件如下：
```bash
├── docs
│   └── Qwen_Export_Guide.md        #Qwen onnx导出和bmodel编译指南
├── models
│   └── BM1684X                     #download.sh下载的bmodel
│       ├── qwen-xxx.bmodel
│       └── qwen1.5-xxx.bmodel
│       └── qwen2-xxx.bmodel
│   └── CV186X                    #download.sh下载的cv186x bmodel
│       └── qwen1.5-xxx.bmodel
│   └── BM1688                    #download.sh下载的bm1688 bmodel
│       └── qwen1.5-xxx.bmodel
├── python
│   ├── qwen.py                     #Qwen python推理脚本
│   ├── web_demo.py                 # web demo
│   ├── openai_api_server.py        # openai api 服务
│   ├── openai_api_request.py       # openai api 调用示例
│   ├── README.md                   #python例程执行指南
│   ├── requirements.txt            #python例程的依赖模块
│   └── config                      #配置文件
│       ├── qwen.yaml               #python demo的配置文件
│       ├── web.yaml                #web demo的配置文件
│       ├── api.yaml                #openai api server的配置文件
│   └── token_config                #tokenizer
│       ├── tokenization_qwen.py
│       ├── tokenizer_config.json
│       └── qwen.tiktoken 
├── README.md                       #Qwen例程指南
├── scripts                         
│   ├── download.sh                 #下载脚本
│   └── gen_bmodel.sh               #模型编译脚本
└── tools
    ├── Qwen-xx-Chat                #修改过的Qwen源码
    │   ├── config.json
    │   └── modeling_qwen.py
    ├── Qwen1.5-xx-Chat             #修改过的Qwen1.5源码
    │   ├── config.json
    │   └── modeling_qwen.py
    ├── Qwen2-xx-Instruct           #修改过的Qwen2源码
    │   ├── config.json
    │   └── modeling_qwen.py
    └── export_onnx_qwen.py              #Qwen导出onnx脚本。
    └── export_onnx_qwen1_5.py           #Qwen1.5导出onnx脚本。
    └── export_onnx_qwen2.py             #Qwen2导出onnx脚本。
    └── export_onnx_qwen2_parallel.py    #Qwen2导出多芯onnx脚本。
```

### 4.2 自行编译模型

此部分请参考[Qwen模型导出与编译](./docs/Qwen_Export_Guide.md)

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

这里的测试输入为："请使用C++写一段冒泡排序算法。"
|   测试平台   |     测试程序       |           测试模型                                  |first token latency(s) |token per second(tokens/s)| 
| ----------- | ----------------  | ------------------------------------------------- | --------------------- | ------------------------ | 
| SE7-32      | qwen.py           | qwen-7b_int4_seq512_1dev.bmodel                   |    0.739              |    9.840                 | 
| SE7-32      | qwen.py           | qwen-7b_int4_seq2048_1dev.bmodel                  |    3.328              |    7.245                 | 
| SE7-32      | qwen.py           | qwen1.5-7b_int4_seq512_1dev.bmodel                |    0.728              |    9.504                 | 
| SE7-32      | qwen.py           | qwen1.5-7b_int4_seq2048_1dev.bmodel               |    3.234              |    7.083                 | 
| SE7-32      | qwen.py           | qwen2-7b_int4_seq512_1dev.bmodel                  |    0.728              |    9.504                 | 
| SC7-HP75    | qwen.py           | qwen1.5-7b_int4_seq4096_2dev_dyn.bmodel           |    >=1.56             |    9.748                 |
| SE9-16      | qwen.py           | qwen1.5-1.8b_int4_seq512_bm1688_1dev.bmodel       |    1.094              |    12.995                | 
| SE9-16      | qwen.py           | qwen1.5-1.8b_int4_seq512_bm1688_1dev_2core.bmodel |    0.701              |    14.858                | 
| SE9-8       | qwen.py           | qwen1.5-1.8b_int4_seq512_cv186x_1dev.bmodel       |    1.007              |    13.226                | 


> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 3. 这里使用的SDK版本是BM1684X V24.04.01, BM1688/CV186X V1.5.0；
