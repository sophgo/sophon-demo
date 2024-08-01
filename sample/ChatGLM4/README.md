# ChatGLM4

## 目录
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型](#4-准备模型)
  - [5. 例程测试](#5-例程测试)
  - [3. 程序性能测试](#6-程序性能测试)

## 1. 简介
ChatGLM4-9B 是开源中英双语对话模型 ChatGLM 的第四代版本，关于它的特性，请前往源repo查看：https://huggingface.co/THUDM/glm-4-9b-chat。本例程对ChatGLM4-9B进行移植，使之能在SOPHON BM1684X上进行推理测试。

该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在1684X SoC设备（如SE7、SM7、Airbox等）上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持INT8、INT4模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持多轮对话


## 3. 运行环境准备
在PCIe上无需修改内存，以下为SoC模式相关：
对于1684X系列设备（如SE7/SM7），都可以通过这种方式完成环境准备，使得满足ChatGLM4运行条件。首先，在1684x SoC环境上，参考如下命令修改设备内存。
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
> **注意：**
> 1. tpu总内存为npu/vpu/vpp三者之和，int8模型应满足tpu内存 >= 7168MB，int4模型应满足tpu内存 >= 4608MB。
> 2. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)

## 4. 准备模型
该模型目前只支持在1684X上运行，已提供编译好的bmodel。
### 4.1 使用提供的模型

​本例程在`scripts`目录下提供了下载脚本`download.sh`

**注意：**在运行前，应该保证存储空间大于23GB。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行下载脚本后，当前目录下的文件如下：

```bash
├── docs
│   └── ChatGLM4_Export_Guide.md    #ChatGLM4 onnx导出和bmodel编译指南
├── models
│   └── BM1684X                     #download.sh下载的bmodel
│       ├── glm4-9b_int4_1dev.bmodel
│       └── glm4-9b_int8_1dev.bmodel
├── python
│   ├── ChatGLM4.py                 #ChatGLM4 python推理脚本
│   ├── README.md                   #python例程执行指南
│   ├── requirements.txt            #python例程的依赖模块
│   └── token_config                #download.sh下载的tokenizer
│       ├── tokenization_chatglm.py
│       ├── tokenizer_config.json
│       └── tokenizer.model
├── README.md                       #ChatGLM4例程指南
├── scripts                         
│   ├── download.sh                 #下载脚本
│   └── gen_bmodel.sh               #模型编译脚本
└── tools
    ├── glm-4-9b-chat               #修改过的ChatGLM4源码
    │   ├── config.json
    │   └── modeling_chatglm.py
    └── export_onnx.py              #ChatGLM4导出onnx脚本。
```


### 4.2 自行编译模型

此部分请参考[ChatGLM4模型导出与编译](./docs/ChatGLM4_Export_Guide.md)

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

这里的测试输入为："请使用C++写一段冒泡排序算法。"
|    测试平台   |     测试程序       |           测试模型             |first token latency(s)|token per second(tokens/s)| 
| -----------  | ---------------- | ---------------------------     | --------------------- | ----------------------- | 
| SE7-32       | ChatGLM4.py      | ChatGLM4-9b_int8.bmodel         |    1.181              |    4.471          | 
| SE7-32       | ChatGLM4.py      | ChatGLM4-9b_int4.bmodel         |    1.114              |    7.489          | 

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 3. 这里使用的SDK版本是V24.04.01；
