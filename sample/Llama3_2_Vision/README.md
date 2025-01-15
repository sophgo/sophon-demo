# Llama3_2_Vision

## 目录
- [Llama3_2_Vision](#llama3_2_vision)
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
Llama3_2_Vision是开源中英双语对话模型，关于它的特性，请前往源repo查看：https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct。 本例程对Llama3_2_Vision进行移植，使之能在SOPHON BM1684X上进行推理测试。

对于BM1684X，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在1684X SoC设备（如SE7、SM7、Airbox等）上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持INT8、INT4模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持多轮对话


## 3. 运行环境准备
在PCIe上无需修改内存，以下为soc模式相关：
对于1684X系列设备（如SE7/SM7）可以通过这种方式完成环境准备，使得满足Llama3_2_Vision运行条件。参考如下命令修改设备内存。
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
./memory_edit.sh -c -npu 7615 -vpu 3071 -vpp 3071 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
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
# llama3_2_vision 1684x
./scripts/download_bm1684x_bmodel.sh
```

执行下载脚本后，当前目录下的文件如下：
```bash
├── docs
│   └── Llama3_2_Vision_Export_Guide.md        #Llama3_2_Vision onnx导出和bmodel编译指南
├── models
│   └── BM1684X                     #download.sh下载的bmodel
│       └── llama3_2_vision-xxx.bmodel
├── pics
│   ├── test.jpg
├── python
│   ├── llama3.py                     #Llama3_2_Vision python推理脚本
│   ├── README.md                   #python例程执行指南
│   ├── requirements.txt            #python例程的依赖模块
│   ├── config                      #配置文件
│   │   └── llama3.yaml           #python demo的配置文件
│   └── token_config                #tokenizer 
├── README.md                       #Llama3_2_Vision例程指南
├── scripts
│   ├── download_bm1684x_bmodel.sh       #下载bmodel脚本
│   ├── download_onnx.sh                 #下载onnx脚本
│   └── gen_bmodel.sh                    #模型编译脚本             
└── tools
    ├── Llama-3.2-11B-Vision-Instruct                #修改过的Llama3_2_Vision源码
    │   ├── config.json
    │   └── modeling_mllama.py
    └── export_onnx.py              #Llama3_2_Vision导出onnx脚本。
```

### 4.2 自行编译模型

此部分请参考[Llama3_2_Vision模型导出与编译](./docs/Llama3_2_Vision_Export_Guide.md)

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

这里的测试输入为："what's in the room?"
|   测试平台   |     测试程序       |           测试模型                                  |first token latency(s) |token per second(tokens/s)| 
| ----------- | ----------------  | ------------------------------------------------- | --------------------- | ------------------------ | 
| SE7-32      | llama3.py           | llama3.2-11b-vision_int4_512seq.bmodel                   |    5.723              |    7.506                 | 
| SC7-HP75      | llama3.py           | llama3.2-11b-vision_int4_512seq.bmodel                   |    5.732              |    6.024                 | 

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 3. 这里使用的SDK版本是BM1684X V24.04.01；
