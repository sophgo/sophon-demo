# Vila

## 目录
- [Vila](#vila)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型](#3-准备模型)
    - [3.1 使用提供的模型](#31-使用提供的模型)
    - [3.2 自行编译模型](#32-自行编译模型)
  - [4. 例程测试](#4-例程测试)
  - [5. 程序性能测试](#5-程序性能测试)

## 1. 简介
VILA是一种视觉语言模型（Visual Language Model，简称VLM），它通过大规模交错的图像-文本数据进行预训练，从而能够实现视频理解和多图像理解的能力。它特别适合于视频内容的分析、多图像间关系的推理，以及图像和文本信息的融合处理。，关于它的特性，请前往源repo查看：https://huggingface.co/Efficient-Large-Model/VILA1.5-3b。 本例程对VILA1.5-3b进行移植，使之能在SOPHON BM1684X上进行推理测试。

对于BM1684X，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86/arm主机上运行，也可以在1684X SoC设备（如SE7、SM7、Airbox等）上运行。

对于BM1688，支持在1.7.0及以上的SDK运行。


## 2. 特性
* 支持BM1684X(x86/arm PCIe、SoC)， BM1688
* 支持INT8、INT4模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持基于SAIL推理的CPP例程


## 3. 准备模型
已提供编译好的bmodel。
### 3.1 使用提供的模型

​本例程在`scripts`目录下提供了下载脚本`download.sh`

```bash
chmod +x ./scripts/download.sh
./scripts/download.sh 
```

执行下载脚本后，当前目录下的文件如下：
```bash
├── docs
│   └── Qwen_Export_Guide.md             #Vila bmodel编译指南
├── datasets
│   └── test_car_person_1080P.mp4        #测试视频
├── models
│   ├── BM1684X                     #download.sh下载的BM1684X bmodel
│   |   ├── vision_embedding_6batch.bmodel   
│   |   ├── vision_embedding_1batch.bmodel   
│   |   └── llama_int4_seq2560.bmodel
│   └── BM1688                      #download.sh下载的BM1688 bmodel
├── python
│   ├── vila.py                     #Vila python推理脚本
│   ├── README.md                   #python例程执行指南
│   ├── requirements.txt            #python例程的依赖模块
│   └── config                      #配置文件
├── cpp
│   ├── README.md                   #cpp例程执行指南
│   ├── third_party                 #cpp例程第三方依赖库
│   └── vila_sail                   #sail cpp例程目录
│       ├── CMakeLists.txt          #sail cpp例程编译文件   
│       ├── main.cpp                #sail cpp例程主函数源码文件 
│       ├── vila.cpp                #sail cpp例程Vila源码文件 
│       └── vila.hpp                #sail cpp例程Vila头文件 
├── README.md                       #Vila例程指南
├── scripts                         
│   ├── download.sh                 #下载模型和数据集脚本
│   ├── gen_bmodel.sh               #编译bmodel脚本
│   ├── gen_bmodel_bm1688.sh        #编译1688 bmodel脚本
├── tools
|   ├── builder.py                  #导出onnx脚本的函数封装
|   ├── export_onnx.py              #导出onnx的脚本
|   ├── llava                       #导出onnx依赖的llava模块
```


### 3.2 自行编译模型
此部分请参考[Vila模型导出与编译](./docs/Vila_Export_Guide.md)

## 4. 例程测试

- [Python例程](./python/README.md)
- [CPP例程](./cpp/README.md)

## 5. 程序性能测试
图片embedding性能

|   测试平台   |    输入            |           精度           |       性能 (s)        | 
| ----------- | ----------------  | ------------------------| --------------------- | 
| SE7-32      | (1, 3, 384, 384)  | FP16                    |    0.08               |


LLM性能


|   测试平台   |     测试程序       |           测试模型              |first token latency(s) |token per second(tokens/s)| 
| ----------- | ----------------  | ------------------------------| --------------------- | ------------------------ | 
| SE7-32      | vila.py           | llama_int4_seq512.bmodel      |    0.377              |    24.2                | 
| SE7-32      | vila.py           | llama_int4_seq2560.bmodel     |    1.8                |    17.79               | 
| SE7-32      | vila.py           | llama_int4_seq4096.bmodel     |    3.4                |    14.98               | 
| SE7-32      | vila_sail.soc     | llama_int4_seq2560.bmodel     |    1.7                |    19.9               | 


> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 3. 这里使用的SDK版本是BM1684X V24.04.01；
