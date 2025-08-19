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
Qwen / Qwen1.5/ Qwen2/ Qwen2.5/ Qwen3是开源中英双语对话模型，关于它的特性，请前往源repo查看：[Qwen](https://huggingface.co/Qwen)。 本例程对Qwen / Qwen1.5/ Qwen2/ Qwen2.5/ Qwen3进行移植，使之能在SOPHON BM1684X、BM1688/CV186X上进行推理测试。

本例程还支持DeepSeek-R1-Distill-Qwen-1.5B/7B/14B，关于它们的特性，请前往源repo查看：[DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)，[DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)，[DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)。本例程对这些模型进行移植，使之能在SOPHON BM1684X、BM1688/CV186X上进行推理测试。

本例程还支持QwQ-32B，关于它的特性，请前往源repo查看：[QwQ-32B](https://huggingface.co/Qwen/QwQ-32B)，本例程对这个模型进行移植，使之能在BM1684X(仅限SC7-224T加速卡)上进行推理测试。

对于BM1684X，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86/riscv主机上运行，也可以在1684X SoC设备（如SE7、SM7、Airbox等）上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

对于BM1688/CV186X，该例程支持在V1.7.0及以上的SDK上运行，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC、riscv PCIe)
* 支持BM1688/CV186X(SoC)
* QwQ-32B支持BM1684X(SC7-224T)
* 支持INT8、INT4模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持基于BMRT推理的CPP例程
* 支持多轮对话
* 支持动态模型推理

## 3. 运行环境准备
在PCIe上无需修改内存，以下为soc模式相关：
本例程对应的模型都较小，不用修改内存分布，如遇到设备内存不够的情况，请参考以下说明修改内存分布，并注意留出足够的CPU内存（大约1.7G）。
对于1684X系列设备（如SE7/SM7）和1688/cv186系列设备（SE9-16的8G/16G版本和SE9-8的8G版本）都可以通过这种方式完成环境准备，使得满足Qwen运行条件。参考如下命令修改设备内存。
```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://github.com/sophgo/sophon-tools/releases/download/v24.09.21/memory_edit_v2.10.tar.xz
tar xvf memory_edit_v2.10.tar.xz
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息

#如果是1684x系列设备，执行以下命令
./memory_edit.sh -c -npu 5120 -vpu 3072 -vpp 3072 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot

#如果是se9-16设备或se9-8 8G版本设备，执行以下命令
./memory_edit.sh -c -npu 5120 -vpu 0 -vpp 40 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/boot.itb /boot/boot.itb && sync
sudo reboot

#如果是se9-8 4G版本设备，执行以下命令
./memory_edit.sh -c -npu 2300 -vpu 0 -vpp 40 #npu也可以访问vpu和vpp的内存
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

# qwen2.5 1684x
./scripts/download.sh qwen2.5

# qwen3 1684x
./scripts/download.sh qwen3

# deepseek-r1-distill-qwen2 1684x
./scripts/download.sh deepseek-r1-distill-qwen2

# qwq-32b 1684x
./scripts/download.sh qwq-32b

# Include all bm1688 models
./scripts/download.sh bm1688

# Include all cv186x models
./scripts/download.sh cv186x

```

执行下载脚本后，当前目录下的文件如下：
```bash
├── docs
│   └── Qwen_Export_Guide.md        #Qwen onnx导出和bmodel编译指南
├── models
│   ├── BM1684X                     #download.sh下载的bmodel
│   │   ├── qwen-xxx.bmodel
│   │   ├── qwen1.5-xxx.bmodel
│   │   ├── qwen2-xxx.bmodel
│   │   ├── deepseek-r1-distill-qwen-1.5b
│   │   ├── deepseek-r1-distill-qwen-7b
│   │   ├── deepseek-r1-distill-qwen-14b
│   │   └── qwq-32b
│   ├── CV186X                    #download.sh下载的cv186x bmodel
│   │   └── qwen1.5-xxx.bmodel
│   └── BM1688                    #download.sh下载的bm1688 bmodel
│       ├── qwen1.5-xxx.bmodel
│       ├── qwen2.5-xxx.bmodel
│       ├── deepseek-r1-distill-qwen-1.5b
│       └── deepseek-r1-distill-qwen-7b
├── cpp
│   ├── README.md                 # CPP例程文档
│   └── qwen_bmlib                      
│       ├── CMakeLists.txt        # 编译配置文件
│       ├── main.cpp              # demo
│       ├── qwen.cpp              # qwen cpp源文件
│       ├── qwen.hpp              # qwen cpp头文件
│       └── utils.hpp             # 功能 cpp头文件
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
│   ├── gen_bmodel_qwen2_parallel.sh  #模型编译脚本                         
│   ├── gen_bmodel_deepseek_r1_distill_qwen_1_5b.sh  #模型编译脚本
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
    ├── Qwen2.5-xx-Instruct              #修改过的Qwen2.5源码
    ├── DeepSeek_R1_Distill_Qwen2.5-1.5B-Instruct    #修改过的DS Qwen2.5源码
    │   └── modeling_qwen2.py            
    └── export_onnx_qwen.py              #Qwen导出onnx脚本。
    └── export_onnx_qwen1_5.py           #Qwen1.5导出onnx脚本。
    └── export_onnx_qwen2.py             #Qwen2导出onnx脚本。
    └── export_onnx_qwen2_5.py           #Qwen2.5导出onnx脚本。
    └── export_onnx_qwen2_parallel.py    #Qwen2导出多芯onnx脚本。
    └── model_export_BM1684X_DS_qwen.py  #BM1684X deepseek-r1-distill-qwen2直接导出bmodel脚本
    └── export_onnx_deepseek_r1_sidtill_qwen2_BM1688.py           #BM1688 deepseek-r1-distill-qwen2导出onnx脚本(BM1688)。
```

### 4.2 自行编译模型

此部分请参考[Qwen模型导出与编译](./docs/Qwen_Export_Guide.md)

## 5. 例程测试

- [Python例程](./python/README.md)
- [CPP例程](./cpp/README.md)

## 6. 程序性能测试

这里的测试输入为："请使用C++写一段冒泡排序算法。"
|   测试平台   |     测试程序       |           测试模型                                 |first token latency(s) |token per second(tokens/s)| 
| ----------- | ----------------  | ---------------------------------------------------- | --------------------- | ------------------------ | 
| SE7-32      | qwen.py           | qwen-7b_int4_seq512_1dev.bmodel                      |    0.739              |    9.840                 | 
| SE7-32      | qwen.py           | qwen-7b_int4_seq2048_1dev.bmodel                     |    3.328              |    7.245                 | 
| SE7-32      | qwen.py           | qwen1.5-7b_int4_seq512_1dev.bmodel                   |    0.728              |    9.504                 | 
| SE7-32      | qwen.py           | qwen1.5-7b_int4_seq2048_1dev.bmodel                  |    3.234              |    7.083                 | 
| SE7-32      | qwen.py           | qwen2-7b_int4_seq512_1dev.bmodel                     |    0.728              |    9.504                 | 
| SE7-32      | qwen.py           | qwen2.5-7b_int4_seq512_1dev.bmodel                   |    0.652              |    10.26                 | 
| SE7-32      | qwen.py           | qwen2.5-7b_int4_seq2048_1dev.bmodel                  |    2.704              |    9.753                 | 
| SE7-32      | qwen.py           | qwen3-4b_int4_seq512_1dev.bmodel                     |    0.464              |    15.857                |
| SE7-32      | qwen.py           | deepseek-r1-distill-qwen2-1.5b_w4bf16_seq8192.bmodel |    4.983              |    25.689                | 
| SE7-32      | qwen.py           | deepseek-r1-distill-qwen2-7b_w4bf16_seq2048.bmodel   |    2.937              |    8.301                 | 
| SE7-32      | qwen.py           | deepseek-r1-distill-qwen2-14b_w4bf16_seq512.bmodel   |    1.297              |    5.652                 | 
| SE7-32      | qwen.py           | qwen2.5-1.5b_int4_seq512_1dev.bmodel                 |    0.185              |    41.078                | 
| SE7-32      | qwen.py           | qwen2.5-1.5b_int4_seq1024_1dev.bmodel                |    0.370              |    39.335                | 
| SE7-32      | main.cpp          | qwen2.5-1.5b_int4_seq512_1dev.bmodel                 |    0.200              |    28.188                | 
| SE7-32      | main.cpp          | qwen2.5-1.5b_int4_seq1024_1dev.bmodel                |    0.383              |    27.576                | 
| SC7-HP75    | qwen.py           | qwen1.5-7b_int4_seq4096_2dev_dyn.bmodel              |    >=1.56             |    9.748                 |
| SC7-224T    | qwen.py           | qwq-32b_int4_seq2048_2dev.bmodel                     |    8.398              |    3.852                 |
| SC7-224T    | qwen.py           | qwq-32b_int4_seq2048_4dev.bmodel                     |    5.663              |    5.961                 |
| SC7-224T    | qwen.py           | qwq-32b_int4_seq2048_8dev.bmodel                     |    4.530              |    5.929                 |
| SE9-16      | qwen.py           | qwen1.5-1.8b_int4_seq512_bm1688_1dev_2core.bmodel    |    0.559              |    21.171                |
| SE9-16      | qwen.py           | qwen2.5-1.5b_int4_seq1024_1688_2core.bmodel          |    1.283              |    20.171                | 
| SE9-16      | qwen.py           | qwen3-4b_w4bf16_seq512_bm1688_1core.bmodel            |   2.991              |    6.161                |
| SE9-16      | qwen.py           | qwen3-4b_w4bf16_seq512_bm1688_2core.bmodel            |   1.656               |   7.982                 |
| SE9-16      | qwen.py           | deepseek-r1-distill-qwen-1.5b_int4_seq1024_1688_2core.bmodel   |    1.418              |    19.261                | 
| SE9-16      | qwen.py           | deepseek-r1-distill-qwen-7b_int4_seq1024_1688_2core.bmodel   |    10.565              |    5.286                |
| SE9-8       | qwen.py           | qwen1.5-1.8b_int4_seq512_cv186x_1dev.bmodel          |    1.007              |    13.226                | 
| SRM1-20     | qwen.py           | qwen-7b_int4_seq512_1dev.bmodel                      |    0.915              |    5.850                 | 
| SRM1-20     | qwen.py           | qwen-7b_int4_seq2048_1dev.bmodel                     |    3.984              |    4.751                 | 
| SRM1-20     | qwen.py           | qwen1.5-7b_int4_seq512_1dev.bmodel                   |    0.901              |    5.805                 | 
| SRM1-20     | qwen.py           | qwen1.5-7b_int4_seq2048_1dev.bmodel                  |    3.884              |    4.739                 |
| SRM1-20     | qwen.py           | qwen2-7b_int4_seq512_1dev.bmodel                     |    0.981              |    6.234                 | 
| SRM1-20     | qwen.py           | qwen2.5-1.5b_int4_seq512_1dev.bmodel                 |    0.283              |    14.674                |
| SRM1-20     | qwen.py           | qwen2.5-1.5b_int4_seq1024_1dev.bmodel                |    0.503              |    13.970                | 
| SRM1-20     | qwen.py           | deepseek-r1-distill-qwen2-1.5b_w4bf16_seq8192.bmodel |    5.950              |    12.524                | 
| SRM1-20     | qwen.py           | deepseek-r1-distill-qwen2-7b_w4bf16_seq2048.bmodel   |    3.437              |    6.213                 |  
| SRM1-20     | qwen.py           | deepseek-r1-distill-qwen2-14b_w4bf16_seq512.bmodel   |    1.577              |    3.958                 |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 3. 这里使用的SDK版本是BM1684X V24.04.01, BM1688/CV186X V1.5.0；
