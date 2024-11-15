# MP_SENet

## 目录
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型](#3-准备模型)
  - [4. 例程测试](#4-例程测试)
  - [5. 程序性能测试](#5-程序性能测试)

## 1. 简介
MP_SENet是一种基于深度学习的音频降噪方法，该方法不仅具有较小的模型参数量（2M）还具有当前（论文发布时间之前）最好的音频降噪性能。关于它的特性细节，请前往源repo查看：https://github.com/yxlu-0102/MP-SENet 。本例程对MP_SENet进行移植，使之能在SOPHON BM1684X上进行推理测试，实现优秀的音频降噪效果。

BM1684X系列：该例程支持在V24.04.01（release时间在2024.11.10后）及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在1684X SoC设备（如SE7）上运行。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持FP32 BF16模型编译和推理
* 支持基于SAIL推理的Python例程

## 3.运行环境搭建
### 3.1 BM1684X(x86 PCIe)
如果您在x86平台安装了PCIe加速卡（如SC系列加速卡），并使用它作为运行环境，您需要安装libsophon以及sophon-mw，除此之外，您还需要编译并安装sophon-sail（release时间在2024.11.10后），以上安装步骤具体请参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)。
这里特别提供了开发和运行环境搭建所需得SDK文件和sophon-sail源码（release时间在2024.11.10后）的下载链接：
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/MP_SENet/PCIE/sophon_SDK.tar.gz
python3 -m dfss --url=open@sophgo.com:sophon-demo/MP_SENet/PCIE/sophon_sail.tar.gz
```
### 3.2 BM1684X(SOC)
这里特别提供SE7即BM1684X(SOC)刷机包V24.04.01（release时间在2024.11.10后），刷机包地址如下：
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/MP_SENet/SOC/sdcard.tgz
```
刷机方式可以参考[刷机问题](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/faq/html/devices/SOC/soc_firmware_update.html?highlight=%E5%88%B7%E6%9C%BA)

此外，也提供刷机包对应SDK版本的sophon-sail（python）包，可以直接通过pip3安装，下载地址如下：
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/MP_SENet/SOC/sophon_arm-3.9.0-py3-none-any.whl
pip3 install sophon_arm-3.9.0-py3-none-any.whl
```
完成刷机和whl包安装即完成SOC的运行环境搭建。
## 4. 准备模型
该模型目前支持在bm1684X上运行，已提供编译好的bmodel。
### 4.1 使用提供的模型

本例程在`scripts`目录下提供了下载脚本`download.sh`

```bash
chmod -R +x scripts/
./scripts/download.sh
```

执行下载脚本后，当前目录下的文件如下：

```bash
.
├── datasets
│   └── test.wav              #音频测试数据（带噪）
├── docs
│   └── MP_SENet_Export_Guide.md         # MP_SENet ONNX导出和 bmodel 编译指南
├── models                               # download.sh下载的 bmodel
│   └── BM1684X
│       ├── mpsenet_vb_1b_fp32.bmodel
│       ├── mpsenet_vb_1b_bf16.bmodel
│       ├── mpsenet_dns_1b_fp32.bmodel
│       └── mpsenet_dns_1b_bf16.bmodel
├── python
│   ├── image.png                         #流程图
│   ├── README.md                         # python 例程执行指南
│   ├── requirements.txt                  # python 例程的依赖模块
│   └── mp_senet_sail.py                  # MP_SENet python 推理脚本
├── README.md                             # MP_SENet 例程指南
├── scripts
│   ├── auto_test.sh                      # 自动测试脚本
│   ├── download.sh                       # 下载脚本
│   ├── gen_fp32bmodel_mlir.sh            # FP32 bmodel编译脚本
│   └── gen_bf16bmodel_mlir.sh            # BF16 bmodel编译脚本
└── tools                
    ├── configs
    │   └── config.json                   
    ├── model_onnx.py                     # MP_SENet ONNX导出脚本
    └── requirements_model.txt            # ONNX导出的依赖模块
```

### 4.2 自行编译模型

此部分请参考[MP_SENet模型导出与编译](./docs/MP_SENet_Export_Guide.md)

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/mpsenet_vb_1b_bf16.bmodel
```
测试结果中的`calculate time`就是模型推理的时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                         | calculate time(ms) |
| -------------------------------------------       | ----------------- |
| BM1684X/mpsenet_vb_1b_bf16.bmodel|          1397.35  |
| BM1684X/mpsenet_vb_1b_fp32.bmodel|          2772.37  |

- 测试说明
1. 性能测试结果具有一定的波动性；
2. `calculate time`为固定输入大小（1，201，640）的推理时间；
3. SoC和PCIe的测试结果基本一致。

### 6.2 程序运行性能
例程测试后能在终端看到前处理时间、推理时间、后处理时间，以及从数据载入到生成降噪音频文件的总时间（测试音频长度为12秒）
|    测试平台  |     测试程序      |             测试模型               |preprocess_time(ms) |inference_time(ms)   |postprocess_time(ms) |
| ----------- | ---------------- | -----------------------------------| ---------------| --------------- | -------------- |
|   SE7-32    |mp_senet_sail.py|      mpsenet_vb_1b_bf16.bmodel      |     167.36      |     2883.84      |      51.60     |
|   SE7-32    |mp_senet_sail.py|      mpsenet_vb_1b_fp32.bmodel      |     210.82      |     8641.61      |      74.51     |

- 测试说明
1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
2. SE7-32(BM1684X) SDK版本:V24.04.01（release时间在2024.11.10后）；
3. SE7-32的主控处理器为8核CA53@2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；