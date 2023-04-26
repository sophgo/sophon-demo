# C++例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
    * [1.3 第三方库依赖](#13-第三方库依赖)
* [2. 程序编译](#2-程序编译)
    * [2.1 x86/arm PCIe平台](#21-x86arm-pcie平台)
    * [2.2 SoC平台](#22-soc平台)
* [3. 推理测试](#3-推理测试)
    * [3.1 开启加速固件](#31-开启加速固件)
    * [3.2 参数说明](#32-参数说明)
    * [3.3 测试音频](#33-测试音频)
* [4. FAQ](#4-FAQ)

cpp目录下提供了C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | wenet         | 使用Armadillo + bmcv前处理、BMRT推理  |

## 1. 环境准备
### 1.1 x86/arm PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。  

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。

### 1.3 第三方库依赖
此外，在x86/arm PCIe/soc上都需要依赖以下的第三方库：
```bash
# install armadillo
sudo apt-get install -y liblapack-dev libblas-dev libopenblas-dev libarmadillo-dev libsndfile1-dev
# install yaml-cpp
sudo apt-get install libyaml-cpp-dev libyaml-cpp0.6
# install ctcdecode and follow the readme in ctcdecode-cpp to compile it
cd cpp/
git clone https://github.com/Kevindurant111/ctcdecode-cpp.git
```
将ctcdecode-cpp项目克隆到本地后，请参考其提供的README进行编译。  

## 2. 程序编译
C++程序运行前需要编译可执行文件。
### 2.1 x86/arm PCIe平台
ctcdecode-cpp下载并编译完成后，可以直接在PCIe平台上编译程序：

```bash
cd cpp/
mkdir build && cd build
cmake .. 
make
cd ..
```
编译完成后，会在cpp目录下生成wenet.pcie。

### 2.2 SoC平台
ctcdecode-cpp下载并编译完成后，可以直接在SoC平台上编译程序。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包：

```bash
cd cpp/
mkdir build && cd build
cmake -DTARGET_ARCH=soc ..  
make
```
编译完成后，会在cpp目录下生成wenet.soc。

## 3. 推理测试
PCIe平台和SoC平台上的测试参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 开启加速固件
在BM1684上运行测试时需要先开启加速固件, 否则模型可正常推理但是推理速度变慢; 在BM1684X上不需要这一步骤。
```bash
# step1 下载固件包
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/UVNz2Xi83
# step2 解压固件包
tar -xvf tpu-kernel-1684_v3.1.3-64586553-230125.tar.gz
# step3 进入文件夹
cd tpu-kernel-1684_v3.1.3-64586553-230125
# step4 加载基础固件
python3 ./scripts/load_firmware.py --firmware ./firmware/bm1684_ddr.bin_v3.1.3-64586553-230125 --firmware_tcm ./firmware/bm1684_tcm.bin_v3.1.3-64586553-230125
# step5 加载加速固件
test_update_fw ./firmware/bm1684_tcm_icache.bin_v3.1.3-64586553-230125 ./firmware/bm1684_ddr_icache.bin_v3.1.3-64586553-230125 0
# Note: 设备每次重启之后都需要重新执行step3和step5, 即加载加速固件, 基础固件无需每次重启加载
```

### 3.2 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：
```bash
Usage: wenet.pcie [params]
        --encoder_bmodel (value:../models/BM1684/wenet_encoder_fp32.bmodel)
                encoder bmodel file path
        --decoder_bmodel (value: )
                decoder bmodel file path
        --config_file (value:../config/train_u2++_conformer.yaml)
                config file path
        --result_file (value:./result.txt)
                result file path
        --input (value:../datasets/aishell_S0764/aishell_S0764.list)
                input path, images direction or video file path
        --mode (value:ctc_prefix_beam_search)
                decoding mode, choose from 'ctc_greedy_search', 'ctc_prefix_beam_search' and 'attention_rescoring'
        --dev_id (value:0)
                TPU device id
        --help (value:true)
                print help information.
```
**注意：**  
- CPP传参与python不同，需要用等于号，例如`./wenet.pcie --encoder_bmodel=xxx`。  

### 3.3 测试音频
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
./wenet.pcie --encoder_bmodel=../models/BM1684/wenet_encoder_fp32.bmodel --dict_file=../config/lang_char.txt --config_file=../config/train_u2++_conformer.yaml --result_file=./result.txt --input=../datasets/aishell_S0764/aishell_S0764.list --mode=ctc_prefix_beam_search --dev_id=0
```
默认情况下decoder不开启，如果想要开启decoder重打分，请指定mode和decoder_bmodel参数如下：
```bash
./wenet.pcie --encoder_bmodel=../models/BM1684/wenet_encoder_fp32.bmodel --decoder_bmodel=../models/BM1684/wenet_decoder_fp32.bmodel --dict_file=../config/lang_char.txt --config_file=../config/train_u2++_conformer.yaml --result_file=./result.txt --input=../datasets/aishell_S0764/aishell_S0764.list --mode=attention_rescoring --dev_id=0
```
测试结束后，会将预测的结果文本保存在`result.txt`下，同时会打印预测结果、推理时间等信息。  

## 4. FAQ
- 本例程暂时没有提供针对SoC平台的交叉编译方法。

