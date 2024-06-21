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
    * [3.1 参数说明](#31-参数说明)
    * [3.2 测试音频](#32-测试音频)

cpp目录下提供了C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | wenet         | 使用Armadillo + bmcv前处理、BMRT推理  |

## 1. 环境准备
### 1.1 x86/arm PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。  

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

### 1.3 第三方库依赖
在x86/arm PCIe上需要依赖以下的第三方库(注：在X86平台编译soc程序也需要此步骤)：
```bash
sudo apt-get install libsuperlu-dev
```
在x86/arm PCIe上需要依赖以下的第三方库(注：在X86平台编译soc程序不需此步骤)：

```bash
# install armadillo
sudo apt-get install -y liblapack-dev libblas-dev libopenblas-dev libarmadillo-dev libsndfile1-dev
# install yaml-cpp
sudo apt-get install libyaml-cpp-dev libyaml-cpp0.6
cd cpp/
```
本例程的下载脚本中有提供编译好的ctcdecode-cpp，如果您需要重新编译ctcdecode-cpp，可以参考下面命令克隆ctcdecode-cpp项目。
```bash
rm -r ctcdecode-cpp
git clone https://github.com/Kevindurant111/ctcdecode-cpp.git
```
将ctcdecode-cpp项目克隆到本地后，请参考其提供的README进行编译，编译完成之后，回到`WeNet/`主目录下。 

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
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

```bash
cd cpp/
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
```
编译完成后，会在cpp目录下生成wenet.soc。


## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；

对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据，以及依赖库ctcdecode-cpp拷贝到SoC平台中测试，建议直接拷贝整个WeNet文件夹，此外，还需要设置一些环境变量：
```bash
# SoC设置环境变量，每次开一个新终端都需要重新设置。也可以将下面这些环境变量写到~/.bashrc里面并source ~/.bashrc，这样就不用每次开新终端都重新设置了。
# ${path/to/cpp}表示cross_compile_module所在目录，填绝对路径。
export LD_LIBRARY_PATH=${path/to/cpp}/cross_compile_module/ctcdecode-cpp/openfst-1.6.3/src/lib/.libs/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${path/to/cpp}/cross_compile_module/ctcdecode-cpp/build/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${path/to/cpp}/cross_compile_module/ctcdecode-cpp/build/3rd_party/kenlm/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${path/to/cpp}/cross_compile_module/3rd_party/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${path/to/cpp}/cross_compile_module/3rd_party/lib/blas/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${path/to/cpp}/cross_compile_module/3rd_party/lib/lapack/:$LD_LIBRARY_PATH
```
PCIe和SoC平台的测试参数及运行方式是一致的，下面主要以PCIe模式进行介绍：

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
Usage: wenet.pcie [params]
        --encoder_bmodel (value:../models/BM1684/wenet_encoder_streaming_fp32.bmodel)
                encoder bmodel file path
        --decoder_bmodel (value: )
                decoder bmodel file path
        --config_file (value:../config/train_u2++_conformer.yaml)
                config file path
        --result_file (value:./result.txt)
                result file path
        --input (value:../datasets/aishell_S0764/aishell_S0764.list)
                input path, audio data list
        --mode (value:ctc_prefix_beam_search)
                decoding mode, choose from 'ctc_greedy_search', 'ctc_prefix_beam_search' and 'attention_rescoring'
        --dev_id (value:0)
                TPU device id
        --help (value:true)
                print help information.
```
**注意：**  
- CPP传参与python不同，需要用等于号，例如`./wenet.pcie --encoder_bmodel=xxx`。  

### 3.2 测试音频
流式测试实例如下：
```bash
./wenet.pcie --encoder_bmodel=../models/BM1684/wenet_encoder_streaming_fp32.bmodel --dict_file=../config/lang_char.txt --config_file=../config/train_u2++_conformer.yaml --result_file=./result.txt --input=../datasets/aishell_S0764/aishell_S0764.list --mode=ctc_prefix_beam_search --dev_id=0
```
如果需要测试非流式，只需要设置`--encoder_bmodel`为非流式的encoder bmodel即可。

默认情况下decoder不开启，如果想要开启decoder重打分，请指定mode和decoder_bmodel参数如下：
```bash
./wenet.pcie --encoder_bmodel=../models/BM1684/wenet_encoder_streaming_fp32.bmodel --decoder_bmodel=../models/BM1684/wenet_decoder_fp32.bmodel --dict_file=../config/lang_char.txt --config_file=../config/train_u2++_conformer.yaml --result_file=./result.txt --input=../datasets/aishell_S0764/aishell_S0764.list --mode=attention_rescoring --dev_id=0
```
测试结束后，会将预测的结果文本保存在`result.txt`下，同时会打印预测结果、推理时间等信息。  
