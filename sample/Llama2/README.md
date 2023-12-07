# Llama2

## 目录
- [Llama2](#llama2)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 运行环境准备](#2-运行环境准备)
    - [2.1 方式一](#21-方式一)
  - [3. 准备模型](#3-准备模型)
    - [3.1 使用提供的模型](#31-使用提供的模型)
    - [3.2 开发环境准备](#32-开发环境准备)
    - [3.3 编译模型(分布式)](#33-编译模型分布式)
  - [4. C++例程](#4-C++例程)

## 1. 简介
Llama2-7B 是开源对话模型 Llama2-7B 的第二代版本,相比于初代模型，具有更强大的性能，更长的上下文，更高的推理性能和更开放的协议，Llama2-7B 权重对学术研究完全开放。

该例程支持在V23.07.01(libsophon_0.4.9)及以上的SDK上运行，提供了C++版本，支持在插有1684X系列加速卡的x86主机上运行，也可以SE7上运行。其中在SE7上运行需要额外进行环境配置，请参照[运行环境准备](#2-运行环境准备)完成环境部署。

## 2. 运行环境准备
以下为soc模式相关：
### 2.1 方式一
这里特别提供SE7刷机包，刷机包已经完成环境部署，并且内置llama2_soc版本的程序，刷机包地址如下：
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama2/sd_card_llama2-7b.zip
```
刷机方式可以参考[刷机问题](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/faq/html/devices/SOC/soc_firmware_update.html?highlight=%E5%88%B7%E6%9C%BA),在完成刷机后，代码程序在`/data`目录下。当然，还是建议您使用sophon-demo下的程序,它是最新版本的。 

## 3. 准备模型
该模型目前只支持在1684X上运行，已提供编译好的bmodel。
### 3.1 使用提供的模型

​本例程在`scripts`目录下提供了相关模型载脚本`download.sh`

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行程序后，当前目录下的文件如下：

```
.
├── README.md                            #使用说明
├── cpp                                  #Llama2-7B c++代码文件
│   ├── CMakeLists.txt
│   ├── demo.cpp                         #主程序
│   ├── include                          #编译所需的库文件
│   ├── lib_pcie                         #编译PCIE版本所需头文件
│   ├── lib_soc                          #编译SOC版本所需头文件
│   ├── llama2                           #可执行程序
│   ├── tokenizer.model                  #分词模型
│   └── README.md                        #例程使用说明
├── models
│   └── BM1684X                          #bmodel
│       ├─ llama2-7b_int4_1dev.bmodel    #int4 单芯模型
│       └─ llama2-7b_int8_1dev.bmodel    #int8 单芯模型
├── tools                                #自行编译模型时会需要的工具
│   ├── libsophon-distributed            #需要执行多芯运行(仅限多芯卡)所需的libsophon
│   ├── sentencepiece                    #分词工具
│   ├── soc-sdk                          #交叉编译所需工具(SDK=0.4.9)
└── script
    ├── download.sh                      #模型下载脚本
    └── compile                          #编译模型相关的脚本
        ├── compile.sh                   #编译bmodel脚本
        └── export_onnx.py               #导出onnx所需脚本
```
 
**注意：**在下载模型前，应该保证存储空间大于25GB。

### 3.2 开发环境准备
编译模型需要在x86主机完成。

**注意：** Llama2-7B官方库32G左右，转模型需要保证运行内存至少40G以上，导出onnx模型需要存储空间60G以上。

模型编译的详细信息可以参考[sophgo/Llama2-TPU](https://github.com/sophgo/Llama2-TPU)。
以下是基本步骤：

1. 下载docker，启动容器

```bash
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
当前$PWD应该是sophon-demo/sample/Llama2

后文(模型转换过程)假定环境都在docker的/workspace目录。


2. 下载Llama2-7B

虽然Llama2模型允许商业开源，但是模型下载需要想Meta提交使用申请，因此测试模型时可以使用我们已经下载好的模型
```bash
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama2/llama2-7b-torch.zip
unzip llama2-7b-torch.zip
```
将解压后的文件放至scripts/compile路径下

并对该工程做如下修改：

修改的目的是为了保证model\_tool --combine的时候block和block\_cache权重能对齐

```shell
pip show transformers
```

找到transformers库的位置(其中python3.10为本机所使用的python, 请根据实际所使用对路径进行修改)

```shell
vi /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
```

修改316行左右的代码，修改前为

```python
if past_key_value is not None:
  kv_seq_len += past_key_value[0].shape[-2]
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
```

修改后：

```python
if past_key_value is not None:
  kv_seq_len += past_key_value[0].shape[-2]
if past_key_value is not None:
  cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len-1)
else:
  cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
```

3. 下载`TPU-MLIR`代码并编译，(也可以直接下载编译好的release包解压)

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```

4. 下载[sentencepiece](https://github.com/google/sentencepiece)，并编译得到`sentencepiece.a`(sentencepiece已集成在tools目录下)

```shell
git clone git@github.com:google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j
```

如果要编译SoC环境，则需要在cpp的`CMakeLists.txt`加入如下代码：

```cmake
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
```
(如果需要重新编译sentencepiece,也需要在sentencepiece的`CMakeLists.txt`进行上述修改)
5. 下载libsophon库并安装

在算能官网<https://developer.sophgo.com/site/index/material/all/all.html>可以找到SDK最新版本，如下：

```shell
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/06/15/16/Release_230501-public.zip
```
解压sdk后安装libsophon，如下：

```shell
apt install sophon-libsophon-dev_0.4.8_amd64.deb
```

注意如果是SoC环境则安装arm64版本`sophon-libsophon-dev_0.4.8_arm64.deb`

### 3.3 编译模型(分布式)

分布式编译出来的模型在单芯和多芯上均可使用
(在编译前请先在`TPU-MLIR`中执行)

```shell
source ./envsetup.sh
./build.sh
```

1. 导出所有onnx模型，如果过程中提示缺少某些组件，直接pip install 组件即可

```bash
cd script/compile
python3 export_onnx.py
```
此时有大量onnx模型被导出到compile/tmp目录。

2. 对onnx模型进行编译，生成bmodel，这个过程会花一些时间，最终生成`llama2-7b.bmodel`文件　
```shell
./compile --num_device 1 --mode int8
```
其中num_device决定了后续所需要使用的推理芯片的数量(SOC请使用1), mode目前支持
"int4"(scripts/download.sh 中提供已经转好的bmodel),
"int8"(scripts/download.sh 中提供已经转好的bmodel),
"f16"(不提供已经转好的bmodel，编译模型和推理时num_device至少为2),
提供的模型文件均可以在执行scripts/download.sh 中下载

## 4. C++例程
C++例程请参考[C++例程](./cpp/README.md)