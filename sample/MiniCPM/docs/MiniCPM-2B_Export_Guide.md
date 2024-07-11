# MiniCPM-2B模型导出与编译

- [MiniCPM-2B模型导出与编译](#MiniCPM-2B模型导出与编译)
  - [1. 自行编译模型](#1-自行编译模型)
  - [2. 主要步骤](#2-主要步骤)
    - [2.1 TPU-MLIR环境搭建](#21-tpu-mlir环境搭建)
      - [2.1.1 安装docker](#211-安装docker)
      - [2.1.2 下载并解压TPU-MLIR](#212-下载并解压tpu-mlir)
      - [2.1.3 创建并进入docker](#213-创建并进入docker)
    - [2.2 获取onnx](#22-获取onnx)
      - [2.2.1 下载MiniCPM-2B官方代码](#221-下载MiniCPM-2B官方代码)
      - [2.1.2 修改官方代码](#212-修改官方代码)
    - [2.1.3 导出onnx](#213-导出onnx)
    - [2.2 bmodel编译](#22-bmodel编译)
  - [3. 准备tokenizer](#3-准备tokenizer)
  - [4. 编译sentencepiece)](#4-编译sentencepiece)


## 1. 自行编译模型

MiniCPM-2B模型导出需要依赖[MiniCPM-2B官方仓库](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。


**注意：** 

- 编译模型需要在x86主机完成。
- MiniCPM-2B官方库10G左右，转模型需要保证运行内存至少40G以上，导出onnx模型需要存储空间60G以上。INT8和INT4模型转换需要存储空间10GB左右。

## 2. 主要步骤
模型编译前需要安装TPU-MLIR。安装好后需在TPU-MLIR环境中进入例程目录。先导出onnx，然后使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)相应版本的SDK中获取)。

### 2.1 TPU-MLIR环境搭建

### 2.1.1 安装docker

    若已安装docker，请跳过本节。
```bash
# 安装docker
sudo apt-get install docker.io
# docker命令免root权限执行
# 创建docker用户组，若已有docker组会报错，没关系可忽略
sudo groupadd docker
# 将当前用户加入docker组
sudo usermod -aG docker $USER
# 切换当前会话到新group或重新登录重启X会话
newgrp docker​ 
```
> **提示**：需要logout系统然后重新登录，再使用docker就不需要sudo了。

### 2.1.2. 下载并解压TPU-MLIR

从sftp上获取TPU-MLIR压缩包

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/tpu-mlir_latest.tar.gz
```

### 2.1.3. 创建并进入docker

TPU-MLIR使用的docker是sophgo/tpuc_dev:latest, docker镜像和tpu-mlir有绑定关系，少数情况下有可能更新了tpu-mlir，需要新的镜像。
```bash
docker pull sophgo/tpuc_dev:latest
# 这里将本级目录映射到docker内的/workspace目录,用户需要根据实际情况将demo的目录映射到docker里面
# myname只是举个名字的例子, 请指定成自己想要的容器的名字
docker run --privileged --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest
# 此时已经进入docker，并在/workspace目录下
# 初始化软件环境
cd /workspace/tpu-mlir_vx.y.z-<hash>-<date>
source ./envsetup.sh
```
此镜像仅onnx模型导出和编译量化模型，程序编译和运行请在开发和运行环境中进行。更多TPU-MLIR的教程请参考[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)的《TPU-MLIR快速入门手册》和《TPU-MLIR开发参考手册》。

### 2.2 获取onnx

### 2.2.1 下载MiniCPM-2B官方代码

**注：** MiniCPM-2B官方库10GB左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。

您可以使用方法一，从Huggingface下载`MiniCPM-2B`，比较大，会花较长时间。同时，我们也为您提供了便捷的下载方式，您可以使用下面方法二来下载：

- 方法一：

``` shell
git lfs install
git clone git@hf.co:openbmb/MiniCPM-2B-sft-bf16
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

如果无法从官网下载，也可以下载我们之前下好的，压缩包10G左右

- 方法二：

``` shell
pip3 install dfss
sudo apt-get update
sudo apt-get install unzip
python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/MiniCPM-2B-sft-bf16.zip
unzip MiniCPM-2B-sft-bf16.zip
```

### 2.1.2 修改官方代码：

请注意，将解压后的 `MiniCPM-2B-sft-bf16`文件放至{MiniCPM}/tools路径下。

并对该工程做如下修改：

使用`tools/MiniCPM-2B`下的`modeling_minicpm.py`替换在 `MiniCPM-2B-sft-bf16` 目录下的原模型的对应文件`modeling_minicpm.py`

```bash
cp tools/MiniCPM-2B/modeling_minicpm.py MiniCPM-2B-sft-bf16/
```

## 2.1.3 导出onnx

- 导出所有onnx模型前，您还需要安装其他第三方库：

```bash
pip3 install -r tools/requirements.txt
```

- 如果过程中提示缺少某些组件，直接 **pip install** 组件即可

接下来，您可以使用下面的脚本来导出MiniCPM-2B的onnx模型：

``` shell
cd tools
python3 export_onnx.py --model_path ./MiniCPM-2B-sft-bf16/ 
```

此时有大量onnx模型被导出到../scripts/tmp目录。模型`seq_length`默认为512，如果想要支持更长序列，请在 `export_onnx.py`脚本运行时指定`--seq_length your_seq_length`

### 2.2 bmodel编译

目前TPU-MLIR支持1684x对 MiniCPM-2B 进行INT8(BM1684X)、INT4(BM1684X/BM1688/CV186X)量化，使用如下命令生成bmodel。编译过程会花一些时间，最终会生成`minicpm-2b_XXX.bmodel`文件

```bash
gen_bmodel.sh的参数解析：
    --name minicpm-2b  #模型名字
    --mode int4        #量化模型参数
    --target BM1688   #编译的模型芯片名，支持 BM1684X、BM1688和CV186X
    --num_core 1       #模型所需推理内核数，其中BM1684X不需要指定，默认为1
```


2.1 编译BM1684X的模型，进行INT8量化

```shell
./gen_bmodel.sh --name minicpm-2b --mode int8 --target BM1684X 
```

2.2 目前TPU-MLIR、BM1688支持对MiniCPM进行INT4量化，如果要生成单核模型，则执行以下命令，最终生成`minicpm-2b_bm1688_int4_1core.bmodel`文件

```shell
./gen_bmodel.sh --name minicpm-2b --mode int4 --target BM1688 --num_core 1 
```

2.3 如果要生成双核模型，则执行以下命令，最终生成`minicpm-2b_bm1688_int4_2core.bmodel`文件

```shell
./gen_bmodel.sh --name minicpm-2b --mode int4 --target BM1688 --num_core 2 
```

2.4 如果要生成CV186X的单核模型，则执行以下命令，最终生成`minicpm-2b_cv186x_int4_1core.bmodel`文件，请注意CV186X目前只支持单核模型的转出。

```shell
./gen_bmodel.sh --name minicpm-2b --mode int4 --target CV186X --num_core 1 
```

针对BM1688，其中num_core决定了后续所需要使用的推理芯片的内核数量, (scripts/download.sh 中提供已经转好的`1 core 和 2 core`bmodel),提供的模型文件均可以在执行scripts/download.sh 中下载

## 3 准备tokenizer

如果您之前没有运行过下载脚本，那么您需要运行它以获取tokenizer。经过上面的步骤，现在你的目录下已经存在models文件夹，所以它只会下载tokenizer。
```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

## 4. 编译sentencepiece

下载[sentencepiece](https://github.com/google/sentencepiece)，并编译得到`sentencepiece.a`

我们也在对应的编译文件夹下内置了相关的 `libsentencepiece.a` (已集成在 `cpp/lib_XXX`目录下), 您可以直接使用而无需额外的编译操作。

如果您想自己编译，您也可以参考下面的编译操作来编译适合您处理器架构的`libsentencepiece.a`。

```bash
git clone git@github.com:google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j
```

- 如果要编译SoC环境的sentencepiece链接文件，则需要在 `sentencepiece`的`CMakeLists.txt`加入如下代码来指定使用的编译器：

```cmake
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
```
