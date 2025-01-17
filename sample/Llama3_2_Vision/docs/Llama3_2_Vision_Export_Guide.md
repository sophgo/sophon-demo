# Llama3模型导出与编译

## 1. 准备工作

Llama3模型导出需要依赖[Llama3官方仓库](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。

**注意：** 

- 编译模型需要在x86主机完成。
- Llama-3.2-11B-Vision-Instruct官方库50G左右，转模型需要保证运行内存至少40G以上，导出onnx模型需要存储空间100G以上，请确保有足够的硬件空间完成对应的操作。


## 2. 主要步骤

模型编译前需要安装TPU-MLIR。安装好后需在TPU-MLIR环境中进入例程目录。先导出onnx，然后使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

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
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama3/tpu-mlir_latest.tar.gz
tar -xf tpu-mlir_latest.tar.gz
```

### 2.1.3. 创建并进入docker

TPU-MLIR使用的docker是sophgo/tpuc_dev:latest, docker镜像和tpu-mlir有绑定关系，少数情况下有可能更新了tpu-mlir，需要新的镜像。
```bash
docker pull sophgo/tpuc_dev:latest
# 这里将本级目录映射到docker内的/workspace目录,用户需要根据实际情况将demo的目录映射到docker里面
# myname只是举个名字的例子, 请指定成自己想要的容器的名字
docker run --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest
# 此时已经进入docker，并在/workspace目录下
# 初始化软件环境
cd /workspace/tpu-mlir_vx.y.z-<hash>-<date>
source ./envsetup.sh
```
此镜像仅onnx模型导出和编译量化模型，程序编译和运行请在开发和运行环境中进行。更多TPU-MLIR的教程请参考[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)的《TPU-MLIR快速入门手册》和《TPU-MLIR开发参考手册》。

### 2.2 获取onnx

### 2.2.1 下载Llama3官方代码

**注：** 
- Llama3-4B官方库50G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。


```bash
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

如果您的网络在下载的时候有问题，也可以使用下面的方法下载我们上传的源码模型包：
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:ext_model_information/LLM/torch/Llama-3.2-11B-Vision-Instruct.zip
```

### 2.1.2 修改官方代码：
本例程的`tools`目录下提供了修改好之后的`config.json`和`modeling_mllama.py`。可以直接替换掉原仓库的文件：

Llama3-4B
```bash
cp tools/Llama-3.2-11B-Vision-Instruct/config.json Llama-3.2-11B-Vision-Instruct/
cp tools/Llama-3.2-11B-Vision-Instruct/modeling_mllama.py Llama-3.2-11B-Vision-Instruct/
```

### 2.1.3 导出onnx

- 导出所有onnx模型，如果过程中提示缺少某些组件，直接**pip install**组件即可

Llama3
```bash
# bm1684x 单芯
cd tools
python export_onnx.py -m your_model_path -s 512 --device cpu
```
your_torch_model是你模型下载的位置，比如 Llama-3.2-11B-Vision-Instruct/

* 风险点：注意，如果使用--device cpu在cpu上导出，使用的精度是float32，与训练精度bfloat16不一致，可能导致精度问题
* 如果有cuda，建议使用cuda导出

此时有大量onnx模型被导出到本例程中`Llama3_2_Vision/tools/tmp/onnx`的目录。

### 2.2 bmodel编译
首先需要下载本模型编译所需的特定版本mlir，并在mlir工具下激活环境，可以参考[mlir下载地址可参考](./Llama3_2_Vision_Export_Guide.md#212-下载并解压tpu-mlir)

目前TPU-MLIR支持1684x对Llama3进行INT8和INT4量化，使用如下命令生成bmodel。

使用io_alone, int4精度
```bash
# bm1684x
cd scripts
./gen_bmodel.sh --mode int4 --name Llama3-4b --addr_mode io_alone --seq_length 512 --target bm1684x
```

其中，mode可以指定int4，编译成功之后，BM1684X模型将会存放在`models/BM1684X`目录下。
