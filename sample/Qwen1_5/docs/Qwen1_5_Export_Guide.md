# Qwen1.5模型导出与编译

## 1. 准备工作

Qwen1.5模型导出需要依赖[Qwen官方仓库](https://huggingface.co/Qwen)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。

**注意：** 

- 编译模型需要在x86主机完成。
- Qwen-7B官方库50G左右，转模型需要保证运行内存至少40G以上，导出onnx模型需要存储空间100G以上，请确有足够的内存完成对应的操作。

## 2. 主要步骤

模型编译前需要安装TPU-MLIR。安装好后需在TPU-MLIR环境中进入例程目录。先导出onnx，然后使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

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
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/tpu-mlir_v1.6.113-g7dc59c81-20240105.tar.gz 
    tar -xf tpu-mlir_v1.6.113-g7dc59c81-20240105.tar.gz 
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
此镜像仅onnx模型导出和编译量化模型，程序编译和运行请在开发和运行环境中进行。更多TPU-MLIR的教程请参考[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)的《TPU-MLIR快速入门手册》和《TPU-MLIR开发参考手册》。

### 2.2 获取onnx

### 2.2.1 下载Qwen官方代码

**注：** Qwen1.5-7B官方库50G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。（Qwen1.5-1.8B / Qwen1.5-14B的操作相同，请保证满足对应内存需求）,以下代码以Qwen1.5-1.8B为例

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen1.5-1.8B
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。


### 2.1.2 修改官方代码：
本例程的`tools`目录下提供了修改好之后的config.json和modeling_qwen2.py。(transformers请更新到4.38.2以上)可以直接替换掉原仓库的文件：

```bash
pip3 install transformers==4.38.2
cp tools/Qwen1.5-1.8B-Chat/config.json Qwen1.5-1.8B-Chat/
cp tools/Qwen1.5-1.8B-Chat/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```


### 2.1.3 导出onnx

- 导出所有onnx模型，如果过程中提示缺少某些组件，直接pip install组件即可
- (其中torch请勿使用2.1.1及以上版本，否则会出现SPDA相关报错，建议torch==2.0.1,transformers==4.38.2)

```bash
# 将/workspace/Qwen1.5-1.8B-Chat换成docker环境中您的Qwen1.5-1.8B-Chat仓库的路径
python3 tools/export_onnx.py --model_path /workspace/Qwen1.5-1.8B-Chat --seq_length your_sequence_length
```
此时有大量onnx模型被导出到本例程中`Qwen1_5/models/onnx`的目录。

### 2.2 bmodel编译
首先需要在mlir工具下激活环境
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/tpu-mlir_v1.6.113-g7dc59c81-20240105.tar.gz 
cd tpu-mlir_v1.6.113-g7dc59c81-20240105
source envsetup.sh
```
目前TPU-MLIR支持1684x对Qwen进行BF16(仅限Qwen-1.8B),INT8和INT4量化，使用如下命令生成bmodel。

```bash
./scripts/gen_bmodel.sh --mode int4 --name qwen1.5-1.8b
```

其中，mode可以指定int8/int4，编译成功之后，模型将会存放在`models/BM1684X/`目录下。

### 2.3 准备tokenizer

如果您之前没有运行过下载脚本，那么您需要运行它以获取tokenizer。经过上面的步骤，现在你的目录下已经存在models文件夹，所以它只会下载tokenizer。
```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```