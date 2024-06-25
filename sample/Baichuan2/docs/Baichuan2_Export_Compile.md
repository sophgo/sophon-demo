# Baichuan2模型导出与编译

## 1. 准备工作

Baichuan2模型导出需要依赖[Baichuan2官方仓库](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。

**注意：**

- 编译模型需要在x86主机完成。
- Baichuan2-7B官方库30G左右，转模型需要保证运行内存至少32G以上，导出onnx模型需要存储空间80G以上，fp16模型转换需要存储空间200G以上，int8和int4模型需要的空间会更少。

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

从sftp上获取TPU-MLIR压缩包（需要20240102的版本）
```bash
pip3 install dfss  --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m dfss --url=open@sophgo.com:sophon-demo/baichuan2/tpu-mlir_v1.6.100-g882a3523-20240102.tar.gz
tar zxvf tpu-mlir_v1.6.100-g882a3523-20240102.tar.gz
```

### 2.1.3. 创建并进入docker

TPU-MLIR使用的docker是sophgo/tpuc_dev:v3.2，docker镜像和tpu-mlir有绑定关系，少数情况下有可能更新了tpu-mlir，需要新的镜像。
```bash
docker pull sophgo/tpuc_dev:v3.2
# 这里将本级目录映射到docker内的/workspace目录，用户需要根据实际情况将sophon-demo的目录映射到docker里面
# myname只是举个名字的例子，请指定成自己想要的容器的名字
docker run --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:v3.2
# 此时已经进入docker，并在/workspace目录下
# 初始化软件环境
cd /workspace/tpu-mlir_v1.6.100-g882a3523-20240102
source ./envsetup.sh
cd ..
```
此镜像仅onnx模型导出和编译量化模型，程序编译和运行请在开发和运行环境中进行。更多TPU-MLIR的教程请参考[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)的《TPU-MLIR快速入门手册》和《TPU-MLIR开发参考手册》。

### 2.2 获取onnx

### 2.2.1 下载Baichuan2官方代码

**注：** Baichuan2-7B-Chat官方库16G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。
```bash
sudo apt-get install git-lfs    # 安装Git-LFS软件包，只需执行一次
git lfs install                 # 设置必要的Git钩子来初始化Git-LFS，若遇到Error，需要初始化到一个新的Git仓库或切换到已有的Git仓库目录中
git clone https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

如果无法从官网下载，也可以下载我们之前下好的，压缩包12G左右：
```bash
pip3 install dfss --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m dfss --url=open@sophgo.com:sophon-demo/baichuan2/Baichuan2-7B-Chat.tar.gz
tar zxvf Baichuan2-7B-Chat.tar.gz
```

下载完Baichuan2-7B-Chat官方库后，您还需要设置`BAICHUAN2_PATH`环境变量，模型导出时会使用到。
```bash
export BAICHUAN2_PATH=$PWD/Baichuan2-7B-Chat
```

### 2.2.2 修改官方代码：
进入本例程`sophon-demo/sample/Baichuan2`。
本例程的`./tools`目录下提供了修改好之后的`Baichuan2-7B-Chat/config.json`和`Baichuan2-7B-Chat/modeling_baichuan.py`，可以直接替换掉原仓库的文件：
```bash
cp tools/baichuan2-7b/config.json $BAICHUAN2_PATH
cp tools/baichuan2-7b/modeling_baichuan.py $BAICHUAN2_PATH
```


### 2.2.3 导出onnx

- 指定Baichuan2-7B-Chat官方仓库的python路径：

```bash
export PYTHONPATH=$BAICHUAN2_PATH:$PYTHONPATH
```

- 安装所需的第三方库，然后导出所有onnx模型：

```bash
pip install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 tools/export_onnx.py --model_path $BAICHUAN2_PATH
```
此时有大量onnx模型被导出到本例程中`./models/onnx`的目录。

### 2.3 bmodel编译

目前TPU-MLIR支持1684x对Baichuan2进行F16, INT8和INT4量化，使用如下命令生成bmodel。

```bash
./script/gen_bmodel.sh --mode int8
```

其中，mode可以指定fp16/int8/int4，编译成功之后，模型将会存放在`./models/BM1684X/`目录下。

### 2.4 准备tokenizer

如果您之前没有运行过下载脚本，那么您需要运行它以获取tokenizer。经过上面的步骤，现在你的目录下已经存在models文件夹，所以它只会下载tokenizer。
```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```