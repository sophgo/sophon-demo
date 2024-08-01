# ChatGLM4模型导出与编译

## 1. 准备工作

ChatGLM4模型导出需要依赖[ChatGLM4官方仓库](https://huggingface.co/THUDM/glm-4-9b-chat)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。

**注意：** 

- 编译模型需要在x86主机完成。
- 生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left。

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
    python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/tpu-mlir.tar.gz
    tar zxvf tpu-mlir.tar.gz
    cd tpu-mlir
    source ./envsetup.sh
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

### 2.2.1 下载ChatGLM4官方代码

**注：** ChatGLM4-9B官方库18G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。
```bash
git lfs install
git clone git@hf.co:THUDM/glm-4-9b-chat
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

如果无法从官网下载，也可以下载我们之前下好的，压缩包14G左右
```bash
pip3 install dfss  --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/glm-4-9b-chat-torch.zip
unzip glm-4-9b-chat-torch.zip
```

### 2.1.2 对齐环境和代码：
本例程的`tools`目录下提供了修改好之后的`config.json`和`modeling_chatglm.py`。可以直接替换掉原仓库的文件：
```bash
sudo apt-get update
sudo apt-get install pybind11-dev
pip install -r tools/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cp tools/glm-4-9b-chat/config.json glm-4-9b-chat/
cp tools/glm-4-9b-chat/modeling_chatglm.py glm-4-9b-chat/
```

### 2.1.3 导出onnx

- 指定glm-4-9b-chat官方仓库的python路径

```bash
# 将/workspace/glm-4-9b-chat换成docker环境中您的glm-4-9b-chat仓库的路径
export PYTHONPATH=/workspace/glm-4-9b-chat:$PYTHONPATH
```

- 导出所有onnx模型，如果过程中提示缺少某些组件，直接**pip install**组件即可

```bash
# 将/workspace/glm-4-9b-chat换成docker环境中您的glm-4-9b-chat仓库的路径
python3 tools/export_onnx.py --model_path /workspace/glm-4-9b-chat --seq_length 512
```
此时有大量onnx模型被导出到本例程中`ChatGLM4/models/onnx`的目录。

### 2.2 bmodel编译

目前TPU-MLIR支持1684x对ChatGLM4进行INT8和INT4量化，使用如下命令生成bmodel。

```bash
mv ./tmp ./scripts
./scripts/gen_bmodel.sh --mode int4 #int8
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