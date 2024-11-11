# FaceFormer模型导出与编译

## 1. 准备工作

FaceFormer模型导出需要依赖[FaceFormer官方仓库](https://github.com/EvelynFan/FaceFormer)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。

**注意：** 

- 编译模型需要在x86主机完成。
- 生成bmodel耗时大概30分钟以上，建议32G内存以及10GB以上硬盘空间。

## 2. 主要步骤

模型编译前需要安装TPU-MLIR。安装好后需在TPU-MLIR环境中进入例程目录。先导出onnx，然后使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的 “3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。”

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

### 2.2.1 下载FaceFormer官方代码及权重

请保证已经提前执行了 `download.sh`模型下载脚本，经过本脚本会自动下载官方的代码原始权重以及相关的依赖，并自动解析到 `tools`的文件夹下。

### 2.2.2 导出onnx

如果您不想自己导出onnx，您也可以直接执行下面的命令，下载onnx的模型用于后续的模型编译：
```bash
pip3 install dfss  --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m dfss --url=open@sophgo.com:sophon-demo/FaceFormer/onnx.zip
unzip onnx.zip
```
此时，会解压出一个`onnx`的文件夹，请将该文件夹复制到`models`文件夹下即可。

如果您想自己尝试导出onnx的模型，可以参考以下步骤：

首先请根据`python/requirements.txt`的环境要求，安装对应的环境依赖包：
```bash
pip3 install -r python/requirements.txt
```
- 导出所有onnx模型，如果过程中提示缺少某些组件，直接**pip3 install**组件即可

```bash
cd tools
python3 export_onnx.py --model_name vocaset --wav_path "../Data/wav/test2.mp3" --dataset vocaset
```
此时有大量onnx模型被导出到本例程中`FaceFormer/models/onnx`的目录。

### 2.3 bmodel编译

目前TPU-MLIR支持1684X对FaceFormer进行编译，使用如下命令生成bmodel。
如果您没有下载testInput所需的模型测试输入，您也可以通过`tools/gen_npz.py`的脚本生成：
```bash
cd tools
python3 ./gen_npz.py
```
运行后，会自动在`models`下生成一个`testInput`的文件夹，里面会有多个模型输入测试的npz文件。

准备好所有的数据之后，您可以使用下面的命令生成bmodel：
```bash
./scripts/gen_bmodel_mlir.sh models
```

编译成功之后，`faceformer_f32.bmodel`模型将会存放在`models/BM1684X/`目录下。
