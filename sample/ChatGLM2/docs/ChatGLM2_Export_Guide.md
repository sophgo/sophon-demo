# ChatGLM2模型导出与编译

## 1. 准备工作

ChatGLM2模型导出需要依赖[ChatGLM2官方仓库](https://huggingface.co/THUDM/chatglm2-6b)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。

**注意：** 

- 编译模型需要在x86主机完成。
- ChatGLM2-6B官方库25G左右，转模型需要保证运行内存至少32G以上，导出onnx模型需要存储空间50G以上，fp16模型转换需要存储空间180G以上，int8和int4模型需要的空间会更少。

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

    python3 -m dfss --url=open@sophgo.com:LLM/tpu-mlir_v1.1.0_RC1.114-g1ec6c16b-20231121.tar.gz
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

### 2.2.1 下载ChatGLM2官方代码

**注：** ChatGLM2-6B官方库25G左右

```bash
git lfs install
git clone git@hf.co:THUDM/chatglm2-6b
```

如果无法官网下载，也可以下载我们之前下好的，压缩包20G左右
```bash
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:LLM/chatglm2-6b.tgz
tar zxvf chatglm2-6b.tgz
```


### 2.1.2 对官方代码进行三处修改：

- 将config.json文件中seq_length配置为512；

- 将modeling_chatglm.py文件中的如下代码：

```bash
if attention_mask is not None:
    attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
```
修改为：

```bash
if attention_mask is not None:
    attention_scores = attention_scores + (attention_mask * -10000.0)
```
这样修改可以提升效率，使用masked_fill效率低下。

- 将modeling_chatglm.py文件中的如下代码：

```bash
pytorch_major_version = int(torch.__version__.split('.')[0])
if pytorch_major_version >= 2:
```
修改为

```bash
pytorch_major_version = int(torch.__version__.split('.')[0])
if False:
```
这样修改可以解决pytorch2.0导出有bug的问题。

### 2.1.3 导出onnx

- 指定chatglm2-6B官方仓库的python路径

```bash
# 将/workspace/chatglm2-6b换成docker环境中您的chatglm2-6b仓库的路径
export PYTHONPATH=/workspace/chatglm2-6b:$PYTHONPATH
```

- 导出所有onnx模型，如果过程中提示缺少某些组件，直接**pip install**组件即可

```bash
# 将/workspace/chatglm2-6b换成docker环境中您的chatglm2-6b仓库的路径
python3 tools/export_onnx.py --path /workspace/chatglm2-6b
```
此时有大量onnx模型被导出到本例程中ChatGLM2/models/onnx的目录。

### 2.2 bmodel编译

目前TPU-MLIR支持1684x对ChatGLM2进行F16, INT8和INT4量化。

- 生成FP16 bmodel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，如：

```bash
./scripts/gen_fp16bmodel_mlir.sh
```

​执行上述命令会在`models/BM1684X/`文件夹下生成`chatglm2-6b_f16.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 bmodel

​本例程在`scripts`目录下提供了TPU-MLIR编译INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，如：

```bash
./scripts/gen_int8bmodel_mlir.sh
```

​执行上述命令会在`models/BM1684X/`文件夹下生成`chatglm2-6b_int8.bmodel`文件，即转换好的INT8 BModel。

- 生成INT4 bmodel

​本例程在`scripts`目录下提供了TPU-MLIR编译INT4 BModel的脚本，请注意修改`gen_int4bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，如：

```bash
./scripts/gen_int4bmodel_mlir.sh
```
​执行上述命令会在`models/BM1684X/`文件夹下生成`chatglm2-6b_int4.bmodel`文件，即转换好的INT4 BModel。

### 2.3 准备tokenizer

将官方代码中chatglm2-6b/tokenizer.model放到BM1684X目录下。

至此导出onnx与转模型部分结束。
