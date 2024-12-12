# Vila模型导出与编译

## 1. 准备工作
Vila1.5-3b模型导出需要依赖[Qwen官方仓库](https://huggingface.co/Qwen)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。

**注意：** 

- 编译模型需要在x86主机完成。

## 2. 主要步骤

模型编译前需要安装TPU-MLIR。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。
### 2.1 导出onnx模型

### 2.2.1 下载VILA1.5-3b原模型

```bash
git lfs install
git clone https://huggingface.co/Efficient-Large-Model/VILA1.5-3b
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

### 2.2.2 搭建运行环境
下载github源码，并执行环境搭建脚本
```bash
git clone https://github.com/NVlabs/VILA.git
cd VILA
./environment_setup.sh
```

### 2.2.3 修改transformers源码文件
```bash
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
mv tools/vila1.5-3b/modeling_llama.py $site_pkg_path/transformers/models/llama
mv tools/vila1.5-3b/cache_utils.py $site_pkg_path/transformers/
```

### 2.2.4 导出onnx
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/vila/llava.tar.gz
tar xvf llava.tar.gz && mv llava/ tools/ && rm llava.tar.gz
## 导出onnx
python3 tools/export_onnx.py --model_path your_model_path --seq_length your_seq_length
```
### 3.1 TPU-MLIR环境搭建

### 3.1.1 安装docker

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

### 3.1.2. 下载并解压TPU-MLIR

从sftp上获取TPU-MLIR压缩包
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/tpu-mlir_v1.10.beta.0-31-g896b42e8c-20240809.tar.gz
tar -xf tpu-mlir_v1.10.beta.0-31-g896b42e8c-20240809.tar.gz
```

### 3.1.3. 创建并进入docker

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



### 3.2 bmodel编译
首先需要在mlir工具下激活环境，[mlir下载地址可参考](./Qwen_Export_Guide.md/#212-下载并解压tpu-mlir)
```bash
cd tpu-mlir_v1.10.beta.0-31-g896b42e8c-20240809
source envsetup.sh
```
目前TPU-MLIR支持1684x对Vila进行FP16,INT8和INT4量化，使用如下命令生成bmodel。

```bash
./scripts/gen_bmodel.sh
```
编译成功之后，BM1684X模型将会存放在`models/BM1684X`目录下.

对于BM1688模型，请执行

```bash
./scripts/gen_bmodel_bm1688.sh
```
编译成功后，BM1688模型会存放在`models/BM1688`目录下
