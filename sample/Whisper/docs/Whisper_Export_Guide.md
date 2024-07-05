# Whisper模型的导出与编译
可以直接下载我们已经导出的onnx模型，推荐在mlir部分提供的docker中完成转bmodel模型。
**注意**：
- 编译模型需要在x86主机完成。

## 1 TPU-MLIR环境搭建
模型编译前需要安装TPU-MLIR。安装好后需在TPU-MLIR环境中进入例程目录。先导出onnx，然后使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

### 1.1 安装docker
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

### 1.2 下载并解压TPU-MLIR
从sftp上获取TPU-MLIR压缩包
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/mlir/tpu-mlir_v1.6.135-g12c3f90d8-20240327.tar.gz
```

### 1.3 创建并进入docker
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
此镜像仅onnx模型导出和编译量化模型，程序编译和运行请在开发和运行环境中进行。更多TPU-MLIR的教程请参考算能官网的《TPU-MLIR快速入门手册》和《TPU-MLIR开发参考手册》。


## 2 获取onnx
使用download.sh脚本下载的模型包括 whisper base/small/medium 的onnx模型：
```
./models
    └── onnx
      ├── decoder_loop_with_kvcache_base_5beam_448pad.onnx
      ├── decoder_loop_with_kvcache_medium_5beam_448pad.onnx
      ├── decoder_loop_with_kvcache_small_5beam_448pad.onnx
      ├── decoder_main_with_kvcache_base_5beam_448pad.onnx
      ├── decoder_main_with_kvcache_medium_5beam_448pad.onnx
      ├── decoder_main_with_kvcache_small_5beam_448pad.onnx
      ├── decoder_post_base_5beam_448pad.onnx
      ├── decoder_post_medium_5beam_448pad.onnx
      ├── decoder_post_small_5beam_448pad.onnx
      ├── encoder_base_5beam_448pad.onnx
      ├── encoder_medium_5beam_448pad.onnx
      ├── encoder_small_5beam_448pad.onnx
      ├── kvcache_rearrange_base_5beam_448pad.onnx
      ├── kvcache_rearrange_medium_5beam_448pad.onnx
      ├── kvcache_rearrange_small_5beam_448pad.onnx
      ├── logits_decoder_base_5beam_448pad.onnx
      ├── logits_decoder_medium_5beam_448pad.onnx
      └── logits_decoder_small_5beam_448pad.onnx
```

或者从源码导出
```bash
pip3 install -r ./python/requirements.txt
./scripts/gen_onnx.sh --model base
```

## 3 bmodel编译
目前TPU-MLIR支持1684x对Whisper进行F16量化，使用如下命令生成bmodel。
```bash
./scripts/gen_bmodel.sh --model base
```
其中，model可以指定base/small/medium，编译成功之后的模型放置于`./models/BM1684X/`，以base为例，最终会生成模型`bmwhisper_base_1684x_f16.bmodel`