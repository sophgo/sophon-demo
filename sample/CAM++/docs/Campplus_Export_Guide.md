# CAM++模型导出与编译

- [CAM++模型导出与编译](#CAM++模型导出与编译)
  - [1. 自行编译模型](#1-自行编译模型)
  - [2. 主要步骤](#2-主要步骤)
    - [2.1 TPU-MLIR环境搭建](#21-tpu-mlir环境搭建)
      - [2.1.1 安装docker](#211-安装docker)
      - [2.1.2 下载并解压TPU-MLIR](#212-下载并解压tpu-mlir)
      - [2.1.3 创建并进入docker](#213-创建并进入docker)
    - [2.2 获取onnx](#22-获取onnx)
      - [2.2.1 下载3D-Speaker官方代码](#221-下载3D-Speaker官方代码)
      - [2.1.2 修改官方代码](#212-修改官方代码)
    - [2.1.3 导出onnx](#213-导出onnx)
    - [2.2 bmodel编译](#22-bmodel编译)


## 1. 自行编译模型

CAM++模型导出需要依赖[CAM++官方仓库](https://github.com/modelscope/3D-Speaker)。


**注意：**

- 编译模型需要在x86主机完成。

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
newgrp docker
```
> **提示**：需要logout系统然后重新登录，再使用docker就不需要sudo了。

### 2.1.2. 下载并解压TPU-MLIR

从sftp上获取TPU-MLIR压缩包
转BM1684X的模型，请下载tpu-mlir_bm1684x.tar.gz
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/campplus/mlir_1024/tpu-mlir_bm1684x.tar.gz
```
转BM1688/CV186X的模型，请下载tpu-mlir_bm1688.tar.gz
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/campplus/mlir_1024/tpu-mlir_bm1688.tar.gz
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

### 2.2.1 下载3D-Speaker官方代码

``` shell
git clone https://github.com/modelscope/3D-Speaker
```

### 2.1.2 修改官方代码：

为了让导出的onnx与pytorch计算结果对齐，需要做如下修改，diff如下

```diff
diff --git a/speakerlab/models/campplus/layers.py b/speakerlab/models/campplus/layers.py
index 9104f80..5445064 100644
--- a/speakerlab/models/campplus/layers.py
+++ b/speakerlab/models/campplus/layers.py
@@ -99,7 +99,9 @@ class CAMLayer(nn.Module):

     def seg_pooling(self, x, seg_len=100, stype='avg'):
         if stype == 'avg':
-            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
+            seg_ori = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
+            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True, count_include_pad=False)
+            assert((seg_ori == seg).all())
         elif stype == 'max':
             seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
         else:
```

## 2.1.3 导出onnx

- 导出所有onnx模型前，您还需要安装其他第三方库：

```bash
pip3 install -r tools/requirements.txt
```

- 如果过程中提示缺少某些组件，直接 **pip install** 组件即可

接下来，您可以使用下面的命令来导出CAM++的onnx模型：

```bash
python speakerlab/bin/export_speaker_embedding_onnx.py \
        --experiment_path . \
        --model_id iic/speech_campplus_sv_zh-cn_16k-common \
        --target_onnx_file campplus.onnx
```

此时`iic/speech_campplus_sv_zh-cn_16k-common`模型将导出为`campplus.onnx`。

### 2.2 bmodel编译

将`campplus.onnx`放置在`models/onnx`目录后，在以上提供的docker中执行
```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```
即可导出BM1684X的模型。
