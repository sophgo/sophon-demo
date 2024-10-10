[简体中文](./Environment_Install_Guide.md) | [English](./Environment_Install_Guide_EN.md)

# sophon-demo环境安装指南
## 目录
- [sophon-demo环境安装指南](#sophon-demo环境安装指南)
  - [目录](#目录)
  - [1 TPU-MLIR环境搭建](#1-tpu-mlir环境搭建)
  - [2 TPU-NNTC环境搭建](#2-tpu-nntc环境搭建)
  - [3 x86 PCIe平台的开发和运行环境搭建](#3-x86-pcie平台的开发和运行环境搭建)
    - [3.1 安装libsophon](#31-安装libsophon)
    - [3.2 安装sophon-ffmpeg和sophon-opencv](#32-安装sophon-ffmpeg和sophon-opencv)
    - [3.3 编译安装sophon-sail](#33-编译安装sophon-sail)
  - [4 SoC平台的开发和运行环境搭建](#4-soc平台的开发和运行环境搭建)
    - [4.1 交叉编译环境搭建](#41-交叉编译环境搭建)
    - [4.2 交叉编译安装sophon-sail](#42-交叉编译安装sophon-sail)
  - [5 arm PCIe平台的开发和运行环境搭建](#5-arm-pcie平台的开发和运行环境搭建)
    - [5.1 安装libsophon](#51-安装libsophon)
    - [5.2 安装sophon-ffmpeg和sophon-opencv](#52-安装sophon-ffmpeg和sophon-opencv)
    - [5.3 编译安装sophon-sail](#53-编译安装sophon-sail)
  - [6 riscv PCIe平台的开发和运行环境搭建](#6-riscv-pcie平台的开发和运行环境搭建)
    - [6.1 安装libsophon](#61-安装libsophon)
    - [6.2 安装sophon-ffmpeg和sophon-opencv](#62-安装sophon-ffmpeg和sophon-opencv)
    - [6.3 编译安装sophon-sail](#63-编译安装sophon-sail)

Sophon Demo所依赖的环境主要包括用于编译和量化模型的TPU-NNTC、TPU-MLIR环境，用于编译C++程序的开发环境以及用于部署程序的运行环境。

## 1 TPU-MLIR环境搭建
使用TPU-MLIR编译BModel，通常需要在x86主机上安装TPU-MLIR环境，x86主机已安装Ubuntu16.04/18.04/20.04系统，并且运行内存在12GB以上。TPU-MLIR环境安装步骤主要包括：

1. 安装Docker
   若已安装docker，请跳过本节。
    ```bash
    # 如果您的docker环境损坏，可以先卸载docker
    sudo apt-get remove docker docker.io containerd runc

    # 安装依赖
    sudo apt-get update
    sudo apt-get install \
            ca-certificates \
            curl \
            gnupg \
            lsb-release

    # 获取密钥
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL \
        https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o docker.gpg && \
        sudo mv -f docker.gpg /etc/apt/keyrings/

    # 添加 docker 软件包
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # 安装 docker
    sudo apt-get update
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # docker命令免root权限执行
    # 创建docker用户组，若已有docker组会报错，没关系可忽略
    sudo groupadd docker
    # 将当前用户加入docker组
    sudo usermod -aG docker $USER
    # 切换当前会话到新group或重新登录重启X会话
    newgrp docker​ 
    ```
    > **提示**：需要logout系统然后重新登录，再使用docker就不需要sudo了。

2. 创建并进入docker

    TPU-MLIR使用的docker是sophgo/tpuc_dev:latest, docker镜像和tpu-mlir有绑定关系，少数情况下有可能更新了tpu-mlir，需要新的镜像。
    ```bash
    docker pull sophgo/tpuc_dev:latest
    # 这里将本级目录映射到docker内的/workspace目录,用户需要根据实际情况将demo的目录映射到docker里面
    # myname只是举个名字的例子, 请指定成自己想要的容器的名字
    docker run --privileged --name myname --network host -v $PWD:/workspace -it sophgo/tpuc_dev:latest
    # 此时已经进入docker，并在/workspace目录下  
    ```

3. 安装TPU-MLIR
    
    目前支持三种安装方法：

    (1)直接从pypi下载并安装：
    ```bash
    pip install tpu_mlir -i https://pypi.tuna.tsinghua.edu.cn/simple 
    ```
    (2)从[TPU-MLIR Github](https://github.com/sophgo/tpu-mlir/releases)下载最新`tpu_mlir-*-py3-none-any.whl`，然后使用pip安装：
    ```bash
    pip install tpu_mlir-*-py3-none-any.whl
    ```

    TPU-MLIR在对不同框架模型处理时所需的依赖不同，对于onnx或torch生成的模型文件，
    使用下面命令安装额外的依赖环境:
    ```bash
    pip install tpu_mlir[onnx]
    pip install tpu_mlir[torch]
    ```
    目前支持五种配置: onnx, torch, tensorflow, caffe, paddle。可使用一条命令安装多个配置，也可直接安装全部依赖环境:
    ```bash
    pip install tpu_mlir[onnx,torch,caffe]
    pip install tpu_mlir[all]
    ```
    (3)如果您获取了类似`tpu-mlir_${version}-${hash}-${date}.tar.gz`这种形式的发布包，可以通过这种方式配置：
    ```bash
    # 如果此前有通过pip安装过mlir，需要卸载掉
    pip uninstall tpu_mlir
    
    tar xvf tpu-mlir_${version}-${hash}-${date}.tar.gz
    cd tpu-mlir_${version}-${hash}-${date}
    source envsetup.sh #配置环境变量
    ```

建议TPU-MLIR的镜像仅用于编译和量化模型，程序编译和运行请在开发和运行环境中进行。更多TPU-MLIR的教程请参考[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)的《TPU-MLIR快速入门手册》和《TPU-MLIR开发参考手册》。

## 2 TPU-NNTC环境搭建

**注意，TPU-NNTC已停止维护，如果使用TPU-NNTC出现问题，建议您使用TPU-MLIR。**

使用TPU-NNTC编译BModel，通常需要在x86主机上安装TPU-NNTC环境，x86主机已安装Ubuntu16.04/18.04/20.04系统，并且运行内存在12GB以上。TPU-NNTC环境安装步骤主要包括：

1. 安装Docker

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

2. 下载并解压TPU-NNTC

    从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的TPU-NNTC压缩包，命名如tpu-nntc_vx.y.z-hash-date.tar.gz，x.y.z表示版本号，并进行解压。
    ```bash
    mkdir tpu-nntc
    # 将压缩包解压到tpu-nntc
    tar zxvf tpu-nntc_vx.y.z-<hash>-<date>.tar.gz --strip-components=1 -C tpu-nntc
    ```

3. 创建并进入docker

    TPU-NNTC使用的docker是sophgo/tpuc_dev:2.1, docker镜像和tpu-nntc有绑定关系，少数情况下有可能更新了tpu-nntc，需要新的镜像。
    ```bash
    cd tpu-nntc
    # 进入docker，如果当前系统没有对应镜像，会自动从docker hub上下载
    # 这里将tpu-nntc的上一级目录映射到docker内的/workspace目录,用户需要根据实际情况将demo的目录映射到docker里面
    # 这里用了8001到8001端口映射，之后在使用ufw可视化工具会用到
    # 如果端口已经占用，请更换其他未占用端口，后面根据需要更换进行调整
    docker run --privileged --name myname -v $PWD/..:/workspace -p 8001:8001 -it sophgo/tpuc_dev:v2.1
    # 此时已经进入docker，并在/workspace目录下
    # 下面初始化软件环境
    cd /workspace/tpu-nntc
    source scripts/envsetup.sh
    ```
此镜像仅用于编译和量化模型，程序编译和运行请在开发和运行环境中进行。更多TPU-NNTC的教程请参考[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)的《TPU-NNTC快速入门指南》和《TPU-NNTC开发参考手册》。



## 3 x86 PCIe平台的开发和运行环境搭建
如果您在x86平台安装了PCIe加速卡，开发环境与运行环境可以是统一的，您可以直接在宿主机上搭建开发和运行环境。

**注意：** mlir提供的docker环境用来编译模型的，不建议与运行环境混用，如果您需要在主机上搭建docker测试环境，请参考官网《LIBSOPHON使用手册.pdf》第6章-使用Docker搭建测试环境。

### 3.1 安装libsophon
从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的SOPHONSDK，解压后在libsophon_{date}_{time}文件夹下面有这几个文件:
* sophon-driver_x.y.z_amd64.deb
* sophon-libsophon_x.y.z_amd64.deb
* sophon-libsophon-dev_x.y.z_amd64.deb

其中：x.y.z表示版本号；sophon-driver包含了PCIe加速卡驱动；sophon-libsophon包含了运行时环境（库文件、工具等）；sophon-libsophon-dev包含了开发环境（头文件等）。如果只是在部署环境上安装，则不需要安装 sophon-libsophon-dev。
```bash
# 安装依赖库，只需要执行一次
sudo apt install dkms libncurses5
# 安装libsophon
sudo dpkg -i sophon-*amd64.deb
# 在终端执行如下命令，或者登出再登入当前用户后即可使用bm-smi等命令：
source /etc/profile
```

更多libsophon信息请参考《LIBSOPHON使用手册.pdf》。

### 3.2 安装sophon-ffmpeg和sophon-opencv
从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的SOPHONSDK，解压后在sophon-mw_{date}_{time}文件夹下面有这几个文件:
* sophon-mw-sophon-ffmpeg_x.y.z_amd64.deb
* sophon-mw-sophon-ffmpeg-dev_x.y.z_amd64.deb
* sophon-mw-sophon-opencv_x.y.z_amd64.deb
* sophon-mw-sophon-opencv-dev_x.y.z_amd64.deb

其中：x.y.z表示版本号；sophon-ffmpeg/sophon-opencv包含了ffmpeg/opencv运行时环境（库文件、工具等）；sophon-ffmpeg-dev/sophon-opencv-dev包含了开发环境（头文件、pkgconfig、cmake等）。如果只是在部署环境上安装，则不需要安装 sophon-ffmpeg-dev/sophon-opencv-dev。

sophon-mw-sophon-ffmpeg依赖sophon-libsophon包，而sophon-mw-sophon-opencv依赖sophon-mw-sophon-ffmpeg，因此在安装次序上必须
先安装libsophon, 然后sophon-mw-sophon-ffmpeg, 最后安装sophon-mw-sophon-opencv。

如果运行环境中使用的libstdc++库使用GCC5.1之前的旧版本ABI接口（典型的有CENTOS系统），请使用sophon-mw-sophon-opencv-abi0相关安装包。

```bash
# 安装sophon-ffmpeg
sudo dpkg -i sophon-mw-sophon-ffmpeg_*amd64.deb sophon-mw-sophon-ffmpeg-dev_*amd64.deb
# 安装sophon-opencv
sudo dpkg -i sophon-mw-sophon-opencv_*amd64.deb sophon-mw-sophon-opencv-dev_*amd64.deb
# 在终端执行如下命令，或者logout再login当前用户后即可使用安装的工具
source /etc/profile
```

更多sophon-mw信息请参考《MULTIMEDIA使用手册.pdf》、《MULTIMEDIA开发参考手册.pdf》。

### 3.3 编译安装sophon-sail
如果例程依赖sophon-sail则需要编译和安装sophon-sail，否则可跳过本章节。

需从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的SOPHONSDK，解压后在sophon-sail_{date}_{time}文件夹里面有sophon-sail的压缩包，命名如sophon-sail_x.y.z.tar.gz，x.y.z表示版本号。
您可以打开sophon-sail压缩包里面提供的用户手册(命名为sophon-sail_zh.pdf)，参考编译安装指南章节，选择您需要的模式(C++/Python，PCIE MODE)进行安装。


## 4 SoC平台的开发和运行环境搭建
对于SoC平台，安装好SophonSDK(>=v22.09.02)后内部已经集成了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，位于`/opt/sophon/`下，可直接用于运行环境。通常在x86主机上交叉编译程序，使之能够在SoC平台运行。SophonSDK固件刷新方法可参考[FAQ文档](./FAQ.md#12-soc模式下如何使用sd卡刷更新固件).

### 4.1 交叉编译环境搭建
需要在x86主机上使用SOPHONSDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中。
1. 搭建交叉编译环境，这里提供两种方式：
    
    (1)通过apt安装交叉编译工具链：

    如果您的系统是ubuntu20.04，GLIBC版本<=2.31，可以使用如下命令安装：
    ```bash
    sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
    ```
    卸载方法：
    ```bash
    sudo apt remove cpp-*-aarch64-linux-gnu
    ```

    如果您的环境不满足上述要求，建议使用第(2)种方法。

    (2)通过docker搭建交叉编译环境：
    
    **请注意，不要将下文的stream_dev镜像和用于模型编译的tpuc_dev镜像混用。**

    这里提供一个ubuntu20.04的docker镜像作为交叉编译环境，通过dfss下载：
    ```bash
    pip3 install dfss
    python3 -m dfss --url=open@sophgo.com:/sophon-stream/docker/stream_dev.tar
    ```

    如果是首次使用Docker, 可执行下述命令进行安装和配置(仅首次执行):
    ```bash
    sudo apt install docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker
    ```

    在下载好的镜像目录中加载镜像
    ```bash
    docker load -i stream_dev.tar
    ```
    可以通过`docker images`查看加载好的镜像，默认为stream_dev:latest

    创建容器
    ```bash
    docker run --privileged --name stream_dev -v $PWD:/workspace  -it stream_dev:latest
    # stream_dev只是举个名字的例子, 请指定成自己想要的容器的名字
    ```
    容器中的`workspace`目录会挂载到您运行`docker run`时所在的宿主机目录，您可以在此容器中编译项目

    > 注：该镜像来自[sophon-stream](https://github.com/sophgo/sophon-stream/blob/master/docs/HowToMake.md#使用开发镜像编译)。

2. 打包libsophon

    从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的SOPHONSDK，在sophon-img_{date}_{time}文件夹中包括libsophon_soc_x.y.z_aarch64.tar.gz，x.y.z表示版本号，并进行解压。

    ```bash
    # 创建依赖文件的根目录
    mkdir -p soc-sdk
    # 解压libsophon_soc_x.y.z_aarch64.tar.gz
    tar -zxf libsophon_soc_${x.y.z}_aarch64.tar.gz
    # 将相关的库目录和头文件目录拷贝到依赖文件根目录下
    cp -rf libsophon_soc_${x.y.z}_aarch64/opt/sophon/libsophon-${x.y.z}/lib ${soc-sdk}
    cp -rf libsophon_soc_${x.y.z}_aarch64/opt/sophon/libsophon-${x.y.z}/include ${soc-sdk}
    ```

3. 打包sophon-ffmpeg和sophon-opencv

    从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的SOPHONSDK，在sophon-mw_{date}_{time}文件夹中包括sophon-mw-soc_x.y.z_aarch64.tar.gz，x.y.z表示版本号，并进行解压，如果您使用BM1688 SOPHONSDK，“sophon-mw”这个名字或许需要替换成“sophon-media”。
    ```bash
    # 解压sophon-mw-soc_x.y.z_aarch64.tar.gz
    tar -zxf sophon-mw-soc_${x.y.z}_aarch64.tar.gz
    # 将ffmpeg和opencv的库目录和头文件目录拷贝到soc-sdk目录下
    cp -rf sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-ffmpeg_${x.y.z}/lib ${soc-sdk}
    cp -rf sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-ffmpeg_${x.y.z}/include ${soc-sdk}
    cp -rf sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-opencv_${x.y.z}/lib ${soc-sdk}
    cp -rf sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-opencv_${x.y.z}/include ${soc-sdk}
    ```

4. 如果您使用BM1688的GeminiSDK1.3以上版本，您还需要做这些操作：
    从sdk中获取sophon-img/bsp-debs/目录下的sophon-soc-libisp_${x.y.z}_arm64.deb，然后运行如下命令：
    ```
    dpkg -x sophon-soc-libisp_${x.y.z}_arm64.deb sophon-libisp
    cp -rf sophon-libisp/opt/sophon/sophon-soc-libisp_${x.y.z}/lib ${soc-sdk}
    ```

这里，交叉编译环境已经搭建完成，接下来可以使用打包好的soc-sdk编译需要在SoC平台上运行的程序。更多交叉编译信息请参考《LIBSOPHON使用手册.pdf》。

### 4.2 交叉编译安装sophon-sail
如果例程依赖sophon-sail则需要编译和安装sophon-sail，否则可跳过本章节。需要在x86主机上交叉编译sophon-sail，并在SoC平台上安装。

需从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的SOPHONSDK，进入sophon-sail_{date}_{time}文件夹，sophon-sail的发布包命名如sophon-sail_x.y.z.tar.gz，x.y.z表示版本号，您可以打开同级目录下的用户手册(命名为sophon-sail_zh.pdf或SOPHON-SAIL_zh.pdf)。
参考编译安装指南章节，选择您需要的模式(C++/Python，SoC MODE)进行安装，**注意需要选择包含ffmpeg和opencv的编译方式。**


在您按照教程将sophon-sail的库文件拷贝到目标soc上之后，您还需要设置以下环境变量：
```bash
echo 'export LD_LIBRARY_PATH=/opt/sophon/sophon-sail/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
## 5 arm PCIe平台的开发和运行环境搭建
如果您在arm平台安装了PCIe加速卡，开发环境与运行环境可以是统一的，您可以直接在宿主机上搭建开发和运行环境。
这里提供银河麒麟v10机器的环境安装方法，其他类型机器具体请参考官网开发手册。
### 5.1 安装libsophon
从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的SOPHONSDK，解压后在libsophon_{date}_{time}文件夹下面有安装包，
安装包由一个文件构成，其中“$arch”为当前机器的硬件架构，使用以下命令可以获取当前服务器的arch：
```
uname -m
```
通常x86_64机器对应的硬件架构为x86_64，arm64机器对应的硬件架构为aarch64：
```
libsophon_x.y.z_$arch.tar.gz，x.y.z表示版本号
```
可以通过如下步骤安装：

**注意：如果有旧版本，先参考下面的卸载方式步骤卸载旧版本。**
```
tar -xzvf libsophon_${x.y.z}_aarch64.tar.gz
sudo cp -r libsophon_${x.y.z}_aarch64/* /
sudo ln -s /opt/sophon/libsophon-${x.y.z} /opt/sophon/libsophon-current
```
接下来请先按照您所使用Linux发行版的要求搭建驱动编译环境，然后做如下操作：
```
sudo ln -s /opt/sophon/driver-${x.y.z}/$bin /lib/firmware/bm1684x_firmware.bin
sudo ln -s /opt/sophon/driver-${x.y.z}/$bin /lib/firmware/bm1684_ddr_firmware.bin
sudo ln -s /opt/sophon/driver-${x.y.z}/$bin /lib/firmware/bm1684_tcm_firmware.bin
cd /opt/sophon/driver-${x.y.z}
```
此处“$bin”是带有版本号的bin文件全名, 对于bm1684x板卡，为a53lite_pkg.bin，对于bm1684板卡，如bm1684_ddr.bin_v3.1.1-63a8614d-220906和bm1684_tcm.bin_v3.1.1-63a8614d-220906。

之后就可以编译驱动了（这里不依赖于dkms）：
```
sudo make SOC_MODE=0 PLATFORM=asic SYNC_API_INT_MODE=1 \
          TARGET_PROJECT=sg_pcie_device FW_SIMPLE=0 \
          PCIE_MODE_ENABLE_CPU=1
sudo cp ./bmsophon.ko /lib/modules/$(uname -r)/kernel/
sudo depmod
sudo modprobe bmsophon
```
最后是一些配置工作：

添加库和可执行文件路径：
```
sudo cp /opt/sophon/libsophon-current/data/libsophon.conf /etc/ld.so.conf.d/
sudo ldconfig
sudo cp /opt/sophon/libsophon-current/data/libsophon-bin-path.sh /etc/profile.d/
```
在终端执行如下命令，或者登出再登入当前用户后即可使用bm-smi等命令：
```
source /etc/profile
```
添加cmake config文件：
```
sudo mkdir -p /usr/lib/cmake/libsophon
sudo cp /opt/sophon/libsophon-current/data/libsophon-config.cmake /usr/lib/cmake/libsophon/
```
卸载方式：
```
sudo rm -f /etc/ld.so.conf.d/libsophon.conf
sudo ldconfig
sudo rm -f /etc/profile.d/libsophon-bin-path.sh
sudo rm -rf /usr/lib/cmake/libsophon
sudo rmmod bmsophon
sudo rm -f /lib/modules/$(uname -r)/kernel/bmsophon.ko
sudo depmod
sudo rm -f /lib/firmware/bm1684x_firmware.bin
sudo rm -f /lib/firmware/bm1684_ddr_firmware.bin
sudo rm -f /lib/firmware/bm1684_tcm_firmware.bin
sudo rm -f /opt/sophon/libsophon-current
sudo rm -rf /opt/sophon/libsophon-0.4.6
sudo rm -rf /opt/sophon/driver-0.4.6
```
其他平台机器请参考《LIBSOPHON使用手册.pdf》

### 5.2 安装sophon-ffmpeg和sophon-opencv
从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的SOPHONSDK，解压后在sophon-mw_{date}_{time}文件夹下面有安装包，
安装包由一个文件构成：
```
sophon-mw_x.y.z_aarch64.tar.gz，x.y.z表示版本号
```
可以通过如下步骤安装：

先按照《LIBSOPHON使用手册》安装好libsophon包，然后，
```
tar -xzvf sophon-mw_${x.y.z}_aarch64.tar.gz
sudo cp -r sophon-mw_${x.y.z}_aarch64/* /
sudo ln -s /opt/sophon/sophon-ffmpeg_${x.y.z} /opt/sophon/sophon-ffmpeg-latest
sudo ln -s /opt/sophon/sophon-opencv_${x.y.z} /opt/sophon/sophon-opencv-latest
sudo ln -s /opt/sophon/sophon-sample_${x.y.z} /opt/sophon/sophon-sample-latest
sudo sed -i "s/usr\/local/opt\/sophon\/sophon-ffmpeg-latest/g" /opt/sophon/sophon-ffmpeg-latest/lib/pkgconfig/*.pc
sudo sed -i "s/^prefix=.*$/prefix=\/opt\/sophon\/sophon-opencv-latest/g" /opt/sophon/sophon-opencv-latest/lib/pkgconfig/opencv4.pc
```
最后，**安装bz2 libc6 libgcc依赖库**（这部分需要根据操作系统不同，选择对应的安装包，这里不统一介绍）
然后是一些配置工作：

添加库和可执行文件路径：
```
sudo cp /opt/sophon/sophon-ffmpeg-latest/data/01_sophon-ffmpeg.conf /etc/ld.so.conf.d/
sudo cp /opt/sophon/sophon-opencv-latest/data/02_sophon-opencv.conf /etc/ld.so.conf.d/
sudo ldconfig
sudo cp /opt/sophon/sophon-ffmpeg-latest/data/sophon-ffmpeg-autoconf.sh /etc/profile.d/
sudo cp /opt/sophon/sophon-opencv-latest/data/sophon-opencv-autoconf.sh /etc/profile.d/
sudo cp /opt/sophon/sophon-sample-latest/data/sophon-sample-autoconf.sh /etc/profile.d/
source /etc/profile
```
其他平台机器请参考《MULTIMEDIA使用手册.pdf》、《MULTIMEDIA开发参考手册.pdf》。

### 5.3 编译安装sophon-sail
如果例程依赖sophon-sail则需要编译和安装sophon-sail，否则可跳过本章节。

需从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的SOPHONSDK，解压后在sophon-sail_{date}_{time}文件夹里面有sophon-sail的压缩包，命名如sophon-sail_x.y.z.tar.gz，x.y.z表示版本号。
您可以打开sophon-sail压缩包里面提供的用户手册(命名为sophon-sail_zh.pdf)，参考编译安装指南章节，选择您需要的模式(C++/Python, ARM PCIE MODE)进行安装。

## 6 riscv PCIe平台的开发和运行环境搭建
如果您在riscv平台安装了PCIe加速卡，开发环境与运行环境可以是统一的，您可以直接在宿主机上搭建开发和运行环境。
这里提供SG2042服务器的环境安装方法，其他类型机器具体请参考官网开发手册。

### 6.1 安装libsophon
从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的SOPHONSDK，解压后在libsophon_{date}_{time}文件夹下面有安装包，
安装包由以下3个文件构成：
```bash
sophon-libsophon-dev-{x.y.z}.riscv64.rpm
sophon-libsophon-{x.y.z}.riscv64.rpm
sophon-driver-{x.y.z}.riscv64.rpm
```
安装前需要通过后面“卸载方式”中的步骤卸载旧版本libsophon，可以通过如下步骤安装：
```bash
安装依赖库，只需要执行一次:
sudo yum install -y epel-release
sudo yum install -y dkms
sudo yum install -y ncurses*
安装libsophon：
sudo rpm -ivh sophon-driver-{x.y.z}.riscv64.rpm
sudo rpm -ivh sophon-libsophon-{x.y.z}.riscv64.rpm
sudo rpm -ivh --force sophon-libsophon-dev-{x.y.z}.riscv64.rpm
在终端执行如下命令，或者登出再登入当前用户后即可使用bm-smi等命令：
source /etc/profile
```
卸载方式：
```bash
sudo rpm -e sophon-driver
sudo rpm -e sophon-libsophon-dev
sudo rpm -e sophon-libsophon
```
其他平台机器请参考《LIBSOPHON使用手册.pdf》。

### 6.2 安装sophon-ffmpeg和sophon-opencv
从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的SOPHONSDK，解压后在sophon-mw_{date}_{time}文件夹下面有安装包，

sophon-mw安装包由四个文件构成：
```bash
sophon-mw-sophon-ffmpeg_{x.y.z}_riscv64.rpm
sophon-mw-sophon-ffmpeg-dev_{x.y.z}_riscv64.rpm
sophon-mw-sophon-opencv_{x.y.z}_riscv64.rpm
sophon-mw-sophon-opencv-dev_{x.y.z}_riscv64.rpm
```
其中：

1. sophon-ffmpeg/sophon-opencv包含了ffmpeg/opencv运行时环境（库文件、工具等）；sophon-ffmpeg-dev/sophon-opencv-dev包含了开发环境（头文件、pkgconfig、cmake等）。如果只是在部署环境上安装，则不需要安装sophon-ffmpeg-dev/sophon-opencv-dev。

2. sophon-mw-sophon-ffmpeg依赖sophon-libsophon包，而sophon-mw-sophon-opencv依赖sophon-mw-sophon-ffmpeg，因此在安装次序上必须先安装libsophon,然后sophon-mw-sophon-ffmpeg,最后安装sophon-mw-sophon-opencv。

安装之前请参考"卸载方式"卸载老版本，安装步骤如下：
```bash
sudo rpm -ivh sophon-mw-sophon-ffmpeg_{x.y.z}_riscv64.rpm sophon-mw-sophon-ffmpeg-dev_{x.y.z}_riscv64.rpm
sudo rpm -ivh sophon-mw-sophon-opencv_{x.y.z}_riscv64.rpm sophon-mw-sophon-opencv-dev_{x.y.z}_riscv64.rpm
在终端执行如下命令，或者logout再login当前用户后即可使用安装的工具：
source /etc/profile
```

卸载方式：
```bash
sudo rpm -e sophon-mw-sophon-opencv-dev
sudo rpm -e sophon-mw-sophon-opencv
sudo rpm -e sophon-mw-sophon-ffmpeg-dev
sudo rpm -e sophon-mw-sophon-ffmpeg
```

其他平台机器请参考《MULTIMEDIA使用手册.pdf》、《MULTIMEDIA开发参考手册.pdf》。

### 6.3 编译安装sophon-sail
如果例程依赖sophon-sail则需要编译和安装sophon-sail，否则可跳过本章节。

需从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)上下载符合[环境依赖](../README.md#环境依赖)的SOPHONSDK，解压后在sophon-sail_{date}_{time}文件夹里面有sophon-sail的压缩包，命名如sophon-sail_x.y.z.tar.gz，x.y.z表示版本号。
您可以打开sophon-sail压缩包里面提供的用户手册(命名为sophon-sail_zh.pdf)，参考编译安装指南章节，选择您需要的模式(C++/Python，RISCV PCIE MODE)进行安装。
