# cv Demo

## 目录
- [cv Demo](#cv_demo)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1. SoC平台](#1-soc平台)
  - [2. 程序编译](#2-程序编译)
    - [2.1 SoC平台](#21-soc平台)
  - [3. 程序运行](#3-程序运行)
    - [3.1 Json配置说明](#31-json配置说明)
    - [3.2 运行](#32-运行)
  - [4. 可视化](#4-可视化)





## 1. 环境准备

### 1. SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。
(1) 安装摄像头
本例程适用于提供使用带J1901(母座)-Rx接口的算能模组二次开发底板或者算能evb板
(2) 安装驱动
安装驱动需要进入到超级权限，接着系统驱动目录，安装驱动：
鱼眼拼接使用04e10镜头，安装对应驱动
```bash
sudo -s
insmod /mnt/system/ko/v4l2_os04e10.ko
```

广角拼接使用04a10镜头，安装对应驱动
```bash
sudo -s
insmod /mnt/system/ko/v4l2_os04a10_sync.ko
```
（2）isp参数文件配置,cvi_sdr_bin在准备数据章节下载的data路径中

```bash
sudo -s
mkdir -p /mnt/cfg/param
cp data/param/cvi_sdr_bin /mnt/cfg/param
```

注意:每重启一次，应重新加载相应的驱动
## 2. 程序编译

### 2.1 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：
#### 2.1.1 bmcv
```bash
cd cv-demo/cpp
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
```
编译完成后，会在cv-demo/cpp目录下生成cvdemo.soc。

## 3. 程序运行

### 3.1 Json配置说明

cv_demo demo中各部分参数位于 [config](./config-04e10/) 与 [config](./config-04a10/)目录，结构如下所示：

```bash
./config-04e10/
├── camera_cv_demo.json          # demo按sensor输入的配置文件
├── cv_demo.json            # demo按图片输入的配置文件
├── dwa_L.json                  # 左侧输入的鱼眼展开配置文件
├── dwa_R.json                  # 右侧输入的鱼眼展开配置文件
└── blend.json                  # 拼接配置文件
./config-04a10/
├── camera_cv_demo.json          # demo按sensor输入的配置文件
├── cv_demo.json            # demo按图片输入的配置文件
├── dwa_L.json                  # 左侧输入的鱼眼展开配置文件
├── dwa_R.json                  # 右侧输入的鱼眼展开配置文件
└── blend.json                  # 拼接配置文件
```

其中，[camera_cv_demo.json](./config-04e10/camera_cv_demo.json)是例程的整体配置文件，管理输入码流等信息。在一张图上可以支持多路数据的输入，channels参数配置输入的路数，channel中包含码流url等信息。

已提供默认的config文件，运行时只需要将config-04e10或config-04a10改名为config即可，如
```bash
mv config-04e10/ config
```
### 3.2 运行

对于SoC平台，需将交叉编译生成的动态链接库、可执行文件、所需的模型和测试数据拷贝到SoC平台中测试。

SoC平台上，动态库、可执行文件、配置文件、模型、视频数据的目录结构关系应与原始sophon-demo仓库中的关系保持一致。


1. 运行可执行文件
```bash
./cvdemo.soc
```


## 4. 可视化
运行程序后修改cv_demo.html中 connectWebSocket 的对应ip与端口，默认端口为9002，可在代码对应main函数修改，并在可以与soc网络相通的机器客户端浏览器上打开cv_demo.html，注意前端没有帧率控制，网络不好的情况下可能会卡顿


## 5. 常用debug命令
一般运行demo之前请保证ut出图正常

ut测试命令如下
```bash
cd /opt/sophon/sophon-soc-libisp_1.0.0/bin/
./ispv4l2_ut 6
```
原始yuv图片会保存在/opt/sophon/sophon-soc-libisp_1.0.0/bin/路径下