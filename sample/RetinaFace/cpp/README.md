# C++例程
cpp目录下提供了一系列C++例程以供参考使用，具体情况如下：
| 序号  | C++例程           | 说明                                 |
| ---- | -------------     | -----------------------------------  |
| 1    | retinaface_bmcv   | 使用OpenCV解码、BMCV前处理、BMRT推理   |


## 1. x86 PCIe 平台

## 1.1 环境准备
如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)和sophon-ffmpeg(>=0.2.4),具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)。

## 1.2 程序编译
C++程序需要编译可执行文件，
```bash
cd retinaface_bmcv
mkdir build && cd build
cmake .. && make # 生成retinaface_bmcv.pcie
```

## 1.3 测试命令
编译完成后，会在build文件夹下生成retinaface_bmcv.pcie,具体参数说明如下：

```bash
usage:./retinaface_bmcv.pcie <input mode> <input path> <bmodel path> <nms threshold> <conf threshold>
input mode: 0表示图片，1表示视频流
input path: 输入测试图片集路径或者视频路径；
bmodel path: 用于推理的bmodel路径，默认使用stage 0的网络进行推理；
nms threshold: nms的阈值
conf threshold: 框置信度阈值
```

测试实例如下：

```bash
# 图片模式，1batch，fp32，以测试face文件夹为例
# 如果模型是多batch的，会每攒够batch数的图片做一次推理
# 对于数据face和WIDERVAL，不同bmodel使用相同的<nms threshold>、<conf threshold>参数，这些阈值与数据集相关
$ ./retinaface_bmcv.pcie 0 ../../../data/images/face ../../../data/models/BM1684X/retinaface_mobilenet0.25_fp32_1b.bmodel 0.5 0.02
$ ./retinaface_bmcv.pcie 0 ../../../data/images/WIDERVAL ../../../data/models/BM1688/retinaface_mobilenet0.25_int8_4b.bmodel 0.4 0.02
```

执行完毕后，结果图片和文本文件保存在`results/`文件夹中。

```bash
# 视频模式，1batch或4batch，fp32或fp16或int8
$ ./retinaface_bmcv.pcie 1 ../../../data/videos/station.avi  ../../../data/models/BM1684X/retinaface_mobilenet0.25_fp32_1b.bmodel 0.5 0.02
```
执行完毕后，结果图片和文本文件保存在当前目录下的`results/`的文件夹。

可通过改变模型进行batch_size=4推理。


## 2. arm SoC平台

### 2.1 环境准备
对于SoC平台，内部已经集成了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，位于`/opt/sophon/`下。

## 2.2 交叉编译
通常在x86主机上交叉编译程序，使之能够在SoC平台运行。您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)和sophon-ffmpeg(>=0.2.4)运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件。
编译retinaface_bmcv方法如下：
```bash
$ cd retinaface_bmcv 
$ mkdir build && cd build 
$ cmake -DTARGET_ARCH=soc -DSDK=/{path_to_sdk}/soc-sdk .. && make # 生成retinaface_bmcv.soc
```

### 2.3 测试命令
将生成的可执行文件及所需的模型和测试图片拷贝到SoC目标平台中测试，测试方法请参考x86 PCIe平台的1.3测试命令，需要修改命令中的可执行文件名。
