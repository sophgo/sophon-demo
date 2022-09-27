# C++例程
cpp目录下提供了一系列C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | ssd_opencv | 使用OpenCV解码、OpenCV前处理、BMRT推理 |
| 2    | ssd_bmcv   | 使用OpenCV解码、BMCV前处理、BMRT推理   |


## 1. x86 PCIe 平台

## 1.1 环境准备

如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装 libsophon、sophon-opencv和sophon-ffmpeg。libsophon的安装可参考[LIBSOPHON使用手册]()，sophon-opencv和sophon-ffmpeg的安装可参考[multimedia开发参考手册]()。

## 1.2 程序编译
C++程序需要编译可执行文件，ssd_opencv和ssd_bmcv编译方法相同，以编译ssd_opencv程序为例：
```bash
cd ssd_opencv
mkdir build && cd build
cmake .. && make # 生成ssd_opencv.pcie
```

## 1.3 测试命令

编译完成后，会生成ssd_opencv.pcie，cpp例程暂时只支持batchsize=1推理，具体参数说明如下：

```bash
usage:./ssd_opencv.pcie <video/image> <input path> <bmodel path> <loop count> <device id>
video/image:使用视频或者图片；
input path:推理视频/图片路径；
bmodel path:用于推理的bmodel路径，默认使用stage 0的网络进行推理；
loop count:推理循环次数；
device id:用于推理的tpu设备id。
```

测试实例如下：

```bash
# 测试视频 
./ssd_opencv.pcie video ../../../data/videos/test_car_person.mp4 ../../../data/models/BM1684/ssd300_fp32_1b.bmodel 1 0
```

可通过改变模型进行int8推理。

执行完成后，会将预测结果图片保存在`results/`下，同时会打印预测结果、推理时间等信息。

```bash
** f32
class id:  6 upper-left: ( 238.21771,    3.54632)  object-size: ( 401.86188,  338.74353)
class id: 15 upper-left: ( 101.13682,  143.01151)  object-size: (  54.64425,  156.48785)
class id:  0 upper-left: (   0.00000,    0.00000)  object-size: (   1.00000,    1.00000)
Open /dev/bm-sophon0 successfully, device index = 0, jpu fd = 17, vpp fd = 17
exit: stream EOF

############################
SUMMARY: detect 
############################
[           detection]  loops:    1 avg: 86249 us
[     ssd pre-process]  loops:    1 avg: 46107 us
[       ssd inference]  loops:    1 avg: 39329 us
[    ssd post-process]  loops:    1 avg: 102 us
```

## 2. arm SoC平台
## 2.1 环境准备
对于arm SoC平台，内部已经集成了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，位于`/opt/sophon/`下。
## 2.2 交叉编译
通常在x86主机上交叉编译程序，使之能够在arm SoC平台运行。您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体可参考[LIBSOPHON使用手册]()。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件，ssd_opencv和ssd_bmcv编译方法相同，以编译ssd_opencv程序为例：
```bash
cd ssd_opencv
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make # 生成ssd_opencv.soc
```

## 2.3 测试命令
将生成的可执行文件及所需的模型和测试图片拷贝到SoC目标平台中测试，测试方法请参考x86 PCIe平台的[1.3 测试命令](#13-测试命令)。
