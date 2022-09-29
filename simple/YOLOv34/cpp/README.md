# C++例程
cpp目录下提供了一系列C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | yolov34_bmcv | 使用OpenCV解码、BMCV前处理、BMRT推理 |


## 1. x86 PCIe 平台

## 1.1 环境准备

如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon、sophon-opencv和sophon-ffmpeg。libsophon的安装可参考[LIBSOPHON使用手册]()，sophon-opencv和sophon-ffmpeg的安装可参考[multimedia开发参考手册]()。

## 1.2 程序编译
C++程序需要编译可执行文件，以编译yolov34_bmcv程序为例：
```bash
cd yolov34_bmcv
mkdir build && cd build
cmake .. && make # 生成yolov34_bmcv.pcie
```

## 1.3 测试命令

编译完成后，会生成yolov34_bmcv.pcie,具体参数说明如下：

```bash
usage:./yolov34_bmcv.pcie image <image list> <cfg file> <bmodel file> <test count> <device id> <conf thresh> <nms thresh>
input path:推理图片路径，输入Imagelist和videolist的路径；
bmodel path:用于推理的bmodel路径；
test count:测试数；
device id:用于推理的tpu设备id。
conf thresh:置信度阈值
nms thresh:nms阈值
```

测试实例如下：

```bash
# 测试图片  
./yolov34_bmcv.pcie image ../../../data/images/Imagelist.txt ../configs/yolov4.cfg ../../../data/models/BM1684/yolov4_416_coco_fp32_1b.bmodel 4 0 0.5 0.45
# 测试视频
./yolov34_bmcv.pcie video ../../../data/video/videolist.txt ../configs/yolov4.cfg ../../../data/models/BM1684/yolov4_416_coco_fp32_1b.bmodel 300 0 0.5 0.45
```

根据模型修改conf thresh和nms thresh数值
可通过改变模型进行int8及batch_size=4推理。

执行完成后，会将预测结果保存在`result_imgs/`下，同时会打印预测结果、推理时间等信息。

```bash
sampleFactor=6, cinfo->num_components=3 (1x2, 1x1, 1x1)
Category: 17 Score: 0.993326 : 407,139,598,346
Category: 16 Score: 0.994726 : 60,263,202,346
Category: 0 Score: 0.998255 : 191,96,274,371
Open /dev/bm-sophon1 successfully, device index = 1, jpu fd = 14, vpp fd = 14

############################
SUMMARY: detect 
############################
[      decode overall]  loops:    1 avg: 3784 us
[     stage 0: decode]  loops:    1 avg: 3782 us
[   detection overall]  loops:    1 avg: 54474 us
[stage 1: pre-process]  loops:    1 avg: 907 us
[stage 2: detection  ]  loops:    1 avg: 52072 us
[stage 3:post-process]  loops:    1 avg: 1494 us
```

注：

1. 程序执行完毕后，会通过终端打印的方式给出各阶段耗时

2. 耗时统计存在略微波动属于正常现象

3. 测试视频时，如果输入多batch模型，需要输入匹配数量的视频文件

4. 更换图片以及视频需要修改`/data/images/Imagelist.txt`以及及`/data/video/videolist.txt`,`Imagelist.txt`内容如下：

   ```bash
   ../../../data/images/person.jpg
   ../../../data/images/dog.jpg
   ../../../data/images/bus.jpg
   ../../../data/images/zidane.jpg
   ../../../data/images/horses.jpg
   ```



## 2. arm SoC平台

## 2.1 环境准备
对于arm SoC平台，内部已经集成了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，位于`/opt/sophon/`下。
## 2.2 交叉编译
通常在x86主机上交叉编译程序，使之能够在arm SoC平台运行。您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体可参考[LIBSOPHON使用手册]()。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件，以编译yolov34_bmcv程序为例：
```bash
cd yolov34_bmcv
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make # 生成yolov34_bmcv.soc
```

## 2.3 测试命令
将生成的可执行文件及所需的模型和测试图片拷贝到SoC目标平台中测试，测试方法请参考x86 PCIe平台的[1.3 测试命令](#13-测试命令)。