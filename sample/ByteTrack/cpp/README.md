简体中文 ｜ [English](../docs/README_CPP_EN.md)
# ByteTrack C++ 示例

**此示例可在PCIe和SoC上运行**

- [ByteTrack C++ 示例](#bytetrack-c-示例)
  - [1. 依赖安装](#1-依赖安装)
  - [2. PCIe](#2-pcie)
    - [2.1 环境配置](#21-环境配置)
    - [2.2 构建示例](#22-构建示例)
    - [2.3 运行示例](#23-运行示例)
  - [3. SoC](#3-soc)
    - [3.1 环境配置](#31-环境配置)
    - [3.2 构建示例](#32-构建示例)
    - [3.3 在SoC中运行](#33-在soc中运行)
  - [4 推理测试](#4-推理测试)
    - [4.1 测试MOT数据集](#41-测试mot数据集)
    - [4.2 测试视频](#42-测试视频)
    - [4.3 测试MOT指标](#43-测试mot指标)


## 1. 依赖安装
卡尔曼滤波(kalman_filter)依赖'eigen'，

```shell
sudo apt-get install libeigen3-dev
```

## 2. PCIe

### 2.1 环境配置

需要下载和安装libsophon、sophon-ffmpeg、sophon-opencv和sophon-sail等软件包，具体细节请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建).

### 2.2 构建示例

可以直接在PCIe平台上编译程序：

```shell
cd cpp/bytetrack_bmcv
mkdir build && cd build
cmake .. -DSAIL_DIR=${SAIL_DIR}
make
```

SAIL_DIR 是SAIL的文件位置, 通常在 /opt/sophon/sophon-sail.

### 2.3 运行示例

``` shell
./bytetrack_bmcv.pcie video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>

e.g
./bytetrack_bmcv.pcie image ../../datasets/MOT15/ADL-Rundle-6/img1 ../../models/BM1684/bytetrack_s_fp32_1b.bmodel 10 0.1 0.7 ./results 0

```

- video           : test file is video, otherwise is picture
- video url       : video name or picture path
- bmodel path     : bmodel file name
- test count      : inference count, does not take effect in picture mode
- detect threshold: detect threshold
- nms threshold   : nms threshold
- save path       : result save path
- device id       : device id

所有参数都需要给出

**结果**：

结果将保存在您指定的路径中。

对于图片，保存的图片名称与原始图片名称相同，保存的txt文件名称格式为 [ost picture name]_[bmodel name]_py.txt。

对于视频，结果保存在[video name]_[bmodel name].txt。

## 3. SoC

### 3.1 环境配置

您需要在 x86 主机上使用 sophon SDK 构建交叉编译环境，并将程序所依赖的头文件和库文件打包到 soc sdk 目录中。详见：[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。

### 3.2 构建示例

在x86 主机上，

``` shell
cd cpp/bytetrack_bmcv
mkdir build && cd build
cmake .. -DTARGET_ARCH=soc -DSAIL_DIR=/path-to-sail/sophon-sail -DSDK=/path_to_sdk/soc-sdk
make
```

SAIL_DIR 是通过上述交叉编译环境设置获得的路径，通常是 build_soc/sophon-sail 的路径。
SOC_SDK 在构建交叉编译环境时也会创建。

### 3.3 在SoC中运行

**把结果复制到SoC**

如果因为“error while loading shared libraries: libsail.so: cannot open shared object file: No such file or directory”而无法运行bytetrack_bmcv，请设置环境路径。

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
```

``` shell
./bytetrack_bmcv.soc video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>
```

```shell
e.g
./bytetrack_bmcv.soc video ../../datasets/sample.mp4 ../../models/BM1684/bytetrack_s_fp32_1b.bmodel 10 0.1 0.7 ./results 0
```

- video           : test file is video, otherwise is picture
- video url       : video name or picture folder path
- bmodel path     : bmodel file name
- test count      : inference count, does not take effect in picture mode
- detect threshold: detect threshold(0.25)
- nms threshold   : nms threshold(0.45)
- save path       : result save path
- device id       : device id

**结果**：

结果将保存在您指定的路径中。

对于图片，保存的图片名称与原始图片名称相同，保存的txt文件名称格式为 [ost picture name]_[bmodel name]_py.txt。

对于视频，结果保存在[video name]_[bmodel name].txt。

## 4 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以SOC模式进行介绍。

### 4.1 测试MOT数据集
```bash
./bytetrack_bmcv.soc image ../../datasets/MOT15/ADL-Rundle-6/img1 ../../models/BM1684/bytetrack_s_fp32_1b.bmodel 10 0.1 0.7 ./results 0
```

测试结束后，预测的结果保存在`results/[ost picture name]_[bmodel name]_py.txt`下，同时会打印推理耗时等信息。

```bash
############################
SUMMARY: bytetrack test
############################
[      time per frame]  loops:  100 avg: 52375 us
[          yolox time]  loops:  100 avg: 51598 us
[    yolox preprocess]  loops:  100 avg: 10845 us
[     yolox inference]  loops:  100 avg: 40556 us
[   yolox postprocess]  loops:  100 avg: 190 us
[      bytetrack time]  loops:  100 avg: 768 us
save detect result: ./results/img1_bytetrack_s_fp32_1b_cpp.txt
```

### 4.2 测试视频
视频测试实例如下，支持对视频流进行测试。
```bash
e.g
./bytetrack_bmcv.soc video ../../datasets/sample.mp4 ../../models/BM1684/bytetrack_s_fp32_1b.bmodel 100 0.1 0.7 ./results 0
```

测试结束后，预测的结果保存在`results/[video name]_[bmodel name].txt`下，同时会打印推理耗时等信息。

```bash
############################
SUMMARY: bytetrack test
############################
[      time per frame]  loops:  100 avg: 46310 us
[          yolox time]  loops:  100 avg: 45524 us
[    yolox preprocess]  loops:  100 avg: 4730 us
[     yolox inference]  loops:  100 avg: 40514 us
[   yolox postprocess]  loops:  100 avg: 276 us
[      bytetrack time]  loops:  100 avg: 772 us
save detect result: ./results/sample_bytetrack_s_fp32_1b_cpp.txt
```

### 4.3 测试MOT指标
运行 eval_mot.py 来计算 MOT 指标，其中 ground_truths 是测试数据集的标注文件，通常为 datasets/MOT15/ADL-Rundle-6/gt/gt.txt。--detections 是检测结果文件，位于 cpp/results 和 python/{bytetrack}/results 下。

``` shell
    pip3 install motmetrics
    python3 ../tools/eval_mot.py \
        --ground_truths=your-ground_truths-file \  # txt file
        --detections=your-detections-file   # txt file
```

**For example:**

``` bash
    pip3 install motmetrics
    python3 ../tools/eval_mot.py \
      --ground_truths=../datasets/MOT15/ADL-Rundle-6/gt/gt.txt \
      --detections=../cpp/bytetrack_bmcv/results/img1_bytetrack_s_fp32_1b_cpp.txt
```

输出结果：
```bash
MOTA = -0.4791375524056698
     num_frames      IDF1       IDP       IDR      Rcll     Prcn    GT  MT  PT  ML    FP    FN  IDsw   FM      MOTA      MOTP
acc         525  0.049857  0.058351  0.043522  0.159114  0.21333  5009   0   7  17  2939  4212   258  562 -0.479138  0.342534
```