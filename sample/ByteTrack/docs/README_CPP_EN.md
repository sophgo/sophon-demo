[简体中文](../cpp/README.md) | English
# ByteTrack C++ Example

**this example can run in PCIe and SoC**

- [ByteTrack C++ Example](#bytetrack-c-example)
  - [1. Install Package](#1-install-package)
  - [2. For PCIe](#2-for-pcie)
    - [2.1 Environment Configuration](#21-environment-configuration)
    - [2.2 Build Example](#22-build-example)
    - [2.3 Run Example](#23-run-example)
  - [3. For SoC](#3-for-soc)
    - [3.1 Environment Configuration](#31-environment-configuration)
    - [3.2 Build Example](#32-build-example)
    - [3.3 Run example(in SoC)](#33-run-examplein-soc)
  - [4 Inference Test](#4-inference-test)
    - [4.1 Test MOT Dataset](#41-test-mot-dataset)
    - [4.2 Test Video](#42-test-video)
  - [4.3 Calculate MOT Metrics](#43-calculate-mot-metrics)

## 1. Install Package

kalman_filter rely on package 'eigen',

```shell
sudo apt-get install libeigen3-dev
```

## 2. For PCIe

### 2.1 Environment Configuration

libsophon sophon-ffmpeg sophon-opencv sophon-sail should be download and installed. For details, refer to [x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)。

You can directly compile the program on the PCIe platform:

### 2.2 Build Example

```shell
cd cpp/bytetrack_bmcv
mkdir build && cd build
cmake .. -DSAIL_DIR=${SAIL_DIR}
make
```

SAIL_DIR is the path obtained from the above environment setup, normally /opt/sophon/sophon-sail.

### 2.3 Run Example

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

all params needed

**Result**

result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name].txt

for video, save txt name is [video name]_[bmodel name].txt


## 3. For SoC

### 3.1 Environment Configuration

You need to use the sophon SDK on the x86 host to build a cross compilation environment, and package the header files and library files that the program depends on into the soc sdk directory. For details, see [交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建).

### 3.2 Build Example

On the x86 host,

``` shell
cd cpp/bytetrack_bmcv
mkdir build && cd build
cmake .. -DTARGET_ARCH=soc -DSAIL_DIR=/path-to-sail/sophon-sail -DSDK=/path_to_sdk/soc-sdk
make
```

SAIL_DIR is the path obtained from the above cross compile environment setup，normally the path to build_soc/sophon-sail
SOC_SDK is also created when you build the the cross compilation environment.

### 3.3 Run example(in SoC)

**Copy build result to SoC**

if bytetrack_bmcv.soc can not run because of "error while loading shared libraries: libsail.so: cannot open shared object file: No such file or directory", please set the environment path

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
```

``` shell
./bytetrack_bmcv.soc video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>
```

```shell
e.g ./bytetrack_bmcv.soc video ../../datasets/sample.mp4 ../../models/BM1684/bytetrack_s_fp32_1b.bmodel 10 0.1 0.7 ./results 0
```

- video           : test file is video, otherwise is picture
- video url       : video name or picture folder path
- bmodel path     : bmodel file name
- test count      : inference count, does not take effect in picture mode
- detect threshold: detect threshold(0.25)
- nms threshold   : nms threshold(0.45)
- save path       : result save path
- device id       : device id

**Result**

result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_cpp.txt

for video, save txt name is [video name]_[bmodel name]_cpp.txt


## 4 Inference Test
For PCIe platforms, inference testing can be performed directly on the PCIe platform; for SoC platforms, the cross-compiled executable file and required models and test data must be copied to the SoC platform for testing. The test parameters and operation methods are consistent. The following mainly introduces SOC mode。

### 4.1 Test MOT Dataset
```bash
./bytetrack_bmcv.soc image ../../datasets/MOT15/ADL-Rundle-6/img1 ../../models/BM1684/bytetrack_s_fp32_1b.bmodel 10 0.1 0.7 ./results 0
```

After the test is completed, the predicted results are saved in results/[ost picture name]_[bmodel name]_py.txt, and the inference time and other information will be printed.

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

### 4.2 Test Video
The video test example is as follows, and it supports testing of video streams.
```bash
e.g
./bytetrack_bmcv.soc video ../../datasets/sample.mp4 ../../models/BM1684/bytetrack_s_fp32_1b.bmodel 100 0.1 0.7 ./results 0
```

After the test is completed, the predicted results are saved in results/[video name]_[bmodel name].txt, and the inference time and other information will be printed.

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

## 4.3 Calculate MOT Metrics

Run mot_eval.py to calculate MOT metrics, ground_truths is the lable file of the test dataset, normally data/MOT15/dataset-name/gt/gt.txt. The detections is the detect result file, under cpp/results and python/{bytetrack}/results.

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

**Result:**
```bash
MOTA = -0.4791375524056698
     num_frames      IDF1       IDP       IDR      Rcll     Prcn    GT  MT  PT  ML    FP    FN  IDsw   FM      MOTA      MOTP
acc         525  0.049857  0.058351  0.043522  0.159114  0.21333  5009   0   7  17  2939  4212   258  562 -0.479138  0.342534
```