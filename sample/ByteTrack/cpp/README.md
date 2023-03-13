# Example of ByteTrack with Sophon Inference

**this example can run in PCIe and SoC**

- [Example of ByteTrack with Sophon Inference](#example-of-bytetrack-with-sophon-inference)
  - [Install Package](#install-package)
  - [For PCIe](#for-pcie)
    - [Environment Configuration](#environment-configuration)
    - [Build Example](#build-example)
    - [Run Example](#run-example)
  - [For SoC](#for-soc)
    - [Environment Configuration](#environment-configuration-1)
    - [Build Example](#build-example-1)
    - [Copy build result to SoC](#copy-build-result-to-soc)
    - [Run example(in SoC)](#run-examplein-soc)
  - [Calculate MOT Metrics](#calculate-mot-metrics)

## Install Package

kalman_filter rely on package 'eigen', make sure your machine already has opencv. You can download this package by[Preparation](../README.md#3-preparation)

```shell
cd path-to-eigen
unzip eigen-x.x.x.zip
cd eigen-x.x.x
mkdir build
cd build
cmake ..
sudo make install
```

## For PCIe

### Environment Configuration

libsophon sophon-ffmpeg sophon-opencv sophon-sail should be download and installed. For details, refer to [x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)。


### Build Example

```shell
cmake CMakeLists.txt -DSAIL_DIR=${SAIL_DIR}
make
```

e.g
```shell
cmake CMakeLists.txt -DSAIL_DIR=/opt/sophon/sophon-sail
make
```

SAIL_DIR is the path obtained from the above environment setup, normally /opt/sophon/sophon-sail.

### Run Example

``` shell
./bytetrack-bmcv-cpp video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>

e.g
./bytetrack-bmcv-cpp image ../../data/MOT15/ADL-Rundle-6/img1 ../../data/models/BM1684/bytetrack_s_fp32_1b.bmodel 10 0.1 0.7 ./results 0

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

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_[device id].jpg, save txt name is [video name]_[bmodel name].txt


## For SoC

### Environment Configuration

You need to use the SOPHON SDK on the x86 host to build a cross compilation environment, and package the header files and library files that the program depends on into the soc sdk directory. For details, see [交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建).

### Build Example

``` shell
cmake CMakeLists.txt -DSAIL_DIR=${SAIL_DIR}
make
```

e.g
```shell
cmake CMakeLists.txt -DSAIL_DIR=/opt/sophon/sophon-sail
make
```

SAIL_DIR is the path obtained from the above cross compile environment setup，normally the path to build_soc/sophon-sail
SOC_SDK is also created when you build the the cross compilation environment.

### Copy build result to SoC

### Run example(in SoC)

if bytetrack-bmcv-cpp can not run because of "error while loading shared libraries: libsail.so: cannot open shared object file: No such file or directory", please set the environment path

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
```

``` shell
./bytetrack-bmcv-cpp video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>
```

```shell
e.g ./bytetrack-bmcv-cpp video ../../data/video/sample.mp4 ../../data/models/BM1684/bytetrack_s_fp32_1b.bmodel 10 0.1 0.7 ./results 0
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

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_[device id].jpg, save txt name is [video name]_[bmodel name]_cpp.txt


## Calculate MOT Metrics

Run mot_eval.py to calculate MOT metrics, ground_truths is the lable file of the test dataset, normally data/MOT15/dataset-name/gt/gt.txt. The detections is the detect result file, under cpp/results and python/{bytetrack-version}/results.

``` shell
    pip3 install motmetrics
    python3 ../tools/mot_eval.py \
        --ground_truths=your-ground_truths-file \  # txt file
        --detections=your-detections-file   # txt file
```

**For example:**

``` bash
    pip3 install motmetrics
    python3 ../tools/mot_eval.py \
      --ground_truths=../data/MOT15/ADL-Rundle-6/gt/gt.txt \
      --detections=../cpp/bytetrack_bmcv/results/img1_bytetrack_s_fp32_1b_cpp.txt
```