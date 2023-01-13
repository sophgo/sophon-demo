# Example of YOLOX with Sophon Inference

**this example can run in pcie and soc**

## For pcie 

### Environment configuration 


libsophon sophon-ffmpeg sophon-opencv sophon-sail should be download and installed. For details, see [x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。


### Build example
```
cmake CMakeLists.txt -DTARGET_ARCH=x86 -DSAIL_DIR=${SAIL_DIR}
make
```
SAIL_DIR is the path obtained from the above environment setup, normally /opt/sophon/sophon-sail.

### Run example

``` shell
./yolox_sail.pcie video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>

e.g
./yolox_sail.pcie image ../data/image/val2017 ../data/models/BM1684/yolox_s_fp32_1b.bmodel 1 0.25 0.45 ./results 1

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
### Result
result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name].txt

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_[device id].jpg, save txt name is [video name]_[bmodel name].txt


## For soc

### Environment configuration

You need to use the SOPHON SDK on the x86 host to build a cross compilation environment, and package the header files and library files that the program depends on into the soc sdk directory. For details, see [交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#31-交叉编译环境搭建).

### Build example

``` shell
cmake CMakeLists.txt -DTARGET_ARCH=soc -DSAIL_DIR=${SAIL_DIR} -DSDK=${SOC_SDK}
make
```
SAIL_DIR is the path obtained from the above cross compile environment setup，normally the path to build_soc/sophon-sail
SOC_SDK is also created when you build the the cross compilation environment.

### Copy build result to soc

### Run example(in soc)

if yolox_sail.arm can not run because of "error while loading shared libraries: libsail.so: cannot open shared object file: No such file or directory", please set the environment path

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
```

``` shell
./yolox_sail.arm video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>
```
e.g ./yolox_sail.arm video ../data/video/1080p_1.mp4 ../data/models/BM1684/yolox_s_fp32_1b.bmodel 10 0.25 0.45 ./results 0
- video           : test file is video, otherwise is picture
- video url       : video name or picture folder path
- bmodel path     : bmodel file name
- test count      : inference count, does not take effect in picture mode
- detect threshold: detect threshold(0.25)
- nms threshold   : nms threshold(0.45)
- save path       : result save path
- device id       : device id

### result

result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_cpp.txt

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_[device id].jpg, save txt name is [video name]_[bmodel name]_cpp.txt
