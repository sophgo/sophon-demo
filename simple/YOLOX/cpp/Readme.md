# Example of YOLOX with Sophon Inference

**this example can run in pcie with docker and soc**

## For pcie with docker

### Environment configuration 

```shell
libsophon sophon-ffmpeg sophon-opencv sophon-sail should be download and install
configure the include and lib path in Makefile.pcie
```
### Build example
```
cmake CMakeLists.txt -DTARGET_ARCH={x86 or soc} -DSAIL_DIR=${SAIL_DIR}
make
```

### Run example

``` shell
./yolox_sail.pcie video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>
```
- video           :test file is video, otherwise is picture
- video url       :video name or picture path
- bmodel path     : bmodel file name
- test count      : video inference count, does not take effect in picture mode
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

```shell
libsophon-soc sophon-ffmpeg-soc sophon-opencv-soc sophon-sail should be download and install
configure the include and lib path in Makefile.arm
```

### Build example

``` shell
cmake CMakeLists.txt -DTARGET_ARCH={x86 or soc} -DSAIL_DIR=${SAIL_DIR} -DSDK=${SOC_SDK}
make
```

### Copy build result to soc

### Run example(in soc)

``` shell
./yolox_sail.arm video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>
```
- video           :test file is video, otherwise is picture
- video url       :video name or picture path
- bmodel path     : bmodel file name
- test count      : video inference count, does not take effect in picture mode
- detect threshold: detect threshold(0.25)
- nms threshold   : nms threshold(0.45)
- save path       : result save path
- device id       : device id

### result

result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_cpp.txt

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_[device id].jpg, save txt name is [video name]_[bmodel name]_cpp.txt
