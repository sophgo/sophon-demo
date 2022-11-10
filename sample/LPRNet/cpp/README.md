# C++例程
cpp目录下提供了一系列C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | lprnet_opencv | 使用OpenCV解码、OpenCV前处理、BMRT推理 |
| 2    | lprnet_bmcv   | 使用OpenCV解码、BMCV前处理、BMRT推理   |


## 1. x86 PCIe 平台

## 1.1 环境准备

如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon、sophon-opencv和sophon-ffmpeg,具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。

## 1.2 程序编译
C++程序需要编译可执行文件，lprnet_opencv和lprnet_bmcv编译方法相同，以编译lprnet_opencv程序为例：
```bash
cd lprnet_opencv
mkdir build && cd build
cmake .. && make # 生成lprnet_opencv.pcie
```

## 1.3 测试命令

编译完成后，会生成lprnet_opencv.pcie,具体参数说明如下：

```bash
usage:./lprnet_opencv.pcie <input path> <bmodel path> <device id>
input path:推理图片路径，可输入整个推理图片文件夹的路径；
bmodel path:用于推理的bmodel路径，默认使用stage 0的网络进行推理；
device id:用于推理的tpu设备id。
```

测试实例如下，请根据目标平台、模型精度、batch_size选择相应的bmodel：

```bash
# 测试整个文件夹
./lprnet_opencv.pcie ../../../data/images/test ../../../data/models/BM1684X/lprnet_fp32_1b.bmodel 0
```

执行完成后，会将预测结果保存在`results/lprnet_fp32_1b.bmodel_test_opencv_cpp_result.json`下，同时会打印预测结果、推理时间等信息。

```bash
......
豫RM6396.jpg pred:皖RM6396
闽D33U29.jpg pred:皖D33U29
鲁AW9V20.jpg pred:鲁AW9V20
鲁BE31L9.jpg pred:鲁BE31L9
鲁Q08F99.jpg pred:鲁Q08F99
鲁R8D57Z.jpg pred:鲁R8D57Z
================
result saved in results/lprnet_fp32_1b.bmodel_test_opencv_cpp_result.json
================
infer_time = 0.745000ms
QPS = 625

############################
SUMMARY: lprnet detect
############################
[      lprnet overall]  loops:    1 avg: 1600402 us
[          read image]  loops:  100 avg: 388 us
[           detection]  loops:  100 avg: 1178 us
[  lprnet pre-process]  loops:  100 avg: 151 us
[    lprnet inference]  loops:  100 avg: 745 us
[ lprnet post-process]  loops:  100 avg: 238 us

```

## 2. SoC平台
## 2.1 环境准备
对于SoC平台，内部已经集成了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，位于`/opt/sophon/`下。

## 2.2 交叉编译
通常在x86主机上交叉编译程序，使之能够在SoC平台运行。您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../docs/Environment_Install_Guide.md#31-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件，lprnet_opencv和lprnet_bmcv编译方法相同，以编译lprnet_opencv程序为例：
```bash
cd lprnet_opencv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make # 生成lprnet_opencv.soc
```

## 2.3 测试命令
将生成的可执行文件及所需的模型、测试图片拷贝到SoC目标平台中测试，测试方法请参考x86 PCIe平台的[1.3 测试命令](#13-测试命令)。

## 3. 精度与性能测试

### 3.1 精度测试
本例程在`tools`目录下提供了`eval.py`脚本，可以将前面生成的预测结果json文件与测试集标签的json文件进行对比，计算出车牌识别准确率。具体的测试命令如下：
```bash
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval.py --label_json data/images/test_label.json --result_json cpp/lprnet_opencv/build/results/lprnet_fp32_1b.bmodel_test_opencv_cpp_result.json
```
执行完成后，会打印出车牌识别的准确率：
```bash
INFO:root:ACC = 880/1000 = 0.882
```

### 3.2 性能测试

可以使用bmrt_test测试模型的理论性能：
```bash
bmrt_test --bmodel {path_of_bmodel}
```
也可以参考[1.3 测试命令](#13-测试命令)打印程序运行中的实际性能指标。  
测试中性能指标存在一定的波动属正常现象。

### 3.3 测试结果

[LNRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)中模型在该测试集上的准确率为89.4%。

在BM1684X SoC上，不同例程、不同模型的精度和性能测试结果如下：

|     例程      | 精度 |batch_size|  ACC  |bmrt_test|infer_time| QPS |
|   --------    | ---- | -------  | ----- |  -----  | -----    | --- |
| lprnet_opencv | fp32 |   1      | 88.2% |  0.8ms  |  0.7ms   | 620 |
| lprnet_opencv | fp32 |   4      | 89.3% |  0.7ms  |  0.7ms   | 660 |
| lprnet_opencv | int8 |   1      | 87.4% |  0.3ms  |  0.2ms   | 950 |
| lprnet_opencv | int8 |   4      | 87.9% |  0.2ms  |  0.2ms   | 1000 |
| lprnet_bmcv   | fp32 |   1      | 88.2% |  0.8ms  |  0.8ms   | 660 |
| lprnet_bmcv   | fp32 |   4      | 89.3% |  0.7ms  |  0.7ms   | 700 |
| lprnet_bmcv   | int8 |   1      | 87.4% |  0.3ms  |  0.2ms   | 1050 |
| lprnet_bmcv   | int8 |   4      | 87.9% |  0.2ms  |  0.2ms   | 1150 |

在BM1684 SoC上，不同例程、不同模型的精度和性能测试结果如下：

|     例程      | 精度 |batch_size|  ACC  |bmrt_test|infer_time| QPS |
|   --------    | ---- | -------  | ----- |  -----  | -----    | --- |
| lprnet_opencv | fp32 |   1      | 88.0% |  1.7ms  |  1.6ms   | 400 |
| lprnet_opencv | fp32 |   4      | 89.2% |  0.9ms  |  0.9ms   | 600 |
| lprnet_opencv | int8 |   1      | 87.3% |  0.7ms  |  0.7ms   | 660 |
| lprnet_opencv | int8 |   4      | 88.4% |  0.3ms  |  0.2ms   | 960 |
| lprnet_bmcv   | fp32 |   1      | 88.0% |  1.7ms  |  1.6ms   | 400 |
| lprnet_bmcv   | fp32 |   4      | 89.2% |  0.9ms  |  0.9ms   | 620 |
| lprnet_bmcv   | int8 |   1      | 87.3% |  0.7ms  |  0.7ms   | 660 |
| lprnet_bmcv   | int8 |   4      | 88.4% |  0.3ms  |  0.2ms   | 1000 |
```
bmrt_test: 每张图片的理论推理时间；
infer_time: 程序运行时每张图片的网络推理时间；
QPS: 程序每秒钟全流程处理的图片数。
```

> **测试说明**：  
1.PCIe上的测试精度与SoC相同，性能由于CPU的不同可能存在一定差异；  
2.LPRNet网络中包含mean算子，会把所有batch数据加和求平均，当多batch推理时，同一张图片在不同的batch组合中可能会有不同的推理结果;  
3.性能测试的结果具有一定的波动性。