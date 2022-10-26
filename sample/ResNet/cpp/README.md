# C++例程

cpp目录下提供了一系列C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | resnet_opencv | 使用OpenCV解码、OpenCV前处理、BMRT推理 |
| 2    | resnet_bmcv   | 使用OpenCV解码、BMCV前处理、BMRT推理   |


## 1. x86 PCIe 平台

## 1.1 环境准备

如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)和sophon-ffmpeg(>=0.2.4),具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。

您可能还需要安装其他第三方库：
```bash
sudo apt-get install libboost-dev libjsoncpp-dev
```

## 1.2 程序编译
C++程序需要编译可执行文件，resnet_opencv和resnet_bmcv编译方法相同，以编译resnet_opencv程序为例：
```bash
cd resnet_opencv
mkdir build && cd build
cmake .. && make # 生成resnet_opencv.pcie
```

## 1.3 测试命令

编译完成后，会生成resnet_opencv.pcie,具体参数说明如下：

```bash
usage:./resnet_opencv.pcie <input path> <bmodel path> <device id>
input path:推理图片路径，可输入整个推理图片文件夹的路径；
bmodel path:用于推理的bmodel路径，默认使用stage 0的网络进行推理；
device id:用于推理的tpu设备id。
```

测试实例如下：

```bash
# 测试整个文件夹  
./resnet_opencv.pcie ../../../data/images/imagenet_val_1k/img ../../../data/models/BM1684X/resnet_fp32_b1.bmodel 0
```

可通过改变模型进行int8及batch_size=4推理。

执行完成后，会将预测结果保存在`results/resnet_fp32_b1.bmodel_img_opencv_cpp_result.txt`下，同时会打印预测结果、推理时间等信息。

```bash
......
ILSVRC2012_val_00049892.JPEG pred: 412, score:0.224830
ILSVRC2012_val_00049905.JPEG pred: 116, score:0.405769
ILSVRC2012_val_00049909.JPEG pred: 416, score:0.477843
ILSVRC2012_val_00049948.JPEG pred: 330, score:0.411436
ILSVRC2012_val_00049956.JPEG pred: 239, score:0.479264
ILSVRC2012_val_00049964.JPEG pred: 929, score:0.398404
ILSVRC2012_val_00049992.JPEG pred: 357, score:0.444582
================
result saved in results/resnet_fp32_b1.bmodel_img_opencv_cpp_result.txt

############################
SUMMARY: resnet infer
############################
[      resnet overall]  loops:    1 avg: 17115114 us
[          read image]  loops:  100 avg: 2000 us
[               infer]  loops:  100 avg: 14962 us
[  resnet pre-process]  loops:  100 avg: 2575 us
[    resnet inference]  loops:  100 avg: 11954 us
[ resnet post-process]  loops:  100 avg: 93 us
```

## 2. arm SoC平台
## 2.1 环境准备
对于arm SoC平台，内部已经集成了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，位于`/opt/sophon/`下。
## 2.2 交叉编译
通常在x86主机上交叉编译程序，使之能够在SoC平台运行。您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../docs/Environment_Install_Guide.md#31-交叉编译环境搭建)。本例程主要依赖libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)和sophon-ffmpeg(>=0.2.4)运行库包

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件，resnet_opencv和resnet_bmcv编译方法相同，以编译resnet_opencv程序为例：
```bash
cd resnet_opencv
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make # 生成resnet_opencv.soc
```

## 2.3 测试命令
将生成的可执行文件及所需的模型和测试图片拷贝到SoC目标平台中测试，测试方法请参考x86 PCIe平台的[1.3 测试命令](#13-测试命令)。


## 3. 精度与性能测试
### 3.1 精度测试
本例程在`tools`目录下提供了`eval.py`脚本，可以将预测结果文件与测试集标签文件进行对比，计算出分类准确率。具体的测试命令如下：
```bash
# 请根据实际情况修改文件路径
python3 tools/eval.py --gt_path data/images/imagenet_val_1k/label.txt --pred_path cpp/resnet_opencv/build/results/resnet_fp32_b1.bmodel_img_opencv_cpp_result.txt
```
执行完成后，会打印出分类的准确率：
```bash
INFO:root:ACC: 80.10000%
```
### 3.2 性能测试

可以使用bmrt_test测试模型的理论性能：
```bash
bmrt_test --bmodel {path_of_bmodel}
```
也可以参考[1.2 测试命令](#12-测试命令)打印程序运行中的实际性能指标。  
测试中性能指标存在一定的波动属正常现象。

### 3.3 测试结果

在BM1684X PCIe上，不同例程、不同模型的精度和性能测试结果如下：

| 例程          | 精度 | batch_size | ACC(%) | bmrt_test(ms) | infer_time(ms) | QPS    |
| ------------- | ---- | ---------- | ------ | ------------- | -------------- | ------ |
| resnet_opencv | fp32 | 1          | 80.00  | 8.72          | 12.34          | 65.94  |
| resnet_opencv | fp32 | 4          | 80.00  | 7.85          | 10.67          | 77.34  |
| resnet_opencv | int8 | 1          | 78.20  | 0.73          | 2.33           | 239.46 |
| resnet_opencv | int8 | 4          | 78.80  | 0.45          | 1.64           | 324.86 |
| resnet_bmcv   | fp32 | 1          | 80.00  | 8.73          | 10.44          | 76.47  |
| resnet_bmcv   | fp32 | 4          | 80.00  | 7.84          | 8.68           | 89.35  |
| resnet_bmcv   | int8 | 1          | 78.60  | 0.72          | 2.08           | 227.84 |
| resnet_bmcv   | int8 | 4          | 79.30  | 0.44          | 1.21           | 281.05 |

在BM1684X SoC上，不同例程、不同模型的精度和性能测试结果如下：

| 例程          | 精度 | batch_size | ACC(%) | bmrt_test(ms) | infer_time(ms) | QPS    |
| ------------- | ---- | ---------- | ------ | ------------- | -------------- | ------ |
| resnet_opencv | fp32 | 1          | 80.00  | 8.64          | 14.84          | 61.81  |
| resnet_opencv | fp32 | 4          | 80.00  | 7.84          | 14.00          | 66.28  |
| resnet_opencv | int8 | 1          | 78.20  | 0.74          | 6.95           | 120.64 |
| resnet_opencv | int8 | 4          | 78.80  | 0.45          | 6.57           | 130.24 |
| resnet_bmcv   | fp32 | 1          | 80.00  | 8.64          | 9.05           | 99.09  |
| resnet_bmcv   | fp32 | 4          | 80.00  | 7.84          | 8.23           | 108.16 |
| resnet_bmcv   | int8 | 1          | 78.60  | 0.72          | 1.13           | 466.85 |
| resnet_bmcv   | int8 | 4          | 79.30  | 0.45          | 0.84           | 541.63 |

在BM1684 PCIe上，不同例程、不同模型的精度和性能测试结果如下：
|例程|精度|batch_size|ACC(%)|bmrt_test(ms)|infer_time(ms)|QPS|
|--------|-----|-----|-----|-----|-----|----|
|resnet_opencv|fp32|1|80.20|6.52|8.26|98.24|
|resnet_opencv|fp32|4|80.20|5.20|6.79|119.06|
|resnet_opencv|int8|1|78.60|3.63|5.63|137.71|
|resnet_opencv|int8|4|79.20|1.12|3.13|226.51|
|resnet_bmcv|fp32|1|79.90|6.49|7.45|111.16|
|resnet_bmcv|fp32|4|79.90|5.48|6.09|133.42|
|resnet_bmcv|int8|1|79.00|3.64|4.51|166.94|
|resnet_bmcv|int8|4|79.50|1.12|1.99|295.64|

在BM1684 Soc上，不同例程、不同模型的精度和性能测试结果如下：

| 例程          | 精度 | batch_size | ACC(%) | bmrt_test(ms) | infer_time(ms) | QPS    |
| ------------- | ---- | ---------- | ------ | ------------- | -------------- | ------ |
| resnet_opencv | fp32 | 1          | 80.20  | 6.52          | 13.04          | 68.37  |
| resnet_opencv | fp32 | 4          | 80.20  | 5.20          | 11.72          | 76.86  |
| resnet_opencv | int8 | 1          | 78.60  | 3.65          | 10.19          | 85.06  |
| resnet_opencv | int8 | 4          | 79.20  | 1.12          | 7.71           | 111.13 |
| resnet_bmcv   | fp32 | 1          | 79.90  | 6.51          | 7.05           | 120.17 |
| resnet_bmcv   | fp32 | 4          | 79.90  | 5.20          | 5.70           | 144.84 |
| resnet_bmcv   | int8 | 1          | 79.00  | 3.64          | 4.20           | 182.99 |
| resnet_bmcv   | int8 | 4          | 79.50  | 1.12          | 1.63           | 352.83 |


```
bmrt_test: 使用bmrt_test计算出来的每张图的理论推理时间；
infer_time: 程序运行时每张图的实际推理时间；
QPS: 程序每秒钟全流程处理的图片数。
```

> **测试说明**：  
1. 性能测试的结果具有一定的波动性。
2. INT8模型精度测试结果具有一定的波动性
3. 部分指标暂时缺失，后续更新
