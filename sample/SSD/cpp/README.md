# C++例程
cpp目录下提供了一系列C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | ssd_bmcv   | 使用OpenCV解码、BMCV前处理、BMRT推理   |
| 2    | ssd_opencv | 使用OpenCV解码、OpenCV前处理、BMRT推理 |


## 1. x86 PCIe 平台

### 1.1 环境准备

如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)和sophon-ffmpeg(>=0.2.4),具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。

### 1.2 程序编译
C++程序需要编译可执行文件，ssd_opencv和ssd_bmcv编译方法相同，以编译ssd_bmcv程序为例：
```bash
cd ssd_bmcv
#change macro 'DEBUG' in ssd.hpp, when DEBUG=0, test whole dataset.
mkdir build && cd build
cmake .. && make
```

### 1.3 测试命令

编译完成后，会生成ssd_bmcv.pcie，具体参数说明如下：

```bash
usage:./ssd_bmcv.pcie <image directory or video path> <bmodel path> <device id(default: 0)> <conf_thre(default: unset)> <nms_thre(default: unset)>
```

测试实例如下：

```bash
./ssd_bmcv.pcie ../../../data/VOC2007-test-images/ ../../../data/models/BM1684X/ssd300_fp32_1b.bmodel 0
```

可通过改变模型进行b4或int8推理。

执行完成后，会将预测结果图片保存在`results/`下，预测结果信息保存在bmcv_cpp_result_b1.json中，同时会打印预测结果、推理时间等信息。

```bash
########################

Start SSD inference, total image num: 4952
Read image path: ../../../data/VOC2007-test-images//000001.jpg
Open /dev/bm-sophon0 successfully, device index = 0, jpu fd = 14, vpp fd = 14
Output shape infos: 1 1 15 7
width: 353 height: 500
image 0 :
   class: 12; score: 0.838305; box:[50.7866, 240.05, 201.803, 374.777]
   class: 15; score: 0.986782; box:[12.3152, 6.01171, 354.142, 492.357]
================================================
result saved in ./bmcv_cpp_result_b1.json
================================================

############################
SUMMARY: SSD detect
############################
[ Start SSD inference]  loops:    2 avg: 4291 us
[Start SSD preprocess]  loops:    1 avg: 1316 us
[Start SSD postprocess]  loops:    1 avg: 3683 us
```

## 2. arm SoC平台
### 2.1 环境准备
对于arm SoC平台，内部已经集成了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，位于`/opt/sophon/`下。
### 2.2 交叉编译
通常在x86主机上交叉编译程序，使之能够在arm SoC平台运行。您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../docs/Environment_Install_Guide.md#31-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件，以编译ssd_bmcv程序为例：
```bash
cd ssd_bmcv
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make # 生成ssd_bmcv.soc
```

### 2.3 测试命令
将生成的可执行文件及所需的模型和测试图片拷贝到SoC目标平台中测试，测试方法请参考x86 PCIe平台的[1.3 测试命令](#13-测试命令)。

## 3. 精度与性能测试

### 3.1 精度测试
本例程在`SSD/tools`目录下提供了`eval.py`脚本，以计算推理结果的mAP。具体的测试命令如下：
```bash
# 请根据实际情况修改 --ground_truths 和 --result_json参数
python3 eval.py --result_json ../cpp/ssd_bmcv/build/bmcv_cpp_result_b1.json
```
执行完成后，会打印出mAP信息：
```bash
...
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.717 #mAP
...
```

### 3.2 性能测试

可以使用bmrt_test测试模型的理论性能：
```bash
bmrt_test --bmodel {path_of_bmodel}
```
也可以参考[1.3 测试命令](#13-测试命令)打印程序运行中的实际性能指标。  
测试中性能指标存在一定的波动属正常现象。

### 3.3 测试结果

经本地编译测试，[caffe at SSD](https://github.com/weiliu89/caffe/tree/ssd)中VOC07+12模型在VOC2007-test数据集上的mAP为**71.7%**。

在BM1684X PCIE上，不同例程、不同模型的精度和性能测试结果如下：

|   例程      | 精度 |batch_size|  mAP  |preprocess_time |infer_time|
|   -------- | ---- | ------- | ----- |-----------    |-----    |
| ssd_bmcv   | fp32 |   1      | 71.7% |   1.5ms    |65.7ms   |
| ssd_bmcv   | fp32 |   4      | 71.7% |   5.3ms    |255.1ms |
| ssd_bmcv   | int8 |   1      | 71.5% |   1.4ms    |5.3ms    |
| ssd_bmcv   | int8 |   4      | 71.5% |   5.3ms    |20.1ms   |
| ssd_opencv   | fp32 |   1    | 71.8% |  4.8ms    |66.0ms   |
| ssd_opencv   | fp32 |   4    | 71.8% | 15.5ms    |255.1ms |
| ssd_opencv   | int8 |   1    | 71.5% |  3.4ms    |5.4ms    |
| ssd_opencv   | int8 |   4    | 71.5% | 13.8ms     |20.0ms   |

在BM1684 PCIE上，不同例程、不同模型的精度和性能测试结果如下：

|     例程      | 精度 |batch_size|  mAP  |preprocess_time |infer_time|
|   --------    | ---- | -------  | ----- | ------------| -----    |
| ssd_bmcv   | fp32 |   1      | 71.7% |    2.7ms    |38.2ms   |
| ssd_bmcv   | fp32 |   4      | 71.7% |    10.3ms      |180.5ms |
| ssd_bmcv   | int8 |   1      | 71.5% |    2.7ms      |21.8ms    |
| ssd_bmcv   | int8 |   4      | 71.5% |    10.4ms      |70.1ms   |
| ssd_opencv | fp32 |   1      | 71.7% |    4.1ms    |38.2ms   |
| ssd_opencv   | fp32 |   4      | 71.7% |    10.5ms      |180.5ms |
| ssd_opencv   | int8 |   1      | 71.5% |    3.7ms     |21.8ms    |
| ssd_opencv   | int8 |   4      | 71.5% |    8.1ms      |72.0ms   |

**注:**

1.同一例程、同一模型在soc与pcie上的infer_time相近，mAP相同。

2.ssd_opencv 的 preprocess_time 一定程度上取决于主机硬件。
