
# C++例程
cpp目录下提供了一系列C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | c3d_opencv | 使用OpenCV解码、OpenCV前处理、BMRT推理 |


## 1. x86 PCIe 平台

### 1.1 环境准备

如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)和sophon-ffmpeg(>=0.2.4),具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。

### 1.2 程序编译
C++程序需要编译可执行文件，以编译c3d_opencv程序为例：
```bash
cd c3d_opencv
mkdir build && cd build
cmake .. && make
```

### 1.3 测试命令

编译完成后，会生成c3d_opencv.pcie，具体参数说明如下：

```bash
usage:./c3d_opencv.pcie <dataset path> <bmodel path> <device id(default: 0)>
```

测试实例如下：

```bash
./c3d_opencv.pcie ../../../data/UCF_test_01 ../../../data/models/BM1684X/c3d_fp32_1b.bmodel 0
```

可通过改变模型进行b4推理。

执行完成后，会打印推理时间、准确率等信息。

```bash
========================================
acc now: 0.715356
========================================

############################
SUMMARY: C3D detect
############################
[         C3D overall]  loops:    1 avg: 614241420 us
[      C3D preprocess]  loops:  100 avg: 89400 us
[       C3D inference]  loops:  100 avg: 75612 us 
```

## 2. arm SoC平台
### 2.1 环境准备
对于arm SoC平台，内部已经集成了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，位于`/opt/sophon/`下。
### 2.2 交叉编译
通常在x86主机上交叉编译程序，使之能够在arm SoC平台运行。您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../docs/Environment_Install_Guide.md#31-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件，以编译c3d_opencv程序为例：
```bash
cd c3d_opencv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make # 生成c3d_opencv.soc
```

### 2.3 测试命令
将生成的可执行文件及所需的模型和测试图片拷贝到SoC目标平台中测试，测试方法请参考x86 PCIe平台的[1.3 测试命令](#13-测试命令)。

## 3. 精度与性能测试结果

经本地编译测试，**使用与本例程同样的预处理方法**，[MMAction C3D](https://mmaction2.readthedocs.io/zh_CN/latest/recognition_models.html#ucf-101)中提供的pytorch模型在本地UCF101测试集上的top-1 acc为71.5356。

结果打印信息如下，在其中可以获取到acc、infer_time等信息。
```bash
========================================
acc now: 0.715356 #acc
========================================

############################
SUMMARY: C3D detect
############################
[         C3D overall]  loops:    1 avg: 614241420 us
[      C3D preprocess]  loops:  100 avg: 89400 us
[       C3D inference]  loops:  100 avg: 75612 us #infer_time 
```


在BM1684X PCIE上，不同例程、不同模型的精度和性能测试结果如下：

|   例程      | 精度 |batch_size|  acc  |infer_time|
|   -------- | ---- | ------- | ----- |-----    |
| c3d_opencv   | fp32 |   1    | 71.54% |75.6ms   |
| c3d_opencv   | fp32 |   4    | 71.54% |281.4ms |
| c3d_opencv   | int8 |   1    | 71.54% |10.1ms   |
| c3d_opencv   | int8 |   4    | 71.54% |28.0ms |

在BM1684 PCIE上，不同例程、不同模型的精度和性能测试结果如下：

|   例程      | 精度 |batch_size|  acc  |infer_time|
|   -------- | ---- | ------- | ----- |-----    |
| c3d_opencv   | fp32 |   1    | 71.54% | 55.4ms   |
| c3d_opencv   | fp32 |   4    | 71.54% | 186.7ms |
| c3d_opencv   | int8 |   1    | 69.10% | 42.8ms   |
| c3d_opencv   | int8 |   4    | 69.10% | 69.2ms |

**注:**

1.相同版本驱动下，同一例程、同一模型在soc与pcie上的infer_time误差不超过20%。
2.由于视频解码差异，同一例程、同一模型在soc与pcie上的acc可能会有误差，一般不超过1%。