
## C3D Python例程

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | c3d_opencv.py | 使用OPENCV解码、OPENCV前处理、SAIL推理 |

## 1. x86 PCIe平台
### 1.1 环境准备
如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)、sophon-ffmpeg(>=0.2.4)和sophon-sail(>=3.1.0),具体请参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。
此外您可能还需要安装其他第三方库：
```
pip3 install -r requirements.txt
```
### 1.2 测试命令
python例程不需要编译，可以直接运行。参数说明如下:
```shell
# --bmodel: bmodel path.
# --input_path:  input dataset path.
# --tpu_id: device id.
```
测试实例如下：
```
python3 c3d_opencv.py --bmodel ../data/models/BM1684X/c3d_fp32_1b.bmodel --input_path ../data/UCF_test_01
```
执行完成后，打印结果信息如下：
```
total_time(ms): 501262.83, frame_num: 58820
avg_infer_time(ms): 79.69
ACC:  0.7153558052434457
```
## 2. arm SoC平台
### 2.1 环境准备
如果您使用SoC平台测试本例程，您需要交叉编译安装sophon-sail(>=3.1.0)，具体可参考[交叉编译安装sophon-sail](../../docs/Environment_Install_Guide.md#32-交叉编译安装sophon-sail)。
此外您可能还需要安装其他第三方库：
```
pip3 install -r requirements.txt
```
### 2.2 测试命令
SoC平台的测试方法与x86 PCIe平台相同，请参考[1.2 测试命令](#12-测试命令)。

## 3. 精度与性能测试结果

经本地编译测试，**使用与本例程同样的预处理方法**，[MMAction C3D](https://mmaction2.readthedocs.io/zh_CN/latest/recognition_models.html#ucf-101)中提供的pytorch模型在本地UCF101测试集上的top-1 acc为71.5356。

结果打印信息如下，在其中可以获取到acc等信息。
```bash
total_time(ms): 501262.83, frame_num: 58820
avg_infer_time(ms): 79.69 
ACC:  0.7153558052434457 #acc
```

在BM1684X PCIE上，不同例程、不同模型的精度和性能测试结果如下：

|   例程      | 精度 |batch_size|  acc  |infer_time|
|   -------- | ---- | ------- | ----- |-----    |
| c3d_opencv   | fp32 |   1    | 71.54% |79.7ms   |
| c3d_opencv   | fp32 |   4    | 71.54% |296.5ms |
| c3d_opencv   | int8 |   1    | 71.54% |13.9ms   |
| c3d_opencv   | int8 |   4    | 71.54% |42.0ms |

在BM1684 PCIE上，不同例程、不同模型的精度和性能测试结果如下：

|   例程      | 精度 |batch_size|  acc  |infer_time|
|   -------- | ---- | ------- | ----- |-----    |
| c3d_opencv   | fp32 |   1    | 71.54% | 57.0ms   |
| c3d_opencv   | fp32 |   4    | 71.54% | 191.7ms |
| c3d_opencv   | int8 |   1    | 69.10% | 44.4ms   |
| c3d_opencv   | int8 |   4    | 69.10% | 74.2ms |

**注:**

1.相同版本驱动下，同一例程、同一模型在soc与pcie上的infer_time误差不超过20%。
2.由于视频解码差异，同一例程、同一模型在soc与pcie上的acc可能会有误差，一般不超过1%。