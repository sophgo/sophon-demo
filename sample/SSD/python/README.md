## SSD Python例程

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | ssd_bmcv.py | 使用SAIL解码、BMCV前处理、SAIL推理 |

## 1. x86 PCIe平台
## 1.1 环境准备
如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)、sophon-ffmpeg(>=0.2.4)和sophon-sail(>=3.1.0),具体请参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。
此外您可能还需要安装其他第三方库：
```
pip3 install -r requirements.txt
```
## 1.2 测试命令
python例程不需要编译，可以直接运行。参数说明如下:
```shell
# --bmodel: bmodel path, can be fp32 or int8 model with batch_size = 1 or 4.
# --input_path:  input path, can be image directory or video file.
# --conf_thre:  confidence threshold, default: 0.
# --tpu_id: default: 0.
# --results_directory: default: results.
```
测试实例如下：
```
python3 ssd_bmcv.py --bmodel ../data/models/BM1684X/ssd300_fp32_4b.bmodel --input_path ../data/VOC2007-test-images
```
执行完成后，会将预测结果保存在同目录的results/中，打印日志信息如下：
```
read image:  000000.jpg
..........
read image:  009963.jpg
total_time(ms): 451671.91, img_num: 4952
avg_infer_time(ms): 255.88
```
## 2. arm SoC平台
## 2.1 环境准备
如果您使用SoC平台测试本例程，您需要交叉编译安装sophon-sail(>=3.1.0)，具体可参考[交叉编译安装sophon-sail](../../docs/Environment_Install_Guide.md#32-交叉编译安装sophon-sail)。
## 2.2 测试命令
SoC平台的测试方法与x86 PCIe平台相同，请参考[1.2 测试命令](#12-测试命令)。


### 3.3 测试结果

经本地编译测试，[caffe at SSD](https://github.com/weiliu89/caffe/tree/ssd)中VOC07+12模型在VOC2007-test数据集上的mAP为**71.7%**。

在BM1684X PCIE上，不同模型的精度和性能测试结果如下：

|   例程      | 精度 |batch_size|  mAP   |infer_time|
|   -------- | ---- | ------- | -----  |-----    |
| ssd_bmcv   | fp32 |   1      | 71.5% |66.1ms   |
| ssd_bmcv   | fp32 |   4      | 71.5% |255.8ms |
| ssd_bmcv   | int8 |   1      | 71.1% |5.5ms    |
| ssd_bmcv   | int8 |   4      | 71.1% |22.5ms   |

在BM1684 PCIE上，不同模型的精度和性能测试结果如下：
| ssd_bmcv   | fp32 |   1      | 71.5% |38.5ms   |
| ssd_bmcv   | fp32 |   4      | 71.5% |184.9ms |
| ssd_bmcv   | int8 |   1      | 71.1% |20.1ms    |
| ssd_bmcv   | int8 |   4      | 71.1% |73.2ms   |