# LPRNet Python例程

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | lprnet_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |
| 2    | lprnet_bmcv.py   | 使用SAIL解码、BMCV前处理、SAIL推理     |

## 1. x86 PCIe平台
## 1.1 环境准备
如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail,具体请参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。
此外您可能还需要安装其他第三方库：
```bash
pip3 install -r requirements.txt
```

## 1.2 测试命令
python例程不需要编译，可以直接运行。lprnet_opencv.py和lprnet_bmcv.py的命令参数相同，以lprnet_opencv.py的推理为例，参数说明如下：

```bash
usage:lprnet_opencv.py [--input_path IMG_PATH] [--bmodel BMODEL] [--tpu_id TPU]
--input_path:推理图片路径，可输入整个图片文件夹的路径；
--bmodel:用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--tpu_id:用于推理的tpu设备id。
```

测试实例如下，请根据目标平台、模型精度、batch_size选择相应的bmodel：
```bash
# 测试整个文件夹
python3 lprnet_opencv.py --input_path ../data/images/test --bmodel ../data/models/BM1684X/lprnet_fp32_1b.bmodel --tpu_id 0
```

执行完成后，会将预测结果保存在`results/lprnet_fp32_1b.bmodel_test_opencv_python_result.json`下，同时会打印预测结果、推理时间等信息。

```bash
......
INFO:root:filename: 豫RM6396.jpg, res: 皖RM6396
INFO:root:filename: 皖S08407.jpg, res: 皖S08407
INFO:root:filename: 皖SYZ927.jpg, res: 皖SYZ927
INFO:root:filename: 皖SZ788K.jpg, res: 皖SZ788K
INFO:root:filename: 皖SZH382.jpg, res: 皖SZH382
INFO:root:filename: 川X90621.jpg, res: 川X90621
INFO:root:result saved in ./results/lprnet_fp32_1b.bmodel_test_opencv_python_result.json
INFO:root:------------------ Inference Time Info ----------------------
INFO:root:inference_time(ms): 1.21
INFO:root:total_time(ms): 2228.84, img_num: 1000
INFO:root:average latency time(ms): 2.23, QPS: 448.663952
```


## 2. SoC平台
## 2.1 环境准备
如果您使用SoC平台测试本例程，您需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../docs/Environment_Install_Guide.md#32-交叉编译安装sophon-sail)。
此外您可能还需要安装其他第三方库：
```bash
pip3 install -r requirements.txt
```
## 2.2 测试命令
SoC平台的测试方法与x86 PCIe平台相同，请参考[1.2 测试命令](#12-测试命令)。

## 3. 精度与性能测试
### 3.1 精度测试
本例程在`tools`目录下提供了`eval.py`脚本，可以将前面生成的预测结果json文件与测试集标签json文件进行对比，计算出车牌识别准确率。具体的测试命令如下：
```bash
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval.py --label_json data/images/test_label.json --result_json python/results/lprnet_fp32_1b.bmodel_test_opencv_python_result.json
```
执行完成后，会打印出车牌识别的准确率：
```bash
INFO:root:ACC = 894/1000 = 0.894
```
### 3.2 性能测试

可以使用bmrt_test测试模型的理论性能：
```bash
bmrt_test --bmodel {path_of_bmodel}
```
也可以参考[1.2 测试命令](#12-测试命令)打印程序运行中的实际性能指标。  
测试中性能指标存在一定的波动属正常现象。

### 3.3 测试结果

[LNRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)中模型在该测试集上的准确率为89.4%。

在BM1684X SoC上，不同例程、不同模型的精度和性能测试结果如下：
|       例程    | 精度 |batch_size|  ACC  |bmrt_test|infer_time| QPS |
|   ----------  | ---- | -------  | ----- |  -----  | -----    | --- |
| lprnet_opencv | fp32 |   1      | 89.4% |  0.8ms  |  1.2ms   | 450 |
| lprnet_opencv | fp32 |   4      | 90.1% |  0.7ms  |  0.9ms   | 650 |
| lprnet_opencv | int8 |   1      | 88.7% |  0.3ms  |  0.7ms   | 580 |
| lprnet_opencv | int8 |   4      | 89.8% |  0.2ms  |  0.4ms   | 1000 |
| lprnet_bmcv   | fp32 |   1      | 87.9% |  0.8ms  |  0.9ms   | 420 |
| lprnet_bmcv   | fp32 |   4      | 88.2% |  0.7ms  |  0.8ms   | 530 |
| lprnet_bmcv   | int8 |   1      | 87.6% |  0.3ms  |  0.3ms   | 580 |
| lprnet_bmcv   | int8 |   4      | 87.8% |  0.2ms  |  0.2ms   | 750 |

在BM1684 SoC上，不同例程、不同模型的精度和性能测试结果如下：
|       例程    | 精度 |batch_size|  ACC  |bmrt_test|infer_time| QPS |
|   ----------  | ---- | -------  | ----- |  -----  | -----    | --- |
| lprnet_opencv | fp32 |   1      | 89.4% |  1.7ms  |  2.1ms   | 320 |
| lprnet_opencv | fp32 |   4      | 90.1% |  0.9ms  |  1.1ms   | 600 |
| lprnet_opencv | int8 |   1      | 88.7% |  0.7ms  |  1.1ms   | 480 |
| lprnet_opencv | int8 |   4      | 89.8% |  0.3ms  |  0.4ms   | 950 |
| lprnet_bmcv   | fp32 |   1      | 88.2% |  1.7ms  |  1.8ms   | 300 |
| lprnet_bmcv   | fp32 |   4      | 88.2% |  0.9ms  |  0.9ms   | 480 |
| lprnet_bmcv   | int8 |   1      | 87.1% |  0.7ms  |  0.8ms   | 430 |
| lprnet_bmcv   | int8 |   4      | 87.8% |  0.3ms  |  0.3ms   | 680 |

```
bmrt_test: 每张图片的理论推理时间；
infer_time: 程序运行时每张图片的网络推理时间；
QPS: 程序每秒钟全流程处理的图片数。
```

> **测试说明**：  
1. PCIe上的测试精度与SoC相同，性能由于CPU的不同可能存在一定差异；
2. LPRNet网络中包含mean算子，会把所有batch数据加和求平均，当多batch推理时，同一张图片在不同的batch组合中可能会有不同的推理结果;
3. 性能测试的结果具有一定的波动性。