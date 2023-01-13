# Python例程

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | resnet_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |
| 2    | resnet_bmcv.py   | 使用SAIL解码、BMCV前处理、SAIL推理     |

## 1. x86 PCIe平台
## 1.1 环境准备
如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)、sophon-ffmpeg(>=0.2.4)和sophon-sail(>=3.1.0),具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。

## 1.2 测试命令
resnet_opencv.py和resnet_bmcv.py的命令参数相同，以resnet_opencv.py的推理为例，参数说明如下：

```bash
usage:resnet_opencv.py [--input_path IMG_PATH] [--bmodel BMODEL] [--tpu_id TPU]
--input_path:推理图片路径，可输入整个图片文件夹的路径；
--bmodel:用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--tpu_id:用于推理的tpu设备id。
```

测试实例如下：
```bash
# 测试整个文件夹
python3 resnet_opencv.py --input_path ../data/images/imagenet_val_1k/img --bmodel ../data/models/BM1684X/resnet_fp32_b1.bmodel --tpu_id 0
```

执行完成后，会将预测结果保存在`results/resnet_fp32_b1.bmodel_img_opencv_python_result.txt`下，同时会打印预测结果、推理时间等信息。

```bash
......
INFO:root:filename: ../data/images/imagenet_val_1k/img/ILSVRC2012_val_00038219.JPEG, res: (419, 0.15843831)
INFO:root:filename: ../data/images/imagenet_val_1k/img/ILSVRC2012_val_00041825.JPEG, res: (788, 0.41158476)
INFO:root:filename: ../data/images/imagenet_val_1k/img/ILSVRC2012_val_00041938.JPEG, res: (849, 0.38458076)
INFO:root:filename: ../data/images/imagenet_val_1k/img/ILSVRC2012_val_00017071.JPEG, res: (933, 0.45050952)
INFO:root:filename: ../data/images/imagenet_val_1k/img/ILSVRC2012_val_00043924.JPEG, res: (343, 0.37661916)
INFO:root:filename: ../data/images/imagenet_val_1k/img/ILSVRC2012_val_00033817.JPEG, res: (77, 0.18264356)
INFO:root:result saved in ./results/resnet_fp32_b1.bmodel_img_opencv_python_result.txt
INFO:root:------------------ Inference Time Info ----------------------
INFO:root:inference_time(ms): 12.81
INFO:root:total_time(ms): 21841.05, img_num: 1000
INFO:root:average latency time(ms): 21.84, QPS: 45.785344
```

可通过改变模型进行int8及batch_size=4的推理测试。
> **注意**：`resnet_bmcv.py`暂不支持多batch模型的推理测试。


## 2. arm SoC平台
## 2.1 环境准备

如果您使用SoC平台测试本例程，您需要交叉编译安装sophon-sail(>=3.1.0)，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#32-交叉编译安装sophon-sail)。
此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt
```

## 2.2 测试命令

将生成的可执行文件及所需的模型和测试图片拷贝到SoC目标平台中测试，测试方法请参考x86 PCIe平台的[1.2 测试命令](#12-测试命令)。

## 3. 精度与性能测试
### 3.1 精度测试
本例程在`tools`目录下提供了`eval.py`脚本，可以将预测结果文件与测试集标签文件进行对比，计算出分类准确率。具体的测试命令如下：
```bash
# 请根据实际情况修改文件路径
python3 tools/eval.py --gt_path data/images/imagenet_val_1k/label.txt --pred_path python/results/resnet_fp32_b1.bmodel_img_opencv_python_result.txt
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
| resnet_opencv | fp32 | 1          | 80.10  | 8.72          | 9.92           | 47.27  |
| resnet_opencv | fp32 | 4          | 80.10  | 7.86          | 8.43           | 61.80  |
| resnet_opencv | int8 | 1          | 78.20  | 0.71          | 1.27           | 165.29 |
| resnet_opencv | int8 | 4          | 79.40  | 0.45          | 0.95           | 195.65 |
| resnet_bmcv   | fp32 | 1          | 78.30  | 8.58          | 8.87           | 76.59  |
| resnet_bmcv   | fp32 | 4          | 78.30  | 7.85          | 7.89           | 86.32  |
| resnet_bmcv   | int8 | 1          | 77.50  | 0.69          | 0.91           | 210.59 |
| resnet_bmcv   | int8 | 4          | 77.80  | 0.45          | 0.48           | 270.40 |

在BM1684X SoC上，不同例程、不同模型的精度和性能测试结果如下：

| 例程          | 精度 | batch_size | ACC(%) | bmrt_test(ms) | infer_time(ms) | QPS    |
| ------------- | ---- | ---------- | ------ | ------------- | -------------- | ------ |
| resnet_opencv | fp32 | 1          | 80.10  | 8.64          | 11.18          | 32.15  |
| resnet_opencv | fp32 | 4          | 80.10  | 7.84          | 10.07          | 35.72  |
| resnet_opencv | int8 | 1          | 78.20  | 0.73          | 3.11           | 46.38  |
| resnet_opencv | int8 | 4          | 79.40  | 0.45          | 2.64           | 48.53  |
| resnet_bmcv   | fp32 | 1          | 78.30  | 8.65          | 8.74           | 82.59  |
| resnet_bmcv   | fp32 | 4          | 78.30  | 7.84          | 7.86           | 94.25  |
| resnet_bmcv   | int8 | 1          | 77.50  | 0.74          | 0.83           | 237.65 |
| resnet_bmcv   | int8 | 4          | 77.80  | 0.45          | 0.47           | 309.94 |

在BM1684 PCIe上，不同例程、不同模型的精度和性能测试结果如下：

| 例程          | 精度 | batch_size | ACC(%) | bmrt_test(ms) | infer_time(ms) | QPS    |
| ------------- | ---- | ---------- | ------ | ------------- | -------------- | ------ |
| resnet_opencv | fp32 | 1          | 80.10  | 6.64          | 8.08           | 53.70  |
| resnet_opencv | fp32 | 4          | 80.10  | 5.23          | 6.35           | 68.82  |
| resnet_opencv | int8 | 1          | 78.20  | 3.76          | 4.91           | 76.36  |
| resnet_opencv | int8 | 4          | 79.40  | 1.14          | 1.90           | 145.11 |
| resnet_bmcv   | fp32 | 1          | 77.20  | 6.62          | 6.78           | 98.60  |
| resnet_bmcv   | fp32 | 4          | 77.20  | 5.22          | 5.26           | 121.25 |
| resnet_bmcv   | int8 | 1          | 75.50  | 3.77          | 3.86           | 138.89 |
| resnet_bmcv   | int8 | 4          | 76.90  | 1.15          | 1.17           | 240.95 |

在BM1684 SoC上，不同例程、不同模型的精度和性能测试结果如下：

| 例程          | 精度 | batch_size | ACC(%) | bmrt_test(ms) | infer_time(ms) | QPS    |
| ------------- | ---- | ---------- | ------ | ------------- | -------------- | ------ |
| resnet_opencv | fp32 | 1          | 80.10  | 6.54          | 8.94           | 36.23  |
| resnet_opencv | fp32 | 4          | 80.10  | 5.20          | 7.30           | 39.27  |
| resnet_opencv | int8 | 1          | 78.20  | 3.65          | 5.91           | 40.63  |
| resnet_opencv | int8 | 4          | 79.40  | 1.12          | 3.21           | 46.77  |
| resnet_bmcv   | fp32 | 1          | 77.20  | 6.54          | 6.61           | 91.82  |
| resnet_bmcv   | fp32 | 4          | 77.20  | 5.20          | 5.22           | 113.18 |
| resnet_bmcv   | int8 | 1          | 75.50  | 3.65          | 3.75           | 124.65 |
| resnet_bmcv   | int8 | 4          | 76.90  | 1.12          | 1.15           | 209.22 |

```
bmrt_test: 使用bmrt_test计算出来的每张图的理论推理时间；
infer_time: 程序运行时每张图的实际推理时间；
QPS: 程序每秒钟全流程处理的图片数。
```

> **测试说明**：  
1. 性能测试的结果具有一定的波动性。
2. INT8模型精度测试结果具有一定的波动性
3. 部分指标暂时缺失，后续更新

