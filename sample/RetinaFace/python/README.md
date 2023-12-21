:# Retinaface Python例程

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程            | 说明                        |
| ----   | ----------------     | --------------------------- |
| 1      | retinaface_opencv.py | 使用OpenCV前处理、SAIL推理   |
| 2      | retinaface_bmcv.py   | 使用BMCV前处理、SAIL推理     |

## 1. x86 PCIe平台
## 1.1 环境准备
如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)、sophon-ffmpeg(>=0.2.4)和sophon-sail(>=3.1.0),具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)。
此外您可能还需要安装其他第三方库：
```bash
$ cd python
$ pip3 install -r requirements.txt
```

```
注意：若报错ImportError: libGL.so.1: cannot open shared object file: No such file or directory，需要执行命令sudo apt-get install libgl1
```

## 1.2 测试命令
retinaface_opencv.py和retinaface_bmcv.py的命令参数相同，以retinaface_opencv.py的推理为例，参数说明如下：

```bash
usage:retinaface_opencv.py [--bmodel BMODEL] [--network NETWORK] [--input_path INPUT][--tpu_id TPU] [--conf CONF] [--nms NMS] [--use_np_file_as_input False]
--bmodel:用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--network：backbone,可选择mobile0.25或者resnet50,默认为mobile0.25；
--input_path: 测试图片路径，可输入单张图片或视频，也可输入图片文件夹路径；
--tpu_id:用于推理的tpu设备id;
--conf: 置信度阈值，默认为0.02
--nms：nms阈值，默认为0.3
--use_np_file_as_input：是否使用其他数据作为输入
```

测试实例如下：

```bash
# 以测试WIDERVAL数据集为例
# 使用1batch bmodel
$ python3 retinaface_opencv.py --bmodel ../data/models/BM1684X/retinaface_mobilenet0.25_fp32_1b.bmodel --network mobile0.25 --input_path ../data/images/WIDERVAL --tpu_id 0 --conf 0.02 --nms 0.4 --use_np_file_as_input False
```
执行完成后，会将预测结果保存在`results/retinaface_mobilenet0.25_fp32_1b.bmodel_opencv_WIDERVAL_python_result.txt`文件中，将预测图片保存在`results/retinaface_mobilenet0.25_fp32_1b.bmodel_opencv_WIDERVAL_python_result/`文件夹下，同时会打印预测结果、推理时间、函数运行时间等信息。

如下图所示：
``` bash
- face 1: x,y,w,h,conf = 130 133 67 97 0.99906695
- face 2: x,y,w,h,conf = 685 165 63 82 0.9955299
- face 3: x,y,w,h,conf = 365 154 60 80 0.9952643
- face 4: x,y,w,h,conf = 418 418 50 54 0.027065556
- face 5: x,y,w,h,conf = 996 717 16 21 0.020527808
- face 6: x,y,w,h,conf = 737 429 84 121 0.020261558
- ------------------ Inference Time Info ----------------------
- inference_time(ms): 6.50
- total_time(ms): 68396.19, img_num: 3226
- average latency time(ms): 6.50, QPS: 153.963054
===================================================
+----------------------------------------------------------------------------------------+
|                               Running Time Cost Summary                                |
+------------------------+----------+----------------------+--------------+--------------+
|        函数名称        | 运行次数 |     平均耗时(秒)     | 最大耗时(秒) | 最小耗时(秒) |
+------------------------+----------+----------------------+--------------+--------------+
| preprocess_with_opencv |   3226   | 0.007066955982641045 |    0.012     |    0.007     |
|      infer_numpy       |   3226   | 0.007012089274643524 |     0.01     |    0.007     |
|   postprocess_batch    |   3226   | 0.006522628642281463 |    0.074     |    0.004     |
+------------------------+----------+----------------------+--------------+--------------+
```
可通过改变模型进行batch_size=4模型的推理测试。

对于face和视频数据，使用如下命令进行测试。
```bash
# 以使用fp32 1batch bmodel测试为例，不同bmodel使用相同参数
$ python3 retinaface_opencv.py --bmodel ../data/models/BM1684X/retinaface_mobilenet0.25_fp32_1b.bmodel --network mobile0.25 --input_path ../data/images/face --tpu_id 0 --conf 0.02 --nms 0.5 --use_np_file_as_input False
```
预测图片会保存在`results/retinaface_mobilenet0.25_fp32_1b.bmodel_opencv_face_python_result/`文件夹下，同时会打印预测结果、推理时间、函数运行时间等信息。

```bash
# 以使用fp32 1batch bmodel测试为例，不同bmodel使用相同参数
python3 retinaface_opencv.py --bmodel ../data/models/BM1684X/retinaface_mobilenet0.25_fp32_1b.bmodel --network mobile0.25 --input_path ../data/videos/station.avi --tpu_id 0 --conf 0.02 --nms 0.5 --use_np_file_as_input False
```
预测图片会保存在`results/`文件夹下，同时会打印预测结果、推理时间、函数运行时间等信息。

## 2. arm SoC平台
### 2.1 环境准备
如果您使用SoC平台测试本例程，您需要交叉编译安装sophon-sail(>=3.1.0)，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。
此外您可能还需要安装其他第三方库
```bash
$ cd python
$ pip3 install -r requirements.txt
```
### 2.2 测试命令
Soc平台的测试方法与x86 PCIe平台相同，请参考1.2测试命令。
