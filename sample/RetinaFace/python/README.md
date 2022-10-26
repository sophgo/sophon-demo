:# Retinaface Python例程

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程            | 说明                        |
| ----   | ----------------     | --------------------------- |
| 1      | retinaface_opencv.py | 使用OpenCV前处理、SAIL推理   |
| 2      | retinaface_bmcv.py   | 使用BMCV前处理、SAIL推理     |

## 1. x86 PCIe平台
## 1.1 环境准备
如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)、sophon-ffmpeg(>=0.2.4)和sophon-sail(>=3.1.0),具体请参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。
此外您可能还需要安装其他第三方库：
```bash
$ cd python
$ pip3 install -r requirements.txt
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
# 以测试图片为例
# 使用1batch bmodel
$ python3 retinaface_opencv.py --bmodel ../data/models/BM1684X/retinaface_mobilenet0.25_fp32_1b.bmodel --network mobile0.25 --input_path ../data/images/face/face01.jpg --tpu_id 0 --conf 0.02 --nms 0.4 --use_np_file_as_input False
```
执行完成后，会将预测图片保存在`results/`文件夹下，同时会打印预测结果、推理时间等信息。
```bash
- face 1: x1, y1, x2, y2, conf = [1.4813636e+03 2.3813448e+02 1.5513214e+03 3.2603729e+02 9.9838036e-01]
- face 2: x1, y1, x2, y2, conf = [1.5934717e+03 2.1940230e+02 1.6646621e+03 3.0682361e+02 9.9629390e-01]
- face 3: x1, y1, x2, y2, conf = [1.3471459e+03 2.0241377e+02 1.4162469e+03 2.9239706e+02 9.9526298e-01]
- face 4: x1, y1, x2, y2, conf = [538.8302     306.1469     594.6965     377.89386      0.99249953]
- face 5: x1, y1, x2, y2, conf = [802.2228    221.92374   858.7444    292.33325     0.9888797]
- face 6: x1, y1, x2, y2, conf = [1.1095901e+03 2.1641072e+02 1.1663149e+03 2.8901352e+02 9.8742777e-01]
- face 7: x1, y1, x2, y2, conf = [1.0253922e+03 2.9218658e+02 1.0776205e+03 3.5883188e+02 9.8676348e-01]
- face 8: x1, y1, x2, y2, conf = [412.9569    289.28937   467.5013    357.1796      0.9867334]
- face 9: x1, y1, x2, y2, conf = [899.41113    292.2927     955.6922     363.80243      0.98394644]
- face 10: x1, y1, x2, y2, conf = [1.2130354e+03 2.0754848e+02 1.2684365e+03 2.7846011e+02 9.8286533e-01]
- face 11: x1, y1, x2, y2, conf = [1.2773765e+03 2.8552786e+02 1.3350741e+03 3.6016623e+02 9.7993690e-01]
- face 12: x1, y1, x2, y2, conf = [339.97354    187.02548    393.9029     253.09778      0.97491217]
- face 13: x1, y1, x2, y2, conf = [481.79596   199.71217   534.8542    265.61383     0.9736318]
- face 14: x1, y1, x2, y2, conf = [9.6184399e+02 2.0425456e+02 1.0173577e+03 2.7483292e+02 9.7248906e-01]
- face 15: x1, y1, x2, y2, conf = [1.1389028e+03 3.0996884e+02 1.1967733e+03 3.8043613e+02 9.5937258e-01]
- face 16: x1, y1, x2, y2, conf = [660.9378    198.53073   718.954     271.32382     0.9574182] 
- face 17: x1, y1, x2, y2, conf = [298.1008     256.23212    354.85028    325.1901       0.95197785]
- face 18: x1, y1, x2, y2, conf = [199.78685   253.32855   257.67377   326.9112      0.9292876]
- face 19: x1, y1, x2, y2, conf = [781.88477   348.93954   834.64703   417.14322     0.8764282]
- face 20: x1, y1, x2, y2, conf = [645.5316     299.6266     697.87836    367.65497      0.85817415]
+--------------------------------------------------------------------------------+
|                           Running Time Cost Summary                            |
+------------------------+----------+--------------+--------------+--------------+
|        函数名称        | 运行次数 | 平均耗时(秒) | 最大耗时(秒) | 最小耗时(秒) |
+------------------------+----------+--------------+--------------+--------------+
| preprocess_with_opencv |    1     |    0.009     |    0.009     |    0.009     |
|      infer_numpy       |    1     |    0.011     |    0.011     |    0.011     |
|      postprocess       |    1     |    0.007     |    0.007     |    0.007     |
+------------------------+----------+--------------+--------------+--------------+
```

```bash
# 以测试文件夹为例
$ python3 retinaface_opencv.py --bmodel ../data/models/BM1684X/retinaface_mobilenet0.25_fp32_4b.bmodel --network mobile0.25 --input_path ../data/images/WIDERVAL --tpu_id 0 --conf 0.02 --nms 0.4 --use_np_file_as_input False
```
执行完成后，会将预测结果保存在`results/retinaface_mobilenet0.25_fp32_1b.bmodel_opencv_WIDERVAL_python_result.txt`文件中，将预测图片保存在`results/retinaface_mobilenet0.25_fp32_1b.bmodel_opencv_WIDERVAL_python_result/`文件夹下，同时会打印预测结果、推理时间、函数运行时间等信息。

如下图所示：
``` bash
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

## 2. arm SoC平台
### 2.1 环境准备
如果您使用SoC平台测试本例程，您需要交叉编译安装sophon-sail(>=3.1.0)，具体可参考[交叉编译安装sophon-sail](../../docs/Environment_Install_Guide.md#32-交叉编译安装sophon-sail)。
此外您可能还需要安装其他第三方库
```bash
$ cd python
$ pip3 install -r requirements.txt
```
### 2.2 测试命令
Soc平台的测试方法与x86 PCIe平台相同，请参考1.2测试命令。

## 3. 精度与性能测试
### 3.1 精度测试
本例程在`tools`目录下提供了精度测试工具，可以将WIDERFACE测试集预测结果与ground truth进行对比，计算出人脸检测ap。具体的测试命令如下：
```bash
cd tools/widerface_evaluate
tar -zxvf widerface_txt.tar.gz
# 请根据实际情况，将1.2节生成的预测结果txt文件移动至当前文件夹，并将路径填入transfer.py, 并保证widerface_txt/的二级目录为空
python3 transfer.py   
python3 setup.py build_ext --inplace
python3 evaluation.py
```
执行完成后，会打印出在widerface easy测试集上的AP：
```bash
==================== Results ====================
Easy   Val AP: 0.892260565399806
=================================================
```

### 3.2 性能测试
可以使用bmrt_test测试模型的理论性能：
```bash
bmrt_test --bmodel {path_of_bmodel}
```
也可以参考[1.2 测试命令](#1.2-测试命令)打印程序运行中的实际性能指标。  
测试中性能指标存在一定的波动属正常现象。

### 3.3 测试结果
[Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)中模型使用original image scale在widerface easy测试集上的准确率为90.7%。
本例程更换resize策略，将图片大小resize到640*640进行推理，在该测试集上准确率为89.2%。

在BM1684X上，不同例程、不同模型的性能测试结果如下：
|       例程        | 精度  |batch_size|  ACC    | infer_time | QPS        |
|   ------------    | ---- | -------  |  -----  |  -----     | ---        |
| retinaface_opencv | fp32 |   1      |  89.2%  |  6.50ms    |  153.9     |
| retinaface_opencv | fp32 |   4      |  89.2%  |  6.28ms    |  159.3     |
| retinaface_bmcv   | fp32 |   1      |  89.0%  |  4.84ms    |  206.7     |
| retinaface_bmcv   | fp32 |   4      |  87.1%  |  4.56ms    |  219.3     |

在BM1684上，不同例程、不同模型的性能测试结果如下：
|       例程        | 精度   |batch_size|  ACC    | infer_time | QPS        |
|   ------------    | ----  | -------  |  -----  | -----      | ---        |
| retinaface_opencv | fp32  |   1      |  89.1%  |  9.07ms    | 110.301122 |
| retinaface_opencv | fp32  |   4      |  89.1%  |  8.52ms    | 117.421905 |
| retinaface_bmcv   | fp32  |   1      |  89.1%  |  6.66ms    | 150.184443 |
| retinaface_bmcv   | fp32  |   4      |  89.1%  |  6.35ms    | 157.523946 |


```
infer_time: 程序运行时每张图的实际推理时间；
QPS: 程序每秒钟处理的图片数。
```
说明：性能测试的结果具有一定的波动性。


