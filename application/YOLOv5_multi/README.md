[简体中文](./README.md)

# YOLOv5

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-faq)
  
## 1. 简介
​YOLOv5是非常经典的基于anchor的One Stage目标检测算法，因其优秀的精度和速度表现，在工程实践应用中获得了非常广泛的应用。本例程对[​YOLOv5官方开源仓库](https://github.com/ultralytics/yolov5)v6.1版本的模型和算法进行移植，使之能在SOPHON BM1684\BM1684X\BM1688上进行推理测试。

## 2. 特性
* 支持BM1688(SoC)，BM1684X(x86 PCIe、SoC)，BM1684(x86 PCIe、SoC)
* 支持FP32、FP16(BM1684X、BM1688)、INT8模型推理
* 支持C++多线程，前后处理推理并行的pipeline推理
* 支持单batch和多batch模型推理
* 支持图片和视频测试

## 3. 准备模型与数据
参考[sophon-demo yolov5模型编译](../../sample/YOLOv5/README.md#3-准备模型与数据)

## 4. 模型编译
参考[sophon-demo yolov5模型编译](../../sample/YOLOv5/README.md#4-模型编译)

## 5. 例程测试
- [C++例程](./cpp/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(conf_thresh=0.001、nms_thresh=0.6)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json cpp/yolov5_bmcv/results/yolov5.json
```
### 6.2 测试结果
在coco2017val_1000数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |              测试模型               |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
| BM1684 PCIe  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.375         | 0.573    |
| BM1684 PCIe  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.339         | 0.544    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.375         | 0.573    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.375         | 0.573    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.358         | 0.562    |
| BM1688 soc   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.375         | 0.573    |
| BM1688 soc   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.375         | 0.573    |
| BM1688 soc   | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.355         | 0.565    |


> **测试说明**：  
> 1. batch_size=4和batch_size=1的模型精度一致；
> 2. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 3. AP@IoU=0.5:0.95为area=all对应的指标。


## 7. 性能测试
### 7.1 bmrt_test
参考[sophon-demo yolov5性能测试](../../sample/YOLOv5/README.md#71-bmrt_test)

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)运行程序，并查看统计的fps。

在不同的测试平台上，使用不同的例程、模型测试`datasets/test_car_person_1080P.mp4`，conf_thresh=0.5，nms_thresh=0.5，4预处理线程，8推理线程，性能测试结果如下：
|    测试平台 |     测试程序       |             测试模型                    |   config           | 路数    | tpu利用率(%)| 设备内存(MB) | cpu利用率(%) | 系统内存(MB) | fps  |
| ----------- | ---------------- | ----------------------------------------- | ----------------   | ------- | ----------  | ---------- | ----------  | ------------| ---- |
| SE5-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel       | config_se5.json    | 16      |  90~100     | 1650~1750  | 80~100      | 190~210     |  42  |
| SE5-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel       | config_se5.json    | 16      |  85~100     | 1490~1520  | 150~180     | 190~210     |  77  |
| SE5-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b.bmodel       | config_se5.json    | 16      |  75~95      | 2800~2900  | 270~310     | 190~210     |  129 |
| SE7-32      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel       | config_se7.json    | 32      |  90~100     | 2220~2270  | 70~100      | 220~250     |  35  |
| SE7-32      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel       | config_se7.json    | 32      |  75~90      | 2220~2260  | 190~230     | 220~250     |  95  |
| SE7-32      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel       | config_se7.json    | 32      |  60~80      | 2230~2260  | 350~400     | 230~250     |  167 |
| SE7-32      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b.bmodel       | config_se7.json    | 32      |  60~85      | 4180~4250  | 390~430     | 230~250     |  180 |
| SE9-8       | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b_2core.bmodel | config_se9-8.json  | 8       |  70~90      | 1650~1800  | 250~300     | 160~190     |  91 |
| SE9-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b_2core.bmodel | config_se9-16.json | 16      |  60~75      | 4300~4500  | 390~470     | 200~220     |  132 |


> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. BM1684/1684X SoC的主控处理器均为8核 ARM A53 42320 DMIPS @2.3GHz，SE9-16的主控处理器为8核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 3. 各项指标的查看方式可以参考[测试指标查看方式](../../docs/Check_Statis.md)


## 8. FAQ
其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。