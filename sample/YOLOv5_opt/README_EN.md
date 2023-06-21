[简体中文](./README.md) | [English](./README_EN.md)

# YOLOv5_opt🚀

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. Features](#2-features)
* [3. Prepare Model and Data](#3-prepare-model-and-data)
* [4. Model Compilation](#4-model-compilation)
  * [4.1 Compile BModel with TPU-MLIR](#41-compile-bmodel-with-tpu-mlir)
* [5. Example Testing](#5-example-testing)
* [6. mAP Testing](#6-map-testing)
  * [6.1 Testing Method](#61-testing-method)
  * [6.2 Testing Results](#62-testing-result)
* [7. Performance Testing](#7-performance-testing)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 Program Running Performance](#72-program-running-performance)
* [8. FAQ](#8-faq)

## 1. Introduction
This example is based on [YOLOv5](../YOLOv5/README_EN.md), using the tpu_kernel `tpu_kernel_api_yolov5_detect_out` operator on BM1684X to accelerate post-processing, the acceleration effect is remarkable.
## 2. Features
* Supports BM1684X (x86 PCIe, SoC)
* Supports FP32, FP16(BM1684X), and INT8 model compilation and inference
* Supports C++ inference based on BMCV preprocessing and TPUKERNEL postprocessing
* Supports single batch and multi-batch model inference
* Supports image and video testing

## 3. Prepare Model and Data
It is recommended to use TPU-MLIR to compile BModel, and the Pytorch model needs to be exported to an ONNX model before compilation. This example do not support single-output model, **It is suggested to use the N-output model for better performance. N-output refers to the output of the last N convolutional layers of the source model (N <= 8). To export, please refer to the [YOLOv5_tpukernel_Export_Guide](./docs/YOLOv5_tpukernel_Export_Guide_EN.md)。**

At the same time, you need to prepare the dataset for testing. If you want to quantize the model, you also need to prepare the dataset for quantization.

The download script `download.sh` for relevant models and data is provided in the `scripts` directory of this tutorial. You can also prepare your own models and datasets and refer to [4. Model Compilation](#4-model-compilation) for model conversion.

```bash
# Skip this part if unzip is already installed
sudo apt install unzip # if not ubuntu system, please use other tools.
chmod -R +x scripts/
./scripts/download.sh
```

models downloaded including：
```
./models
├── BM1684X
│   ├── yolov5s_tpukernel_fp32_1b.bmodel   # compile with TPU-MLIR, FP32 BModel for BM1684X,batch_size=1
│   ├── yolov5s_tpukernel_fp16_1b.bmodel   # compile with TPU-MLIR, FP16 BModel for BM1684X,batch_size=1
│   ├── yolov5s_tpukernel_int8_1b.bmodel   # compile with TPU-MLIR, INT8 BModel for BM1684X,batch_size=1
│   └── yolov5s_tpukernel_int8_4b.bmodel   # compile with TPU-MLIR, INT8 BModel for BM1684X,batch_size=4
└── onnx
    └── yolov5s_tpukernel.onnx             # dynamic ONNX model, exported from original weights yolov5s.pt
```

datasets downloaded including：
```
./datasets
├── test                                   # test images
├── test_car_person_1080P.mp4              # test video
├── coco.names                             # coco name file
├── coco128                                # coco 128 images set,for quantize
└── coco                                   
    ├── val2017_1000                       # 1000 images random selected from coco val2017
    └── instances_val2017_1000.json        # coco val2017_1000 labels, for mAP evaluation 
```

## 4. Model Compilation
The exported model needs to be compiled into a BModel to run on SOPHON TPU. If a downloaded BModel is used, this section can be skipped. It is recommended to use TPU-MLIR to compile the BModel.

### 4.1 Compile BModel with TPU-MLIR
Before compiling BModel, TPU-MLIR should be installed, please refer to [TPU-MLIR-SETUP](../../docs/Environment_Install_Guide_EN.md#2-tpu-mlir-environmental-installation). After installation, enter the demo directory in the TPU-MLIR environment. Use TPU-MLIR to compile the ONNX model into a BModel. For specific instructions, please refer to "3. Compiling ONNX models" in the "TPU-MLIR Quick Start Guide" (available from the corresponding version of the SDK on the [Sophon website](https://developer.sophgo.com/site/index/material/31/all.html)).

- Compile FP32 BModel

This tutorial provides a script for compiling FP32 BModel using TPU-MLIR in the `scripts` directory. Please modify the `gen_fp32bmodel_mlir.sh` script with the onnx model path, the directory for storing the generated model, and the input size shapes. Also, specify the target platform for running BModel when executing the script (BM1684X is supported), such as:

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```
The above command will generate the converted FP32 BModel `yolov5s_tpukernel_fp32_1b.bmodel` in the `models/BM1684X/` directory.

- Compile FP16 BModel

This tutorial provides a script for compiling FP16 BModel using TPU-MLIR in the `scripts` directory. Please modify the `gen_fp16bmodel_mlir.sh` script with the onnx model path, the directory for storing the generated model, and the input size shapes. Also, specify the target platform for running BModel when executing the script (BM1684X is supported), such as:

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```
The above command will generate the converted FP16 BModel `yolov5s_tpukernel_fp16_1b.bmodel` in the `models/BM1684X/` directory.

- Compile INT8 BModel

​This tutorial provides a script for compiling INT8 BModel using TPU-MLIR in the `scripts` directory. Please modify the `gen_int8bmodel_mlir.sh` script with the onnx model path, the directory for storing the generated model, and the input size shapes. Also, specify the target platform for running BModel when executing the script (BM1684X is supported), such as:

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x
```
The above command will generate the converted INT8 BModel `yolov5s_tpukernel_int8_1b.bmodel` and `yolov5s_tpukernel_int8_4b.bmodel` in the `models/BM1684X/` directory.

## 5. Example Testing
- [C++ Example](./cpp/README_EN.md)
- [Python Example](./python/README_EN.md)
## 6. mAP Testing
### 6.1 Testing Method

First, refer to [C++ Example](cpp/README_EN.md#32-image-test-demo) or [Python Example](python/README_EN.md#22-image-test-demo) infer your datasets, which will generate `*.json` prediction,note you should modify following parameters: `--input=datasets/coco/val2017_1000 --conf_thresh=0.1 --nms_thresh=0.6`.

Then, Use `tools/eval_coco.py` to calculate mAP, commands such as：
```bash
# Skip this step if already installed.
pip3 install pycocotools 
# Please modify the relevant parameters according to the actual situation.
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json cpp/yolov5_bmcv/results/yolov5s_tpukernel_fp32_1b.bmodel__bmcv_cpp_result.json
```
### 6.2 Testing Result
mAP on coco/val2017_1000 dataset:
|   platform   |      program     |              BModel              |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------------------| ------------- | -------- |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.353         | 0.536    |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.353         | 0.536    |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 0.339         | 0.527    |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.351         | 0.532    |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.351         | 0.532   |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.334         | 0.520    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.351         | 0.536    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.351         | 0.535    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.337         | 0.526    |
| BM1684X PCIe | yolov5_sail.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.351         | 0.536    |
| BM1684X PCIe | yolov5_sail.pcie | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.351         | 0.535    |
| BM1684X PCIe | yolov5_sail.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.337         | 0.526    |

> **Note**:
> 1. mAP of batch_size=4 BModel is the same as batch_size=1.
> 2. mAP of platform SoC is the same as PCIe.
> 3. AP@IoU=0.5:0.95 correspond to area=all.
> 4. To prevent TPU memory overflow due to enormous bboxes output, here we forcibly set `conf_thresh>=0.1, nms_thresh>=0.1`. Under the same parameters`(nms_thresh=0.6，conf_thresh=0.1)`, fp32 bmodel has `AP@IoU=0.5:0.95=0.345` in [YOLOv5 example](../YOLOv5/README.md)。

## 7. Performance Testing
### 7.1 bmrt_test
bmrt_test test BModel's theoretical inference time:
```bash
bmrt_test --bmodel models/BM1684X/yolov5s_tpukernel_fp32_1b.bmodel
```
The `calculate time` in the test results is the time of model inference, and the multi-batch size model should be divided by the corresponding batch size to be the theoretical inference time of each image.
The theoretical reasoning time of each model is tested, and the results are as follows:


|                  BModel                   | calculate time(ms) |
| -----------------------------------------| ----------------- |
| BM1684X/yolov5s_tpukernel_fp32_1b.bmodel | 19.6              |
| BM1684X/yolov5s_tpukernel_fp16_1b.bmodel | 6.2               |
| BM1684X/yolov5s_tpukernel_int8_1b.bmodel | 3.4               |
| BM1684X/yolov5s_tpukernel_int8_4b.bmodel | 3.2               |

> **Note**：  
> 1. The performance test results have certain volatility.
> 2. `calculate time` is converted to average inference time per image.
> 3. The SoC and PCIe test results are basically the same.

### 7.2 Program Running Performance
Refer to [C++ Example](cpp/README_EN.md) or [Python Example](./python/README_EN.md), and view the statistics of decode time, preprocessing time, inference time, and postprocess time. C++ Example print entire batch time, which needs to be divided by the corresponding batch size to be the processing time of each image.

Use different models to test `datasets/val2017_1000` with `conf_thresh=0.5,nms_thresh=0.5`. The performance test results are shown as follows:
|    platform  |     program      |             BModel            |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | --------------------------------| -------- | -------------- | --------- | --------- |
| BM1684X SoC | yolov5_opencv.py | yolov5s_tpukernel_fp32_1b.bmodel | 15.0     | 22.4          | 36.16          | 2.18      |
| BM1684X SoC | yolov5_opencv.py | yolov5s_tpukernel_fp16_1b.bmodel | 15.0     | 22.4          | 22.74          | 2.18      |
| BM1684X SoC | yolov5_opencv.py | yolov5s_tpukernel_int8_1b.bmodel | 15.0     | 22.4          | 20.16          | 2.18      |
| BM1684X SoC | yolov5_opencv.py | yolov5s_tpukernel_int8_4b.bmodel | 15.0     | 23.1          | 5.03           | 2.18      |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_tpukernel_fp32_1b.bmodel | 3.1      | 2.4           | 25.42          | 2.18      |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_tpukernel_fp16_1b.bmodel | 3.1      | 2.4           | 11.92          | 2.18      |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_tpukernel_int8_1b.bmodel | 3.1      | 2.4           | 9.05           | 2.18      |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_tpukernel_int8_4b.bmodel | 2.9      | 2.3           | 8.21           | 2.18      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_tpukernel_fp32_1b.bmodel | 4.7      | 0.8           | 19.67          | 1.20      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_tpukernel_fp16_1b.bmodel | 4.7      | 0.8           | 6.27           | 1.20      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_tpukernel_int8_1b.bmodel | 4.7      | 0.8           | 3.40           | 1.20      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_tpukernel_int8_4b.bmodel | 4.7      | 0.8           | 3.17           | 1.20      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_tpukernel_fp32_1b.bmodel | 2.8      | 3.1           | 19.70          | 1.38      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_tpukernel_fp16_1b.bmodel | 2.8      | 3.1           | 6.30           | 1.38      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_tpukernel_int8_1b.bmodel | 2.8      | 3.1           | 3.41           | 1.38      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_tpukernel_int8_4b.bmodel | 2.6      | 2.5           | 3.18           | 1.38      |

> **Note**：  
> 1. The time units are milliseconds (ms), and the statistical time is the average processing time of each image.
> 2. Performance test results are volatile, and it is recommended to average multiple tests；
> 3. CPU of BM1684X SoC is 8 core ARM A53 42320 DMIPS @2.3GHz, Performance on PCIe can vary greatly depending on the CPU
> 4. Image resolution has a great influence on decode time, inference results have a greater impact on postprocess time, different test images may have great differences, and different thresholds have a greater impact on postprocess time.

## 8. FAQ
Please refer to [FAQ](../../docs/FAQ_EN.md) to see some frequently asked questions and answers.