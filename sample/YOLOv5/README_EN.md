[简体中文](./README.md) | [English](./README_EN.md)

# YOLOv5

## Catalogue

* [1. Introduction](#1-introduction)
* [2. Characteristics](#2-characteristics)
* [3. Prepare Models and Data](#3-prepare-models-and-data)
* [4. Model Compilation](#4-model-compilation)
* [5. Example Test](#5-example-test)
* [6. Precision Test](#6-precision-test)
  * [6.1 Testing Method](#61-testing-method)
  * [6.2 Test Result](#62-test-result)
* [7. Performance Testing](#7-performance-testing)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 Program Performance](#72-program-performance)
* [8. FAQ](#8-faq)
  
## 1. Introduction
YOLOv5 is a very classical One Stage target detection algorithm based on anchor. Because of its excellent accuracy and speed performance, it has been widely used in engineering practice. This example [​YOLOv5 official open source repository](https://github.com/ultralytics/yolov5) transplants the v6.1 version of the model and algorithm so that it can be inference tested on SOPHON BM1684 and BM1684X.

## 2. Characteristics
* Support for BM1684X(x86 PCIe、SoC) and BM1684(x86 PCIe、SoC、arm PCIe)
* Support for FP32, FP16 (BM1684X), INT8 model compilation and inference
* Support C++ inference based on BMCV preprocessing
* Support Python inference based on OpenCV and BMCV preprocessing
* Support single batch and multi-batch model inference
* Support 1 output and 3 output model inference
* Support for picture and video testing
 
## 3. Prepare Models and Data
It is recommended to use TPU-MLIR to compile BModel, Pytorch model to export to onnx model before compilation. For more information, please see [YOLOv5 Model Export](./docs/YOLOv5_Export_Guide_EN.md).

At the same time, you need to prepare a dataset for testing and, if you quantify the model, a dataset for quantification.

This example provides a download script `download.sh` for related models and data in the `scripts` directory. You can also prepare your own models and data sets, and refer to [4. Model Compilation](#4-model compilation) for model transformation.

```bash
# Install unzip, skip if it is already installed
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

Downloaded models include:
```
./models
├── BM1684
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 for BM1684
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel   # Compiled with TPU-MLIR, INT8 BModel,batch_size=1 for BM1684
│   └── yolov5s_v6.1_3output_int8_4b.bmodel   # Compiled with TPU-MLIR, INT8 BModel,batch_size=4 for BM1684
├── BM1684X
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel   # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 for BM1684X
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel   # Compiled with TPU-MLIR, FP16 BModel,batch_size=1 for BM1684X
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel   # Compiled with TPU-MLIR, INT8 BModel,batch_size=1 for BM1684X
│   └── yolov5s_v6.1_3output_int8_4b.bmodel   # Compiled with TPU-MLIR, INT8 BModel,batch_size=4 for BM1684X
│── torch
│   └── yolov5s_v6.1_3output.torchscript.pt   # Torchscript model after trace
└── onnx
    └── yolov5s_v6.1_3output.onnx             # Derived onnx dynamic model       
```
The downloaded data include:
```
./datasets
├── test                                      # Test picture
├── test_car_person_1080P.mp4                 # Test video
├── coco.names                                # Coco category name file
├── coco128                                   # Coco128 dataset for model quantization
└── coco                                      
    ├── val2017_1000                          # coco val2017_1000 dataset：1000 randomly selected samples from coco val2017
    └── instances_val2017_1000.json           # coco val2017_1000Dataset label file, used to calculate accuracy evaluation indicators 
```

## 4. Model Compilation
The exported model needs to be compiled into BModel to run on SOPHON TPU. If you use the downloaded BModel, you can skip this section. It is recommended that you use TPU-MLIR to compile BModel.

You need to install TPU-MLIR before compiling the model. For more information, please see [TPU-MLIR Environment Building](../../docs/Environment_Install_Guide_EN.md#1-tpu-mlir-environmental-installation). After installation, you need to enter the example directory in the TPU-MLIR environment. Use TPU-MLIR to compile the onnx model to BModel. For specific methods, please refer to "chapter 3.5" of the TPU-MLIR Quick start Manual. Compile the ONNX model (please obtain it from the corresponding version of SDK of [Sophgo official website](https://developer.sophgo.com/site/index/material/31/all.html)).

- Generate FP32 BModel

This example provides a script for TPU-MLIR to compile FP32 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_fp32bmodel_mlir.sh`, and specify the target platform on which BModel runs (**BM1684/BM1684X is supported**) during execution, such as:

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684
#or 
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

Executing the above command will generate the `yolov5s_v6.1_3output_fp32_1b.bmodel` file under `models/BM1684 or models/BM1684X/`, that is, the converted FP32 BModel.

- Generate FP16 BModel

This example provides a script for TPU-MLIR to compile FP16 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_fp16bmodel_mlir.sh`, and specify the target platform on which BModel runs (**BM1684X is supported**) during execution, such as:

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

Executing the above command will generate the `yolov5s_v6.1_3output_fp16_1b.bmodel` file under `models/BM1684X/`, that is, the converted FP16 BModel.

- Generate INT8 BModel

This example provides a script for quantifying INT8 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_int8bmodel_mlir.sh`, and enter the target platform of BModel (**BM1684/BM1684X is supported**) during execution, such as:

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684
# or
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

The above script will generate files such as `yolov5s_v6.1_3output_int8_1b.bmodel` under `models/BM1684 or models/BM1684X`, that is, the converted INT8 BModel.

## 5. Example Test
- [C++ Example](./cpp/README_EN.md)
- [Python Example](./python/README_EN.md)

## 6. Precision Test
### 6.1 Testing Method

First of all, refer to [C++ example](cpp/README_EN.md#32-image-test-demo) or [Python example](python/README_EN.md#22-image-test-demo) to deduce the dataset to be tested, generate the predicted json file, and pay attention to modifying the dataset (datasets/coco/val2017_1000) and related parameters (conf_thresh=0.001, nms_thresh=0.6).
Then, use the `test generated .py` script under the `tools` directory to compare the json file generated by the test with the test set tag json file, and calculate the evaluation metrics for target detection. The command is as follows:
```bash
# Install pycocotools, skip if it is already installed
pip3 install pycocotools
# Please modify the program path and json file path according to the actual situation
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov5s_v6.1_3output_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 Test Result
On the coco2017val_1000 dataset, the accuracy test results are as follows:
| Test Platform|  Test Program    |            Test model               |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
| BM1684 PCIe  | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.377         | 0.580    |
| BM1684 PCIe  | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 0.344         | 0.553    |
| BM1684 PCIe  | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.367         | 0.567    |
| BM1684 PCIe  | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.332         | 0.536    |
| BM1684 PCIe  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.362         | 0.568    |
| BM1684 PCIe  | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.331         | 0.540    |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.377         | 0.580    |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.377         | 0.580    |
| BM1684X PCIe | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 0.363         | 0.572    |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.368         | 0.567    |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.368         | 0.567    |
| BM1684X PCIe | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.353         | 0.561    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.362         | 0.568    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.362         | 0.568    |
| BM1684X PCIe | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.348         | 0.561    |

> **Test Description**:
> 1. The model accuracy of batch_size=4 and batch_size=1 is the same.
> 2. The model accuracy of SoC and PCIe is the same.
> 3. AP@IoU=0.5:0.95 is the corresponding indicator of area=all.

## 7. Performance Testing
### 7.1 bmrt_test
Use bmrt_test to test the theoretical performance of the model:
```bash
# Please modify the bmodel path and devid parameters to be tested according to the actual situation
bmrt_test --bmodel models/BM1684/yolov5s_v6.1_3output_fp32_1b.bmodel
```
The `calculate time` in the test results is the inference time of the model, and the theoretical inference time of each image is when the multi-batch size model is divided by the corresponding batch size.
The theoretical inference time of each model is tested, and the results are as follows:

|                  Test model                 | calculate time(ms) |
| ------------------------------------------- | ----------------- |
| BM1684/yolov5s_v6.1_3output_fp32_1b.bmodel  | 22.6              |
| BM1684/yolov5s_v6.1_3output_int8_1b.bmodel  | 11.5              |
| BM1684/yolov5s_v6.1_3output_int8_4b.bmodel  | 6.4               |
| BM1684X/yolov5s_v6.1_3output_fp32_1b.bmodel | 20.8              |
| BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel | 7.2               |
| BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel | 3.5               |
| BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel | 3.3               |

> **Test Description**：  
> 1. The performance test results have a certain volatility.
> 2. The `calculate time` has been converted to the average inference time per picture.
> 3. The test results of SoC and PCIe are basically the same.

### 7.2 Program Performance
Refer to [C++ example](cpp/README_EN.md) or [Python example](python/README_EN.md) to run the program, and check the statistical decoding time, preprocessing time, inference time, post-processing time. The preprocessing time, inference time and post-processing time of C++ example printing are the whole batch processing time, which needs to be divided by the corresponding batch size to get the processing time of each picture.

Use different examples and models to test `datasets/val2017_1000` with `conf_thresh=0.5,nms_thresh=0.5` on different test platforms. The performance test results are shown as follows:
|Test Platform|  Test Program    |            Test model               |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
| BM1684 SoC  | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 14.0     | 27.8          | 33.5          | 115       |
| BM1684 SoC  | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 13.9     | 23.5          | 33.5          | 111       |
| BM1684 SoC  | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel | 13.8     | 24.2          | 28.2          | 115       |
| BM1684 SoC  | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 3.0      | 3.0           | 28.5          | 111       |
| BM1684 SoC  | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 3.0      | 2.4           | 17.4          | 111       |
| BM1684 SoC  | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 2.8      | 2.3           | 11.5          | 115       |
| BM1684 SoC  | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 5.4      | 1.5           | 22.6          | 19.3      |
| BM1684 SoC  | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 5.4      | 1.5           | 11.5          | 19.3      |
| BM1684 SoC  | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 5.2      | 1.6           | 6.2           | 19.2      |
| BM1684 SoC  | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 3.3      | 3.1           | 23.3          | 22.6      |
| BM1684 SoC  | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 3.3      | 1.9           | 12.2          | 22.3      |
| BM1684 SoC  | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 3.1      | 1.8           | 6.9           | 22.5      |
| BM1684X SoC | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 15.0     | 22.4          | 32.0          | 104       |
| BM1684X SoC | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel | 15.0     | 22.4          | 18.5          | 104       |
| BM1684X SoC | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 15.0     | 22.4          | 14.2          | 104       |
| BM1684X SoC | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel | 14.9     | 23.1          | 14.5          | 108       |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 3.1      | 2.4           | 28.8          | 104       |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 3.1      | 2.4           | 15.5          | 104       |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 3.1      | 2.4           | 10.9          | 104       |
| BM1684X SoC | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 2.9      | 2.3           | 9.8           | 109       |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 4.7      | 0.8           | 22.1          | 18.7      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 4.7      | 0.8           | 8.5           | 18.7      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 4.7      | 0.8           | 4.3           | 18.8      |
| BM1684X SoC | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 4.7      | 0.8           | 4.1           | 18.6      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 2.8      | 3.1           | 23.0          | 21.0      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 2.8      | 3.1           | 9.5           | 21.0      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 2.8      | 3.1           | 5.2           | 20.9      |
| BM1684X SoC | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 2.6      | 2.5           | 4.8           | 21.1      |


> **Test Description**：  
> 1. The time units are all milliseconds (ms), and the statistical time is the average processing time of each image.
> 2. The performance test results are volatile to a certain extent, so it is recommended that the average value should be taken from multiple tests.
> 3. BM1684/1684X SoC's master CPU are all 8-core ARM A53 42320 DMIPS @ 2.3GHz CPU performance on PCIe may vary greatly due to different PCIes.
> 4. The image resolution has a great influence on the decoding time, the reasoning result has a great influence on the post-processing time, different test pictures may be different, and different thresholds have a great influence on the post-processing time.

## 8. FAQ
Please refer to [YOLOv5 Common Problems](./docs/YOLOv5_Common_Problems_EN.md) to see some problems of YOLOv5 inference.For other questions ,please refer to [FAQ](../../docs/FAQ_EN.md) to see some common questions and answers.