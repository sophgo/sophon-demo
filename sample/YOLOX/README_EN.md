[简体中文](./README.md) | [English](./README_EN.md)

# YOLOx

## Catalogue

* [1. Introduction](#1-introduction)
* [2. Feature](#2-feature)
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
YOLOx, proposed by Megvii Technology Limited, is based on the improvement of YOLO series, introducing decoupling head and Anchor-free to improve the overall detection performance of the algorithm.

**URL of the paper** (https://arxiv.org/abs/2107.08430)

**URL of github repository** (https://github.com/Megvii-BaseDetection/YOLOX)

## 2. Feature
* Support for BM1684X(x86 PCIe、SoC) and BM1684(x86 PCIe、SoC、arm PCIe)
* Support for FP32, FP16 (BM1684X), INT8 model compilation and inference
* Support C++ inference based on BMCV preprocessing
* Support Python inference based on OpenCV and BMCV preprocessing
* Support single batch and multi-batch model inference
* Support for picture and video testing
 
## 3. Prepare Models and Data

It is recommended to use TPU-MLIR to compile BModel, Pytorch model to export to onnx model before compilation. For detail, please see https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime

At the same time, you need to prepare datasets for testing and, if quantifying the model, for quantification.

This routine provides download scripts `download.sh` for the relevant models and datasets in the `scripts` directory, or you can prepare the models and datasets yourself and refer to [4. Model Compilation] (#4-model-compilation)for model conversion.


```bash
# Install unzip, skip if already installed, use yum or other means to install on non-ubuntu systems as appropriate.
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

The downloadable models include
```
./models
├── BM1684
│   ├── yolox_s_fp32_1b.bmodel   # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 for BM1684
│   ├── yolox_s_fp32_4b.bmodel   # Compiled with TPU-MLIR, FP32 BModel,batch_size=4 for BM1684
│   ├── yolox_s_int8_1b.bmodel   # Compiled with TPU-MLIR, INT8 BModel,batch_size=1 for BM1684
│   └── yolox_s_int8_4b.bmodel   # Compiled with TPU-MLIR, INT8 BModel,batch_size=4 for BM1684
├── BM1684X
│   ├── yolox_s_fp32_1b.bmodel   # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 for BM1684X
│   ├── yolox_s_fp32_4b.bmodel   # Compiled with TPU-MLIR, FP32 BModel,batch_size=4 for BM1684X
│   ├── yolox_s_fp16_1b.bmodel   # Compiled with TPU-MLIR, FP16 BModel,batch_size=1 for BM1684X
│   ├── yolox_s_fp16_4b.bmodel   # Compiled with TPU-MLIR, FP16 BModel,batch_size=4 for BM1684X
│   ├── yolox_s_int8_1b.bmodel   # Compiled with TPU-MLIR, INT8 BModel,batch_size=1 for BM1684X
│   └── yolox_s_int8_4b.bmodel   # Compiled with TPU-MLIR, INT8 BModel,batch_size=4 for BM1684X
│── torch
│   └── yolox_s.pt               # Torchscript model after trace
└── onnx
    └── yolox.onnx               # Derived onnx dynamic model      
    └── yolox_s.qtable           # 用于MLIR混精度移植的配置文件
```

Downloaded data included:
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

Executing the above command will generate the `yolox_s_fp32_1b.bmodel` file under `models/BM1684 or models/BM1684X/`, that is, the converted FP32 BModel.

- GenerateFP16 BModel

This example provides a script for TPU-MLIR to compile FP16 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_fp16bmodel_mlir.sh`, and specify the target platform on which BModel runs (**BM1684X is supported**) during execution, such as:

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

Executing the above command will generate the `yolox_s_fp16_1b.bmodel` file under `models/BM1684X/`, that is, the converted FP16 BModel.

- Generate INT8 BModel

This example provides a script for quantifying INT8 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_int8bmodel_mlir.sh`, and enter the target platform of BModel (**BM1684/BM1684X is supported**) during execution, such as:

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684
#or
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

The above script will generate files such as `yolox_s_int8_1b.bmodel` under `models/BM1684 or models/BM1684X`, that is, the converted INT8 BModel.


## 5. Example Test
- [C++ Example](./cpp/README.md)
- [Python Example](./python/README.md)

## 6. Precision Test
### 6.1 Testing Method

First of all, refer to [C++ example](cpp/README_EN.md#32-image-test-demo) or [Python example](python/README_EN.md#22-image-test-demo) to deduce the dataset to be tested, generate the predicted json file, and pay attention to modifying the dataset (datasets/coco/val2017_1000) and related parameters (conf_thresh=0.001, nms_thresh=0.6). 
Then, use the `eval_coco.py` script under the `tools` directory to compare the json file generated by the test with the test set tag json file, and calculate the evaluation metrics for target detection. The command is as follows:
```bash
# Install pycocotools, skip if it is already installed
pip3 install pycocotools
# Please modify the program path and json file path according to the actual situation
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolox_s_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```

### 6.2 Test Result
On the coco2017val_1000 dataset, the accuracy test results are as follows:
| Test Platform|   Test Program  |             Test model              |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ----------------| ----------------------------------- | ------------- | -------- |
| BM1684 PCIe  | yolox_opencv.py  | yolox_s_fp32_1b.bmodel            |      0.366      |   0.530    |
| BM1684 PCIe  | yolox_opencv.py  | yolox_s_int8_1b.bmodel            |      0.335      |   0.493    |
| BM1684 PCIe  | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel            |      0.363      |   0.525    |
| BM1684 PCIe  | yolox_bmcv.py    | yolox_s_int8_1b.bmodel            |      0.329      |   0.485    |
| BM1684 PCIe  | yolox_bmcv.pcie  | yolox_s_fp32_1b.bmodel            |      0.364      |   0.534    |
| BM1684 PCIe  | yolox_bmcv.pcie  | yolox_s_int8_1b.bmodel            |      0.332      |   0.498    |
| BM1684 PCIe  | yolox_sail.pcie  | yolox_s_fp32_1b.bmodel            |      0.351      |   0.516    |
| BM1684 PCIe  | yolox_sail.pcie  | yolox_s_int8_1b.bmodel            |      0.319      |   0.478    |
| BM1684X PCIe | yolox_opencv.py  | yolox_s_fp32_1b.bmodel            |      0.366      |   0.530    |
| BM1684X PCIe | yolox_opencv.py  | yolox_s_fp16_1b.bmodel            |      0.366      |   0.530    |
| BM1684X PCIe | yolox_opencv.py  | yolox_s_int8_1b.bmodel            |      0.357      |   0.529    |
| BM1684X PCIe | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel            |      0.363      |   0.525    |
| BM1684X PCIe | yolox_bmcv.py    | yolox_s_fp16_1b.bmodel            |      0.363      |   0.525    |
| BM1684X PCIe | yolox_bmcv.py    | yolox_s_int8_1b.bmodel            |      0.353      |   0.524    |
| BM1684X PCIe | yolox_bmcv.pcie  | yolox_s_fp32_1b.bmodel            |      0.363      |   0.534    |
| BM1684X PCIe | yolox_bmcv.pcie  | yolox_s_fp16_1b.bmodel            |      0.363      |   0.534    |
| BM1684X PCIe | yolox_bmcv.pcie  | yolox_s_int8_1b.bmodel            |      0.351      |   0.527    |
| BM1684X PCIe | yolox_sail.pcie  | yolox_s_fp32_1b.bmodel            |      0.350      |   0.516    |
| BM1684X PCIe | yolox_sail.pcie  | yolox_s_fp16_1b.bmodel            |      0.350      |   0.516    |
| BM1684X PCIe | yolox_sail.pcie  | yolox_s_int8_1b.bmodel            |      0.337      |   0.506    |

> **Test Description**:
> 1. The model accuracy of batch_size=4 and batch_size=1 is the same.
> 2. The model accuracy of SoC and PCIe is the same.
> 3. AP@IoU=0.5:0.95 is the corresponding indicator of area=all.

## 7. Performance Testing
### 7.1 bmrt_test
Use bmrt_test to test the theoretical performance of the model:
```bash
# Please modify the bmodel path and devid parameters to be tested according to the actual situation
bmrt_test --bmodel models/BM1684/yolox_s_fp32_1b.bmodel
```
The `calculate time` in the test results is the inference time of the model, and the theoretical inference time of each image is when the multi-batch size model is divided by the corresponding batch size.
The theoretical inference time of each model is tested, and the results are as follows:

|          Test model            | calculate time(ms) |
| ------------------------------ | ----------------- |
| BM1684/yolox_s_fp32_1b.bmodel  |       26.01       |
| BM1684/yolox_s_fp32_4b.bmodel  |       25.62       |
| BM1684/yolox_s_int8_1b.bmodel  |       16.54       |
| BM1684/yolox_s_int8_4b.bmodel  |       11.72       |
| BM1684X/yolox_s_fp32_1b.bmodel |       27.92       |
| BM1684X/yolox_s_fp32_4b.bmodel |       25.63       |
| BM1684X/yolox_s_fp16_1b.bmodel |       6.27        |
| BM1684X/yolox_s_fp16_4b.bmodel |       6.15        |
| BM1684X/yolox_s_int8_1b.bmodel |       3.86        |
| BM1684X/yolox_s_int8_4b.bmodel |       3.69        |

> **Test Description**：  
> 1. The performance test results have a certain volatility.
> 2. The `calculate time` has been converted to the average inference time per picture.
> 3. The test results of SoC and PCIe are basically the same.

### 7.2 Program Performance
Refer to [C++ example](cpp/README_EN.md) or [Python example](python/README_EN.md) to run the program, and check the statistical decoding time, preprocessing time, inference time, post-processing time. The preprocessing time, inference time and post-processing time of C++ example printing are the whole batch processing time, which needs to be divided by the corresponding batch size to get the processing time of each picture.

Use different examples and models to test `datasets/val2017_1000` with `conf_thresh=0.5,nms_thresh=0.5` on different test platforms. The performance test results are shown as follows:
|Test Platform|  Test Program    |             Test model              |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
| BM1684 SoC  | yolox_opencv.py | yolox_s_fp32_1b.bmodel | 3.29     | 13.77   | 40.40     | 75.62    |
| BM1684 SoC  | yolox_opencv.py | yolox_s_int8_1b.bmodel | 3.28     | 13.86   | 43.68     | 76.78    |
| BM1684 SoC  | yolox_opencv.py | yolox_s_int8_4b.bmodel | 4.07     | 14.73   | 38.44     | 74.24    |
| BM1684 SoC  | yolox_bmcv.py   | yolox_s_fp32_1b.bmodel | 3.70     | 2.88    | 28.16     | 75.79    |
| BM1684 SoC  | yolox_bmcv.py   | yolox_s_int8_1b.bmodel | 2.64     | 2.45    | 18.75     | 75.86    |
| BM1684 SoC  | yolox_bmcv.py   | yolox_s_int8_4b.bmodel | 2.58     | 2.30    | 13.86     | 74.06    |
| BM1684 SoC  | yolox_bmcv.soc  | yolox_s_fp32_1b.bmodel | 4.52     | 1.72    | 25.78     | 2.71     |
| BM1684 SoC  | yolox_bmcv.soc  | yolox_s_int8_1b.bmodel | 4.57     | 1.78    | 16.32     | 2.73     |
| BM1684 SoC  | yolox_bmcv.soc  | yolox_s_int8_4b.bmodel | 4.57     | 1.73    | 11.58     | 2.61     |
| BM1684 SoC  | yolox_sail.soc  | yolox_s_fp32_1b.bmodel | 2.58     | 3.91    | 26.24     | 2.10     |
| BM1684 SoC  | yolox_sail.soc  | yolox_s_int8_1b.bmodel | 2.59     | 2.38    | 16.83     | 2.08     |
| BM1684 SoC  | yolox_sail.soc  | yolox_s_int8_4b.bmodel | 2.51     | 2.31    | 11.97     | 2.10     |
| BM1684X SoC | yolox_opencv.py | yolox_s_fp32_1b.bmodel | 3.37     | 13.56   | 52.09     | 77.07    |
| BM1684X SoC | yolox_opencv.py | yolox_s_fp16_1b.bmodel | 3.38     | 13.43   | 29.83     | 78.84    |
| BM1684X SoC | yolox_opencv.py | yolox_s_int8_1b.bmodel | 3.37     | 13.48   | 27.52     | 77.18    |
| BM1684X SoC | yolox_opencv.py | yolox_s_int8_4b.bmodel | 3.84     | 15.42   | 28.95     | 76.63    |
| BM1684X SoC | yolox_bmcv.py   | yolox_s_fp32_1b.bmodel | 2.49     | 2.74    | 33.98     | 77.05    |
| BM1684X SoC | yolox_bmcv.py   | yolox_s_fp16_1b.bmodel | 2.46     | 2.73    | 11.97     | 78.40    |
| BM1684X SoC | yolox_bmcv.py   | yolox_s_int8_1b.bmodel | 2.44     | 2.76    | 9.37      | 77.40    |
| BM1684X SoC | yolox_bmcv.py   | yolox_s_int8_4b.bmodel | 2.25     | 2.59    | 10.11     | 75.72    |
| BM1684X SoC | yolox_bmcv.soc  | yolox_s_fp32_1b.bmodel | 4.28     | 0.95    | 29.05     | 2.68     |
| BM1684X SoC | yolox_bmcv.soc  | yolox_s_fp16_1b.bmodel | 4.20     | 0.95    | 7.10      | 2.65     |
| BM1684X SoC | yolox_bmcv.soc  | yolox_s_int8_1b.bmodel | 4.23     | 0.95    | 4.50      | 2.65     |
| BM1684X SoC | yolox_bmcv.soc  | yolox_s_int8_4b.bmodel | 4.08     | 0.93    | 4.46      | 2.92     |
| BM1684X SoC | yolox_sail.soc  | yolox_s_fp32_1b.bmodel | 2.37     | 3.67    | 29.50     | 2.05     |
| BM1684X SoC | yolox_sail.soc  | yolox_s_fp16_1b.bmodel | 2.38     | 3.62    | 7.55      | 2.05     |
| BM1684X SoC | yolox_sail.soc  | yolox_s_int8_1b.bmodel | 2.351    | 3.63    | 4.95      | 2.04     |
| BM1684X SoC | yolox_sail.soc  | yolox_s_int8_4b.bmodel | 2.21     | 3.22    | 4.86      | 2.04     |

> **Test Description**：  
> 1. The time units are all milliseconds (ms), and the statistical time is the average processing time of each image.
> 2. The performance test results are volatile to a certain extent, so it is recommended that the average value should be taken from multiple tests.
> 3. test platform of BM1684 SoC is standard SE5, and test platform of BM1684X SoC is standard SE7
> 4. BM1684/1684X SoC's master CPU are all 8-core ARM A53 42320 DMIPS @ 2.3GHz CPU performance on PCIe may vary greatly due to different PCIes.
> 5. The image resolution has a great influence on the decoding time, the reasoning result has a great influence on the post-processing time, different test pictures may be different, and different thresholds have a great influence on the post-processing time.

## 8. FAQ
[Frequently Asked Questions](../../docs/FAQ_EN.md)
