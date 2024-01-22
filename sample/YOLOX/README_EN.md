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
* [9. Acknowledgments](#9-acknowledgments)

## 1. Introduction
YOLOx, proposed by Megvii Technology Limited, is based on the improvement of YOLO series, introducing decoupling head and Anchor-free to improve the overall detection performance of the algorithm.

**URL of the paper** (https://arxiv.org/abs/2107.08430)

**URL of github repository** (https://github.com/Megvii-BaseDetection/YOLOX)

## 2. Feature
* Support for BM1688(SoC)、BM1684X(x86 PCIe、SoC) and BM1684(x86 PCIe、SoC、arm PCIe)
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
├── BM1688
│   ├── yolox_s_fp32_1b.bmodel         # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 and single core for BM1688
│   ├── yolox_s_fp32_1b_2core.bmodel   # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 and double cores for BM1688
│   ├── yolox_s_fp16_1b.bmodel         # Compiled with TPU-MLIR, FP16 BModel,batch_size=1 and single core for BM1688
│   ├── yolox_s_fp16_1b_2core.bmodel   # Compiled with TPU-MLIR, FP16 BModel,batch_size=1 and double cores for BM1688
│   ├── yolox_s_int8_1b.bmodel         # Compiled with TPU-MLIR, INT8 BModel,batch_size=1 and single core for BM1688
│   ├── yolox_s_int8_4b.bmodel         # Compiled with TPU-MLIR, INT8 BModel,batch_size=4 and single core for BM1688
│   ├── yolox_s_int8_1b_2core.bmodel   # Compiled with TPU-MLIR, INT8 BModel,batch_size=1 and double cores for BM1688
│   └── yolox_s_int8_4b_2core.bmodel   # Compiled with TPU-MLIR, INT8 BModel,batch_size=4 and double cores for BM1688
│── torch
│   ├── yolox_s.pt               # source model
|   └── yolox_s.torchscript.pt   # traced torchscript model
└── onnx
    └── yolox.onnx               # Derived onnx dynamic model      
    └── yolox_s.qtable           # qtable for mlir model_deploy
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

This example provides a script for TPU-MLIR to compile FP32 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_fp32bmodel_mlir.sh`, and specify the target platform on which BModel runs (**BM1684/BM1684X/BM1688 is supported**) during execution, such as:

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

Executing the above command will generate the `yolox_s_fp32_1b.bmodel` file under a folder like `models/BM1684`, that is, the converted FP32 BModel.

- Generate FP16 BModel

This example provides a script for TPU-MLIR to compile FP16 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_fp16bmodel_mlir.sh`, and specify the target platform on which BModel runs (**BM1684X/BM1688 is supported**) during execution, such as:

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688
```

Executing the above command will generate the `yolox_s_fp16_1b.bmodel` file under a folder like`models/BM1684X/`, that is, the converted FP16 BModel.

- Generate INT8 BModel

This example provides a script for quantifying INT8 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_int8bmodel_mlir.sh`, and enter the target platform of BModel (**BM1684/BM1684X is supported**) during execution, such as:

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

The above script will generate files such as `yolox_s_int8_1b.bmodel` under a folder like `models/BM1684`, that is, the converted INT8 BModel.


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
On the `datasets/coco/val2017_1000` dataset, the accuracy test results with `nms_thresh=0.6, conf_thresh=0.001` are as follows:
| Test Platform|   Test Program   |             Test model              |AP@IoU=0.5:0.95|AP@IoU=0.5  |
| ------------ | ---------------- | ----------------------------------- | ------------- | --------   |
| BM1684 PCIe  | yolox_opencv.py  |    yolox_s_fp32_1b.bmodel           |      0.403    |   0.590    |
| BM1684 PCIe  | yolox_opencv.py  |    yolox_s_int8_1b.bmodel           |      0.397    |   0.583    |
| BM1684 PCIe  | yolox_bmcv.py    |    yolox_s_fp32_1b.bmodel           |      0.402    |   0.590    |
| BM1684 PCIe  | yolox_bmcv.py    |    yolox_s_int8_1b.bmodel           |      0.397    |   0.582    |
| BM1684 PCIe  | yolox_bmcv.pcie  |    yolox_s_fp32_1b.bmodel           |      0.400    |   0.594    |
| BM1684 PCIe  | yolox_bmcv.pcie  |    yolox_s_int8_1b.bmodel           |      0.396    |   0.587    |
| BM1684 PCIe  | yolox_sail.pcie  |    yolox_s_fp32_1b.bmodel           |      0.400    |   0.594    |
| BM1684 PCIe  | yolox_sail.pcie  |    yolox_s_int8_1b.bmodel           |      0.396    |   0.587    |
| BM1684X PCIe | yolox_opencv.py  |    yolox_s_fp32_1b.bmodel           |      0.402    |   0.590    |
| BM1684X PCIe | yolox_opencv.py  |    yolox_s_fp16_1b.bmodel           |      0.402    |   0.590    |
| BM1684X PCIe | yolox_opencv.py  |    yolox_s_int8_1b.bmodel           |      0.402    |   0.587    |
| BM1684X PCIe | yolox_bmcv.py    |    yolox_s_fp32_1b.bmodel           |      0.402    |   0.590    |
| BM1684X PCIe | yolox_bmcv.py    |    yolox_s_fp16_1b.bmodel           |      0.402    |   0.590    |
| BM1684X PCIe | yolox_bmcv.py    |    yolox_s_int8_1b.bmodel           |      0.402    |   0.586    |
| BM1684X PCIe | yolox_bmcv.pcie  |    yolox_s_fp32_1b.bmodel           |      0.400    |   0.594    |
| BM1684X PCIe | yolox_bmcv.pcie  |    yolox_s_fp16_1b.bmodel           |      0.400    |   0.594    |
| BM1684X PCIe | yolox_bmcv.pcie  |    yolox_s_int8_1b.bmodel           |      0.401    |   0.592    |
| BM1684X PCIe | yolox_sail.pcie  |    yolox_s_fp32_1b.bmodel           |      0.400    |   0.594    |
| BM1684X PCIe | yolox_sail.pcie  |    yolox_s_fp16_1b.bmodel           |      0.400    |   0.594    |
| BM1684X PCIe | yolox_sail.pcie  |    yolox_s_int8_1b.bmodel           |      0.401    |   0.592    |
| BM1688 SoC   | yolox_opencv.py  |    yolox_s_fp32_1b.bmodel           |      0.403    |   0.590    |
| BM1688 SoC   | yolox_opencv.py  |    yolox_s_fp16_1b.bmodel           |      0.402    |   0.590    |
| BM1688 SoC   | yolox_opencv.py  |    yolox_s_int8_1b.bmodel           |      0.402    |   0.587    |
| BM1688 SoC   | yolox_bmcv.py    |    yolox_s_fp32_1b.bmodel           |      0.402    |   0.590    |
| BM1688 SoC   | yolox_bmcv.py    |    yolox_s_fp16_1b.bmodel           |      0.402    |   0.590    |
| BM1688 SoC   | yolox_bmcv.py    |    yolox_s_int8_1b.bmodel           |      0.402    |   0.587    |
| BM1688 SoC   | yolox_bmcv.soc   |    yolox_s_fp32_1b.bmodel           |      0.397    |   0.594    |
| BM1688 SoC   | yolox_bmcv.soc   |    yolox_s_fp16_1b.bmodel           |      0.396    |   0.594    |
| BM1688 SoC   | yolox_bmcv.soc   |    yolox_s_int8_1b.bmodel           |      0.398    |   0.592    |
| BM1688 SoC   | yolox_sail.soc   |    yolox_s_fp32_1b.bmodel           |      0.397    |   0.594    |
| BM1688 SoC   | yolox_sail.soc   |    yolox_s_fp16_1b.bmodel           |      0.396    |   0.594    |
| BM1688 SoC   | yolox_sail.soc   |    yolox_s_int8_1b.bmodel           |      0.398    |   0.592    |

> **Note**:
> 1. The same programs on SoC or PCIe have the same mAP, bmodel with batch_size=4/batch_size=1 have the same mAP, BM1688's 1/2 core bmodel have the same mAP.
> 2. AP@IoU=0.5:0.95 is the corresponding indicator of area=all.
> 3. Due to possible differences between SDK versions, it is normal for the mAP error of <1% between the actual running results and this table.

## 7. Performance Testing
### 7.1 bmrt_test
Use bmrt_test to test the theoretical performance of the model:
```bash
# Please modify the bmodel path and devid parameters to be tested according to the actual situation
bmrt_test --bmodel models/BM1684/yolox_s_fp32_1b.bmodel
```
The `calculate time` in the test results is the inference time of the model, and the theoretical inference time of each image is when the multi-batch size model is divided by the corresponding batch size.
The theoretical inference time of each model is tested, and the results are as follows:

|          Test model                 | calculate time(ms) |
| ------------------------------      | ----------------- |
| BM1684/yolox_s_fp32_1b.bmodel       |       26.01       |
| BM1684/yolox_s_fp32_4b.bmodel       |       25.62       |
| BM1684/yolox_s_int8_1b.bmodel       |       16.54       |
| BM1684/yolox_s_int8_4b.bmodel       |       11.72       |
| BM1684X/yolox_s_fp32_1b.bmodel      |       27.92       |
| BM1684X/yolox_s_fp32_4b.bmodel      |       25.63       |
| BM1684X/yolox_s_fp16_1b.bmodel      |       6.27        |
| BM1684X/yolox_s_fp16_4b.bmodel      |       6.15        |
| BM1684X/yolox_s_int8_1b.bmodel      |       3.86        |
| BM1684X/yolox_s_int8_4b.bmodel      |       3.69        |
| BM1688/yolox_s_fp32_1b.bmodel       |      155.60       |
| BM1688/yolox_s_fp16_1b.bmodel       |      36.11        |
| BM1688/yolox_s_int8_1b.bmodel       |      21.44        |
| BM1688/yolox_s_int8_4b.bmodel       |      20.40        |
| BM1688/yolox_s_fp32_1b_2core.bmodel |      104.13       |
| BM1688/yolox_s_fp16_1b_2core.bmodel |      23.58        |
| BM1688/yolox_s_int8_1b_2core.bmodel |      15.97        |
| BM1688/yolox_s_int8_4b_2core.bmodel |      11.92        |

> **Note**：  
> 1. The performance test results have a certain volatility.
> 2. The `calculate time` has been converted to the average inference time per picture.
> 3. The test results of SoC and PCIe are basically the same.

### 7.2 Program Performance
Refer to [C++ example](cpp/README_EN.md) or [Python example](python/README_EN.md) to run the program, and check the statistical decoding time, preprocessing time, inference time, post-processing time. The preprocessing time, inference time and post-processing time of C++ example printing are the whole batch processing time, which needs to be divided by the corresponding batch size to get the processing time of each picture.

Use different examples and models to test `datasets/coco/val2017_1000` with `conf_thresh=0.5, nms_thresh=0.5` on different test platforms. The performance test results are shown as follows:
|Test Platform|  Test Program    |             Test model              |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | --------  | ---------     | ---------    | ---------      |
| BM1684 SoC  | yolox_opencv.py  | yolox_s_fp32_1b.bmodel              | 15.18     | 3.63          | 39.70        | 2.71           |
| BM1684 SoC  | yolox_opencv.py  | yolox_s_int8_1b.bmodel              | 15.20     | 3.64          | 33.27        | 2.69           |
| BM1684 SoC  | yolox_opencv.py  | yolox_s_int8_4b.bmodel              | 15.20     | 5.54          | 22.66        | 2.36           |
| BM1684 SoC  | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel              | 3.70      | 2.88          | 28.16        | 2.72           |
| BM1684 SoC  | yolox_bmcv.py    | yolox_s_int8_1b.bmodel              | 3.50      | 2.22          | 21.62        | 2.71           |
| BM1684 SoC  | yolox_bmcv.py    | yolox_s_int8_4b.bmodel              | 3.38      | 2.06          | 10.68        | 2.35           |
| BM1684 SoC  | yolox_bmcv.soc   | yolox_s_fp32_1b.bmodel              | 4.86      | 1.47          | 25.78        | 2.68           |
| BM1684 SoC  | yolox_bmcv.soc   | yolox_s_int8_1b.bmodel              | 4.57      | 1.46          | 19.44        | 2.68           |
| BM1684 SoC  | yolox_bmcv.soc   | yolox_s_int8_4b.bmodel              | 4.76      | 1.41          | 8.96         | 2.07           |
| BM1684 SoC  | yolox_sail.soc   | yolox_s_fp32_1b.bmodel              | 3.22      | 3.11          | 26.16        | 2.08           |
| BM1684 SoC  | yolox_sail.soc   | yolox_s_int8_1b.bmodel              | 3.21      | 3.11          | 19.83        | 2.07           |
| BM1684 SoC  | yolox_sail.soc   | yolox_s_int8_4b.bmodel              | 3.14      | 2.73          | 9.26         | 2.10           |
| BM1684X SoC | yolox_opencv.py  | yolox_s_fp32_1b.bmodel              | 13.87     | 3.40          | 44.02        | 2.84           |
| BM1684X SoC | yolox_opencv.py  | yolox_s_fp16_1b.bmodel              | 13.88     | 3.24          | 22.04        | 2.84           |
| BM1684X SoC | yolox_opencv.py  | yolox_s_int8_1b.bmodel              | 13.98     | 3.49          | 20.94        | 2.82           |
| BM1684X SoC | yolox_opencv.py  | yolox_s_int8_4b.bmodel              | 13.90     | 5.17          | 20.87        | 2.52           |
| BM1684X SoC | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel              | 3.22      | 2.40          | 30.33        | 2.86           |
| BM1684X SoC | yolox_bmcv.py    | yolox_s_fp16_1b.bmodel              | 3.20      | 2.39          | 7.93         | 2.86           |
| BM1684X SoC | yolox_bmcv.py    | yolox_s_int8_1b.bmodel              | 3.19      | 2.38          | 6.81         | 2.87           |
| BM1684X SoC | yolox_bmcv.py    | yolox_s_int8_4b.bmodel              | 3.03      | 2.19          | 6.18         | 2.51           |
| BM1684X SoC | yolox_bmcv.soc   | yolox_s_fp32_1b.bmodel              | 4.51      | 0.75          | 27.86        | 2.72           |
| BM1684X SoC | yolox_bmcv.soc   | yolox_s_fp16_1b.bmodel              | 4.55      | 0.75          | 5.49         | 2.75           |
| BM1684X SoC | yolox_bmcv.soc   | yolox_s_int8_1b.bmodel              | 4.51      | 0.75          | 4.35         | 2.74           |
| BM1684X SoC | yolox_bmcv.soc   | yolox_s_int8_4b.bmodel              | 4.28      | 0.72          | 4.26         | 2.73           |
| BM1684X SoC | yolox_sail.soc   | yolox_s_fp32_1b.bmodel              | 2.92      | 2.72          | 28.29        | 2.12           |
| BM1684X SoC | yolox_sail.soc   | yolox_s_fp16_1b.bmodel              | 2.88      | 2.71          | 5.91         | 2.10           |
| BM1684X SoC | yolox_sail.soc   | yolox_s_int8_1b.bmodel              | 2.81      | 2.72          | 4.76         | 2.10           |
| BM1684X SoC | yolox_sail.soc   | yolox_s_int8_4b.bmodel              | 2.67      | 2.63          | 4.55         | 2.12           |
| BM1688 SoC  | yolox_opencv.py  | yolox_s_fp32_1b.bmodel              |  21.62    | 4.17          | 174.24       | 3.98           |
| BM1688 SoC  | yolox_opencv.py  | yolox_s_fp16_1b.bmodel              |  20.42    | 4.09          | 54.62        | 3.98           |
| BM1688 SoC  | yolox_opencv.py  | yolox_s_int8_1b.bmodel              |  20.19    | 4.14          | 40.05        | 3.97           |
| BM1688 SoC  | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel              |  4.68     | 5.18          | 157.73       | 4.01           |
| BM1688 SoC  | yolox_bmcv.py    | yolox_s_fp16_1b.bmodel              |  5.16     | 5.25          | 38.22        | 4.03           |
| BM1688 SoC  | yolox_bmcv.py    | yolox_s_int8_1b.bmodel              |  4.53     | 5.18          | 23.63        | 3.99           |
| BM1688 SoC  | yolox_bmcv.soc   | yolox_s_fp32_1b.bmodel              |  5.84     | 1.96          | 154.59       | 3.76           |
| BM1688 SoC  | yolox_bmcv.soc   | yolox_s_fp16_1b.bmodel              |  5.83     | 1.94          | 35.05        | 3.75           |
| BM1688 SoC  | yolox_bmcv.soc   | yolox_s_int8_1b.bmodel              |  5.78     | 1.95          | 20.42        | 3.74           |
| BM1688 SoC  | yolox_sail.soc   | yolox_s_fp32_1b.bmodel              |  3.94     | 5.22          | 155.22       | 2.91           |
| BM1688 SoC  | yolox_sail.soc   | yolox_s_fp16_1b.bmodel              |  3.92     | 5.21          | 35.65        | 2.91           |
| BM1688 SoC  | yolox_sail.soc   | yolox_s_int8_1b.bmodel              |  4.01     | 5.23          | 21.04        | 2.91           |

> **Note**：  
> 1. The time units are all milliseconds (ms), and the statistical time is the average processing time of each image.
> 2. The performance test results are volatile to a certain extent, so it is recommended that the average value should be taken from multiple tests.
> 3. test platform of BM1684 SoC is standard SE5, and test platform of BM1684X SoC is standard SE7
> 4. BM1684/1684X SoC's processors are all 8-core ARM A53 42320 DMIPS @ 2.3GHz, performance on PCIe may vary greatly due to different processors.
> 5. The image resolution has a great influence on the decoding time, the reasoning result has a great influence on the post-processing time, different test pictures may be different, and different thresholds have a great influence on the post-processing time.
> 6. The performance of programs which using BM1688's 1/2 core bmodel has only difference in inference_time, please refer to the test data in [Section 7.1](#71-bmrt_test) for inference performance differences. 
> 7. `yolox_opencv.py`'s decode_time is based on official opencv.

## 8. FAQ
[Frequently Asked Questions](../../docs/FAQ_EN.md)

## 9. Acknowledgments
* Thanks to "Ling Yun Zhi Xin" for optimizing YOLOX's python demo