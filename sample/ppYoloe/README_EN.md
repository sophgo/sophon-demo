[简体中文](./README.md) | [English](./README_EN.md)

# ppYoloe

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
PPYOLOe is a state-of-the-art single-stage anchor-free model, built upon PP-YOLOv2, pioneered by Baidu, surpassing various popular YOLO models.

**URL of the paper** (https://arxiv.org/pdf/2203.16250.pdf)

**URL of github repository** (https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyoloe)

## 2. Feature
* Support for BM1684X(x86 PCIe、SoC) and BM1684(x86 PCIe、SoC、arm PCIe)
* Support for FP32, FP16 (BM1684X) model compilation and inference
* Support C++ inference based on BMCV preprocessing
* Support Python inference based on OpenCV and BMCV preprocessing
* Support single batch and multi-batch model inference
* Support for picture and video testing

## 3. Prepare Models and Data

It is recommended to use TPU-MLIR to compile BModel, Pytorch model to export to onnx model before compilation. For detail, please see https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/EXPORT_ONNX_MODEL.md and https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/end2end_ppyoloe/README.md

The model used in this demo is the official [ppyoloe_crn_s_400e_coco](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_400e_coco.pdparams). Due to the complexity of the post-processing in the model, when compiling this model into a BModel using TPU-MLIR, two outputs related to the bounding boxes in the model are specified, namely, `p2o.Div.1` and `p2o.Concat.29`. The post-processing part is implemented in the deployment program. If you want to add reference inputs to verify the correctness of the Bmodel model conversion, you can refer to the [document](./docs/prepare_npz.md).

The old version of the compilation toolchain TPU-NNTC is slow in terms of updates and maintenance, and it is not recommended for your compilation and usage.

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
│   └── ppyoloe_fp32_1b.bmodel   # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 for BM1684
├── BM1684X
│   ├── ppyoloe_fp32_1b.bmodel   # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 for BM1684X
│   └── ppyoloe_fp16_1b.bmodel   # Compiled with TPU-MLIR, FP16 BModel,batch_size=1 for BM1684X
└── onnx
    └── yolox.onnx               # Derived onnx dynamic model      
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

Executing the above command will generate the `ppyoloe_fp32_1b.bmodel` file under `models/BM1684 or models/BM1684X/`, that is, the converted FP32 BModel.

- GenerateFP16 BModel

This example provides a script for TPU-MLIR to compile FP16 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_fp16bmodel_mlir.sh`, and specify the target platform on which BModel runs (**BM1684X is supported**) during execution, such as:

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

Executing the above command will generate the `ppyoloe_fp16_1b.bmodel` file under `models/BM1684X/`, that is, the converted FP16 BModel.


## 5. Example Test
- [C++ Example](./cpp/README.md)
- [Python Example](./python/README.md)

## 6. Precision Test
### 6.1 Testing Method

First of all, refer to [C++ example](cpp/README_EN.md#32-image-test-demo) or [Python example](python/README_EN.md#22-image-test-demo) to deduce the dataset to be tested, generate the predicted json file, and pay attention to modifying the dataset (datasets/coco/val2017_1000) and related parameters (conf_thresh=0.4, nms_thresh=0.6). 
Then, use the `eval_coco.py` script under the `tools` directory to compare the json file generated by the test with the test set tag json file, and calculate the evaluation metrics for target detection. The command is as follows:

```bash
# Install pycocotools, skip if it is already installed
pip3 install pycocotools
# Please modify the program path and json file path according to the actual situation
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/ppyoloe_fp32_1b.bmodel_val2017_1000_bmcv_python_result.json
```

### 6.2 Test Result
On the coco2017val_1000 dataset, the accuracy test results are as follows:
| Test Platform|   Test Program  |             Test model              |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ----------------| ----------------------------------- | ------------- | -------- |
| BM1684 PCIe   | ppyoloe_opencv.py | ppyoloe_fp32_1b.bmodel | 0.377           | 0.508      |
| BM1684 PCIe   | ppyoloe_bmcv.py   | ppyoloe_fp32_1b.bmodel | 0.380           | 0.513      |
| BM1684 PCIe   | ppyoloe_bmcv.pcie | ppyoloe_fp32_1b.bmodel | 0.378           | 0.510      |
| BM1684 PCIe   | ppyoloe_sail.pcie | ppyoloe_fp32_1b.bmodel | 0.378           | 0.510      |
| BM1684X PCIe  | ppyoloe_opencv.py | ppyoloe_fp32_1b.bmodel | 0.377           | 0.508      |
| BM1684X PCIe  | ppyoloe_opencv.py | ppyoloe_fp16_1b.bmodel | 0.377           | 0.508      |
| BM1684X PCIe  | ppyoloe_bmcv.py   | ppyoloe_fp32_1b.bmodel | 0.380           | 0.513      |
| BM1684X PCIe  | ppyoloe_bmcv.py   | ppyoloe_fp16_1b.bmodel | 0.380           | 0.513      |
| BM1684X PCIe  | ppyoloe_bmcv.pcie | ppyoloe_fp32_1b.bmodel | 0.379           | 0.510      |
| BM1684X PCIe  | ppyoloe_bmcv.pcie | ppyoloe_fp16_1b.bmodel | 0.378           | 0.510      |
| BM1684X PCIe  | ppyoloe_sail.pcie | ppyoloe_fp32_1b.bmodel | 0.379           | 0.510      |
| BM1684X PCIe  | ppyoloe_sail.pcie | ppyoloe_fp16_1b.bmodel | 0.378           | 0.510      |

> **Test Description**:
> 
> 1. The model accuracy of SoC and PCIe is the same.
> 2. AP@IoU=0.5:0.95 is the corresponding indicator of area=all.

## 7. Performance Testing
### 7.1 bmrt_test
Use bmrt_test to test the theoretical performance of the model:
```bash
# Please modify the bmodel path and devid parameters to be tested according to the actual situation
bmrt_test --bmodel models/BM1684/ppyoloe_fp32_1b.bmodel
```
The `calculate time` in the test results is the inference time of the model, and the theoretical inference time of each image is when the multi-batch size model is divided by the corresponding batch size.
The theoretical inference time of each model is tested, and the results are as follows:

|          Test model            | calculate time(ms) |
| ------------------------------ | ----------------- |
| BM1684/ppyoloe_fp32_1b.bmodel  |       26.01       |
| BM1684/ppyoloe_fp32_4b.bmodel  |       25.62       |
| BM1684X/ppyoloe_fp32_1b.bmodel |       35.80       |
| BM1684X/ppyoloe_fp32_4b.bmodel |       35.15       |
| BM1684X/ppyoloe_fp16_1b.bmodel |       10.12       |
| BM1684X/ppyoloe_fp16_4b.bmodel |       8.90        |

> **Test Description**：  
> 1. The performance test results have a certain volatility.
> 2. The `calculate time` has been converted to the average inference time per picture.
> 3. The test results of SoC and PCIe are basically the same.

### 7.2 Program Performance
Refer to [C++ example](cpp/README_EN.md) or [Python example](python/README_EN.md) to run the program, and check the statistical decoding time, preprocessing time, inference time, post-processing time. The preprocessing time, inference time and post-processing time of C++ example printing are the whole batch processing time, which needs to be divided by the corresponding batch size to get the processing time of each picture.

Use different examples and models to test `datasets/val2017_1000` with `conf_thresh=0.4,nms_thresh=0.6` on different test platforms. The performance test results are shown as follows:
|Test Platform|  Test Program    |             Test model              |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
| BM1684 SoC  | yolox_opencv.py | ppyoloe_fp32_1b.bmodel | 15.19       | 45.22           | 45.78          | 12.86            |
| BM1684 SoC  | yolox_bmcv.py   | ppyoloe_fp32_1b.bmodel | 6.82        | 3.58            | 33.72          | 12.83            |
| BM1684 SoC  | yolox_bmcv.soc  | ppyoloe_fp32_1b.bmodel | 5.02        | 1.72            | 30.78          | 16.91            |
| BM1684 SoC  | ppyoloeail.soc  | ppyoloe_fp32_1b.bmodel | 3.22        | 4.19            | 31.18          | 7.99             |
| BM1684X SoC | yolox_opencv.py | ppyoloe_fp32_1b.bmodel | 3.37        | 41.60           | 43.79          | 12.75            |
| BM1684X SoC | yolox_opencv.py | ppyoloe_fp16_1b.bmodel | 3.21        | 40.63           | 24.48          | 12.62            |
| BM1684X SoC | yolox_bmcv.py   | ppyoloe_fp32_1b.bmodel | 3.05        | 2.68            | 30.28          | 13.09            |
| BM1684X SoC | yolox_bmcv.py   | ppyoloe_fp16_1b.bmodel | 3.07        | 2.68            | 11.09          | 13.10            |
| BM1684X SoC | yolox_bmcv.soc  | ppyoloe_fp32_1b.bmodel | 4.42        | 0.98            | 27.03          | 8.64             |
| BM1684X SoC | yolox_bmcv.soc  | ppyoloe_fp16_1b.bmodel | 4.48        | 1.00            | 7.88           | 8.69             |
| BM1684X SoC | ppyoloeail.soc  | ppyoloe_fp32_1b.bmodel | 2.78        | 3.28            | 27.44          | 8.04             |
| BM1684X SoC | ppyoloeail.soc  | ppyoloe_fp16_1b.bmodel | 2.69        | 3.28            | 8.20           | 8.06             |

> **Test Description**：  
> 1. The time units are all milliseconds (ms), and the statistical time is the average processing time of each image.
> 2. The performance test results are volatile to a certain extent, so it is recommended that the average value should be taken from multiple tests.
> 3. test platform of BM1684 SoC is standard SE5, and test platform of BM1684X SoC is standard SE7
> 4. BM1684/1684X SoC's master CPU are all 8-core ARM A53 42320 DMIPS @ 2.3GHz CPU performance on PCIe may vary greatly due to different PCIes.
> 5. The image resolution has a great influence on the decoding time, the reasoning result has a great influence on the post-processing time, different test pictures may be different, and different thresholds have a great influence on the post-processing time.

## 8. FAQ
[Frequently Asked Questions](../../docs/FAQ_EN.md)
