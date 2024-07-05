[简体中文](./README.md) | [English](./README_EN.md)

# YOLOv5

## Catalogue

* [1. Introduction](#1-introduction)
* [2. Characteristics](#2-characteristics)
  * [2.1 SDK Characteristics](#21-sdk-characteristics)
  * [2.2 Algorithm Characteristics](#22-algorithm-characteristics)
* [3. Prepare Models and Data](#3-prepare-models-and-data)
* [4. Model Compilation](#4-model-compilation)
* [5. Example Test](#5-example-test)
* [6. Precision Test](#6-precision-test)
  * [6.1 Testing Method](#61-testing-method)
  * [6.2 Test Result](#62-test-result)
* [7. Performance Testing](#7-performance-testing)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 Program Performance](#72-program-performance)
* [8. YOLOv5 cpu opt](#8-yolov5-cpu-opt)
  * [8.1 NMS Optimization Item](#81-nms-optimization-item)
  * [8.2 Precision Test](#82-precision-test)
  * [8.3 Performance Test](#83-performance-test)
* [9. FAQ](#9-faq)
  
## 1. Introduction
YOLOv5 is a very classical One Stage target detection algorithm based on anchor. Because of its excellent accuracy and speed performance, it has been widely used in engineering practice. This example [​YOLOv5 official open source repository](https://github.com/ultralytics/yolov5) transplants the v6.1 version of the model and algorithm so that it can be inference tested on SOPHON BM1684/BM1684X/BM1688/CV186X.

## 2. Characteristics
### 2.1 SDK Characteristics
* Support for BM1688/CV186X(SoC), BM1684X(x86 PCIe、SoC、riscv PCIe), BM1684(x86 PCIe、SoC、arm PCIe)
* Support for FP32, FP16 (BM1684X/BM1688/CV186X), INT8 model compilation and inference
* Support C++ inference based on BMCV preprocessing
* Support Python inference based on OpenCV and BMCV preprocessing
* Support single batch and multi-batch model inference
* Support 1 output and 3 output model inference
* Support for picture and video testing
* Support NMS postprocessing acceleration

### 2.2 Algorithm Characteristics
PYTHON CODE
1. `sophon-demo/sample/YOLOv5/python`: Source code is available in this repository, implementing simple demos based on both bmcv and opencv interfaces. These are used for testing model accuracy and are **not recommended for performance evaluation**.
2. `sophon-demo/sample/YOLOv5_opt/python`: Source code is available in this repository, supporting only in `1684x`. It implements YOLOv5 decoding layers and NMS operations using TPU to enhance end-to-end performance.
3. `sophon-sail/sample/python/yolov5_multi_3output_pic.py`: Source code resides in the SDK's sophon-sail folder. It utilizes Python to call C++ encapsulated interfaces, allocating decoding, pre-processing, inference, and post-processing in separate threads to improve overall performance.
4. Sections 8 and 9 of the "TPU-MLIR Quick Start Guide" involve leveraging TPU for pre-processing and post-processing. Please refer to the corresponding documentation and develop corresponding routines to integrate pre-processing and post-processing into the algorithm, thereby improving end-to-end performance.
5. Utilize the Python `multiprocessing` module to invoke time-consuming function via multiple processes, enhancing overall throughput.

C++ CODE
1. `sophon-demo/sample/YOLOv5/cpp`: Source code is available in this repository, demonstrating simple demos based on bmcv and sail interfaces for accuracy verification.
2. `sophon-demo/sample/YOLOv5_opt/cpp`: Source code in this repository supports only in `1684x`. It utilizes TPU for YOLOv5 decoding layers and NMS operations, improving end-to-end performance.
3. `sophon-stream/samples/yolov5`: Source code is in the SDK (version V23.10.01 and above) under `sophon-stream`, separating pre-processing, inference, and post-processing into different threads to significantly improve overall performance.
4. `sophon-pipeline/examples/yolov5`: [Source code](https://github.com/sophgo/sophon-pipeline) , implementing the entire algorithmic inference process based on a thread pool to enhance overall performance.
5. Sections 8 and 9 of the "TPU-MLIR Quick Start Guide" involve leveraging TPU for pre-processing and post-processing, thereby improving end-to-end performance.

> **note**  
> This code supports both triple-output and single-output models. The single-output model demonstrates higher performance, but it might encounter issues during quantization. On the other hand, the triple-output model simplifies quantization. **When used for model accuracy validation, it is recommended to opt for the triple-output model.**
 
## 3. Prepare Models and Data
It is recommended to use TPU-MLIR to compile BModel, Pytorch model to export to onnx model before compilation, if the tpu-mlir version you are using is >= v1.3.0 (i.e. official website v23.07.01), you can use the torchscript model directly. For more information, please see [YOLOv5 Model Export](./docs/YOLOv5_Export_Guide_EN.md).

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
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel       # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 for BM1684
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel       # Compiled with TPU-MLIR, INT8 BModel,batch_size=1 for BM1684
│   └── yolov5s_v6.1_3output_int8_4b.bmodel       # Compiled with TPU-MLIR, INT8 BModel,batch_size=4 for BM1684
├── BM1684X    
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel       # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 for BM1684X
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel       # Compiled with TPU-MLIR, FP16 BModel,batch_size=1 for BM1684X
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel       # Compiled with TPU-MLIR, INT8 BModel,batch_size=1 for BM1684X
│   └── yolov5s_v6.1_3output_int8_4b.bmodel       # Compiled with TPU-MLIR, INT8 BModel,batch_size=4 for BM1684X
├── BM1688
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel       # Compiled with TPU-MLIR, FP32 BModel,batch_size=1,num_core=1 for BM1688
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel       # Compiled with TPU-MLIR, FP32 BModel,batch_size=1,num_core=1 for BM1688
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel       # Compiled with TPU-MLIR, FP32 BModel,batch_size=1,num_core=1 for BM1688
│   ├── yolov5s_v6.1_3output_int8_4b.bmodel       # Compiled with TPU-MLIR, FP32 BModel,batch_size=1,num_core=1 for BM1688
│   ├── yolov5s_v6.1_3output_fp16_1b_2core.bmodel # Compiled with TPU-MLIR, FP32 BModel,batch_size=1,num_core=2 for BM1688
│   ├── yolov5s_v6.1_3output_fp32_1b_2core.bmodel # Compiled with TPU-MLIR, FP32 BModel,batch_size=1,num_core=2 for BM1688
│   ├── yolov5s_v6.1_3output_int8_1b_2core.bmodel # Compiled with TPU-MLIR, FP32 BModel,batch_size=1,num_core=2 for BM1688
│   └── yolov5s_v6.1_3output_int8_4b_2core.bmodel # Compiled with TPU-MLIR, FP32 BModel,batch_size=1,num_core=2 for BM1688
├── CV186X
│   ├── yolov5s_v6.1_3output_fp16_1b.bmodel       # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 for CV186X
│   ├── yolov5s_v6.1_3output_fp32_1b.bmodel       # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 for CV186X
│   ├── yolov5s_v6.1_3output_int8_1b.bmodel       # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 for CV186X
│   └── yolov5s_v6.1_3output_int8_4b.bmodel       # Compiled with TPU-MLIR, FP32 BModel,batch_size=1 for CV186X
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

You need to install TPU-MLIR before compiling the model. For more information, please see [TPU-MLIR Environment Building](../../docs/Environment_Install_Guide_EN.md#1-tpu-mlir-environmental-installation). After installation, you need to enter the example directory in the TPU-MLIR environment. Use TPU-MLIR to compile the onnx model to BModel. For specific methods, please refer to "chapter 3.5" of the TPU-MLIR Quick start Manual. Compile the ONNX model (please obtain it from the corresponding version of SDK of [Sophgo official website](https://developer.sophgo.com/site/index.html?categoryActive=material)).

- Generate FP32 BModel

This example provides a script for TPU-MLIR to compile FP32 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_fp32bmodel_mlir.sh`, and specify the target platform on which BModel runs (**BM1684/BM1684X/BM1688/CV186X is supported**) during execution, such as:

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

Executing the above command will generate the `yolov5s_v6.1_3output_fp32_1b.bmodel` file under a folder like `models/BM1684`, that is, the converted FP32 BModel.

- Generate FP16 BModel

This example provides a script for TPU-MLIR to compile FP16 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_fp16bmodel_mlir.sh`, and specify the target platform on which BModel runs (**BM1684X/BM1688/CV186X is supported**) during execution, such as:

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

Executing the above command will generate the `yolov5s_v6.1_3output_fp16_1b.bmodel` file under a folder like`models/BM1684X/`, that is, the converted FP16 BModel.

- Generate INT8 BModel

This example provides a script for quantifying INT8 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_int8bmodel_mlir.sh`, and enter the target platform of BModel (**BM1684/BM1684X/BM1688/CV186X is supported**) during execution, such as:

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688/cv186x
```

The above script will generate files such as `yolov5s_v6.1_3output_int8_1b.bmodel` under a folder like `models/BM1684`, that is, the converted INT8 BModel.

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
CPP set `--use_cpu_opt=false` or Python not set `--use_cpu_opt` for testing. On the coco2017val_1000 dataset, the accuracy test results are as follows:
| Test Platform |  Test Program    |            Test model               |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------  | ---------------- | ----------------------------------- | ------------- | -------- |
| SE5-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.377 |    0.580 |
| SE5-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.344 |    0.553 |
| SE5-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.344 |    0.553 |
| SE5-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.373 |    0.573 |
| SE5-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.337 |    0.544 |
| SE5-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.337 |    0.544 |
| SE5-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.375 |    0.572 |
| SE5-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.338 |    0.544 |
| SE5-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.338 |    0.544 |
| SE5-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.375 |    0.572 |
| SE5-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.338 |    0.544 |
| SE5-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.338 |    0.544 |
| SE7-32       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.377 |    0.580 |
| SE7-32       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.377 |    0.580 |
| SE7-32       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.363 |    0.572 |
| SE7-32       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.363 |    0.572 |
| SE7-32       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.373 |    0.573 |
| SE7-32       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.373 |    0.573 |
| SE7-32       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.356 |    0.563 |
| SE7-32       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.356 |    0.563 |
| SE7-32       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.374 |    0.572 |
| SE7-32       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.374 |    0.572 |
| SE7-32       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.357 |    0.562 |
| SE7-32       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.357 |    0.562 |
| SE7-32       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.374 |    0.572 |
| SE7-32       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.374 |    0.572 |
| SE7-32       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.357 |    0.562 |
| SE7-32       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.357 |    0.562 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.377 |    0.580 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.377 |    0.580 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.358 |    0.567 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.358 |    0.567 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.373 |    0.573 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.373 |    0.573 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.355 |    0.565 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.355 |    0.565 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.374 |    0.573 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.374 |    0.572 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.354 |    0.565 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.354 |    0.565 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.374 |    0.573 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.374 |    0.572 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.354 |    0.565 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.354 |    0.565 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp32_1b_2core.bmodel |    0.377 |    0.580 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_fp16_1b_2core.bmodel |    0.377 |    0.580 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_1b_2core.bmodel |    0.358 |    0.567 |
| SE9-16       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_4b_2core.bmodel |    0.358 |    0.567 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp32_1b_2core.bmodel |    0.373 |    0.573 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp16_1b_2core.bmodel |    0.373 |    0.573 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_1b_2core.bmodel |    0.355 |    0.565 |
| SE9-16       | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_4b_2core.bmodel |    0.355 |    0.565 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp32_1b_2core.bmodel |    0.374 |    0.573 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp16_1b_2core.bmodel |    0.374 |    0.572 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_1b_2core.bmodel |    0.354 |    0.565 |
| SE9-16       | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_4b_2core.bmodel |    0.354 |    0.565 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp32_1b_2core.bmodel |    0.374 |    0.573 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_fp16_1b_2core.bmodel |    0.374 |    0.572 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_1b_2core.bmodel |    0.354 |    0.565 |
| SE9-16       | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_4b_2core.bmodel |    0.354 |    0.565 |
| SE9-8        | yolov5_opencv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.377 |    0.580 |
| SE9-8        | yolov5_opencv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.377 |    0.580 |
| SE9-8        | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.358 |    0.567 |
| SE9-8        | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.358 |    0.567 |
| SE9-8        | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.373 |    0.573 |
| SE9-8        | yolov5_bmcv.py     | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.373 |    0.573 |
| SE9-8        | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.355 |    0.565 |
| SE9-8        | yolov5_bmcv.py     | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.355 |    0.565 |
| SE9-8        | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.374 |    0.572 |
| SE9-8        | yolov5_bmcv.soc    | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.374 |    0.572 |
| SE9-8        | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.354 |    0.564 |
| SE9-8        | yolov5_bmcv.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.354 |    0.564 |
| SE9-8        | yolov5_sail.soc    | yolov5s_v6.1_3output_fp32_1b.bmodel      |    0.374 |    0.572 |
| SE9-8        | yolov5_sail.soc    | yolov5s_v6.1_3output_fp16_1b.bmodel      |    0.374 |    0.572 |
| SE9-8        | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.354 |    0.564 |
| SE9-8        | yolov5_sail.soc    | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.354 |    0.564 |
> **Note**:
> 1. The model mAP of batch_size=4 and batch_size=1 is the same.
> 2. Due to possible differences between SDK versions, it is normal for the mAP error of <0.01 between the actual running results and this table;
> 3. AP@IoU=0.5:0.95 is the corresponding indicator of area=all.
> 4. On a PCIe or SoC platform equipped with the same TPU and SOPHONSDK, the mAP of the same program is the same, SE5 series corresponds to BM1684, SE7 series corresponds to BM1684X. In SE9 series, SE9-16 corresponds to BM1688, SE9-8 corresponds to CV186X;

## 7. Performance Testing
### 7.1 bmrt_test
Use bmrt_test to test the theoretical performance of the model:
```bash
# Please modify the bmodel path and devid parameters to be tested according to the actual situation
bmrt_test --bmodel models/BM1684/yolov5s_v6.1_3output_fp32_1b.bmodel
```
The `calculate time` in the test results is the inference time of the model, and the theoretical inference time of each image is when the multi-batch size model is divided by the corresponding batch size.
The theoretical inference time of each model is tested, and the results are as follows:

|                  Test model                       | calculate time(ms) |
| -------------------------------------------       | ----------------- |
| BM1684/yolov5s_v6.1_3output_fp32_1b.bmodel |          22.41  |
| BM1684/yolov5s_v6.1_3output_int8_1b.bmodel |          11.26  |
| BM1684/yolov5s_v6.1_3output_int8_4b.bmodel |           6.04  |
| BM1684X/yolov5s_v6.1_3output_fp32_1b.bmodel|          21.66  |
| BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel|           7.37  |
| BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel|           3.51  |
| BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel|           3.34  |
| BM1688/yolov5s_v6.1_3output_fp32_1b.bmodel|         101.57  |
| BM1688/yolov5s_v6.1_3output_fp16_1b.bmodel|          29.92  |
| BM1688/yolov5s_v6.1_3output_int8_1b.bmodel|           9.33  |
| BM1688/yolov5s_v6.1_3output_int8_4b.bmodel|           8.90  |
| BM1688/yolov5s_v6.1_3output_fp32_1b_2core.bmodel|          66.89  |
| BM1688/yolov5s_v6.1_3output_fp16_1b_2core.bmodel|          20.62  |
| BM1688/yolov5s_v6.1_3output_int8_1b_2core.bmodel|           8.53  |
| BM1688/yolov5s_v6.1_3output_int8_4b_2core.bmodel|           6.87  |
| CV186X/yolov5s_v6.1_3output_fp32_1b.bmodel|         100.68  |
| CV186X/yolov5s_v6.1_3output_fp16_1b.bmodel|          29.93  |
| CV186X/yolov5s_v6.1_3output_int8_1b.bmodel|           8.18  |
| CV186X/yolov5s_v6.1_3output_int8_4b.bmodel|           7.90  |

> **Note**：  
> 1. The performance test results have a certain volatility.
> 2. The `calculate time` has been converted to the average inference time per picture.
> 3. The test results of SoC and PCIe are basically the same.

### 7.2 Program Performance
Refer to [C++ example](cpp/README_EN.md) or [Python example](python/README_EN.md) to run the program, and check the statistical decoding time, preprocessing time, inference time, post-processing time. The time info which C++/Python example prints have already been converted into processing time per image.

CPP set `--use_cpu_opt=false` or Python not set `--use_cpu_opt` for testing. Use different examples and models to test `datasets/coco/val2017_1000` with `conf_thresh=0.5,nms_thresh=0.5` on different test platforms. The performance test results are shown as follows:
|Test Platform|  Test Program    |            Test model               |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
|   SE5-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      15.08      |      21.95      |      31.40      |     107.61      |
|   SE5-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      15.07      |      26.16      |      34.45      |     110.48      |
|   SE5-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      15.03      |      23.94      |      27.53      |     111.78      |
|   SE5-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.61       |      2.83       |      29.06      |     106.85      |
|   SE5-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      3.59       |      2.31       |      17.92      |     106.40      |
|   SE5-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      3.44       |      2.13       |      11.82      |     110.71      |
|   SE5-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.87       |      1.54       |      22.33      |      15.68      |
|   SE5-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      4.85       |      1.53       |      11.20      |      15.66      |
|   SE5-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      4.75       |      1.47       |      6.03       |      15.64      |
|   SE5-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.23       |      3.04       |      23.31      |      14.07      |
|   SE5-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.22       |      1.80       |      12.21      |      13.93      |
|   SE5-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.08       |      1.71       |      6.88       |      13.80      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      15.09      |      27.82      |      33.27      |     108.98      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      15.01      |      27.27      |      19.10      |     109.18      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      15.08      |      27.02      |      15.18      |     109.33      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      14.99      |      25.01      |      13.31      |     108.20      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.09       |      2.35       |      28.98      |     103.87      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      3.09       |      2.34       |      14.75      |     103.75      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      3.08       |      2.34       |      10.92      |     103.89      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      2.93       |      2.16       |      9.82       |     108.36      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.32       |      0.74       |      21.63      |      15.91      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.32       |      0.74       |      7.38       |      15.94      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      4.33       |      0.74       |      3.48       |      15.94      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      4.17       |      0.71       |      3.32       |      15.73      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      2.71       |      2.58       |      22.61      |      14.15      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      2.71       |      2.59       |      8.35       |      14.19      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      2.70       |      2.59       |      4.45       |      14.18      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      2.56       |      2.50       |      4.20       |      14.06      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      23.43      |      35.69      |     112.71      |     151.54      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      19.48      |      36.12      |      42.15      |     149.94      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      19.24      |      34.85      |      21.46      |     148.00      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      19.21      |      33.15      |      19.40      |     150.54      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.36       |      5.04       |     107.62      |     143.80      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.35       |      5.06       |      36.97      |     143.51      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      4.36       |      5.04       |      16.47      |     143.20      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      4.22       |      4.74       |      14.96      |     149.54      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      5.69       |      1.88       |      97.91      |      22.33      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      5.80       |      1.88       |      27.36      |      22.33      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      5.75       |      1.88       |      7.12       |      22.40      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      5.57       |      1.79       |      7.02       |      22.07      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.76       |      5.06       |     100.46      |      19.93      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      3.81       |      5.05       |      29.87      |      19.84      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.74       |      5.04       |      9.60       |      19.91      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.69       |      4.82       |      9.32       |      19.73      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      19.35      |      35.64      |      78.13      |     150.70      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      19.32      |      35.89      |      32.72      |     150.59      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      19.22      |      36.24      |      21.08      |     148.49      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      19.28      |      32.70      |      17.31      |     150.59      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      4.36       |      5.05       |      72.89      |     143.53      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      4.38       |      5.08       |      27.64      |     143.77      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      4.38       |      5.06       |      15.90      |     143.50      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      4.21       |      4.74       |      13.45      |     149.79      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      5.79       |      1.87       |      63.28      |      22.35      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      5.78       |      1.88       |      18.07      |      22.35      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      5.79       |      1.88       |      6.32       |      22.37      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      5.61       |      1.79       |      4.99       |      22.12      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      3.83       |      5.06       |      65.81      |      19.84      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      3.80       |      5.05       |      20.57      |      19.86      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      3.79       |      5.05       |      8.81       |      19.88      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      3.61       |      4.81       |      7.27       |      19.80      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      32.30      |      36.30      |     115.17      |     150.69      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      22.49      |      36.42      |      44.46      |     150.59      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      19.31      |      36.46      |      22.61      |     150.61      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      22.00      |      32.78      |      20.00      |     150.11      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      6.22       |      5.13       |     109.91      |     143.27      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.24       |      5.15       |      39.16      |     143.25      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      4.19       |      5.14       |      17.52      |     143.37      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      4.07       |      4.79       |      16.06      |     149.06      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      5.66       |      2.14       |     100.47      |      22.23      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      5.72       |      2.15       |      29.81      |      22.21      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      5.72       |      2.14       |      8.05       |      22.23      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      5.89       |      2.04       |      7.88       |      21.98      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.68       |      5.21       |     103.01      |      19.75      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      3.73       |      5.16       |      32.29      |      19.73      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.69       |      5.15       |      10.51      |      19.76      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.52       |      4.96       |      10.17      |      19.59      |

> **Note**：  
> 1. The time units are all milliseconds (ms), and the statistical time is the average processing time of each image.
> 2. The performance test results are volatile to a certain extent, so it is recommended that the average value should be taken from multiple tests.
> 3. SE5-16/SE7-32's processors are all 8-cores ARM CA53@2.3GHz, SE9-16's processor is 8-cores ARM CA53@1.6GHz and SE9-8 use 6-cores ARM CA53@1.6GHz, performance on PCIe may vary greatly due to different processors.
> 4. The image resolution has a great influence on the decoding time, the reasoning result has a great influence on the post-processing time, different test pictures may be different, and different thresholds have a great influence on the post-processing time.

## 8. YOLOv5 cpu opt

Based on the YOLOv5 mentioned above, this section optimizes the YOLOv5 postprocessing algorithm NMS. The following mainly explains the content and performance accuracy results of NMS optimization.

### 8.1. NMS Optimization Item
* Place the operation that filters the noise anchors before all other operations. Subsequent operations only need to process candidate boxes with significantly reduced numbers
* Remove a large number of sigmoid calculations during anchor filtering by setting a new threshold
* Optimize storage space to reduce traversal of data, and only retain coordinates, confidence, highest category score, and corresponding index of candidate boxes when decoding outputs
* Increase conf_thresh, filtering more noise boxes
* Remove some other redundant calculations

The time bottleneck of the optimized NMS algorithm lies in the size of the output map. Attempting to reduce the height or width or number of channels of the output map can further reduce the NMS computation time.

### 8.2. Precision Test
On SE5-16, use different examples and models, use dataset `datasets/coco/val2017_1000`, use threshold `conf_thresh=0.001，nms_thresh=0.6`, set `--use_cpu_opt=true` in cpp example or set `--use_cpu_opt` in python example, here is mAP test results:
| Test Platform |  Test Program    |            Test model               |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
| SE5-16       | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel |    0.373      |    0.579 |
| SE5-16       | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel |    0.370      |    0.572 |
| SE5-16       | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel |    0.375      |    0.573 |
| SE5-16       | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel |    0.375      |    0.573 |

> **Note**：  
> 1. The test notes in Section 6.2 apply here;
> 2. Postprocess acceleration does not involve hardware acceleration, and only the test data of the SE5-16 platform and fp32 model are provided here.

### 8.3. Performance Test
On SE5-16, use different examples and models, use dataset `datasets/coco/val2017_1000`, use threshold `conf_thresh=0.5，nms_thresh=0.5`, set `--use_cpu_opt=true` in cpp example or set `--use_cpu_opt` in python example, here is performance test results:
|Test Platform|  Test Program    |            Test model               |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
|   SE5-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      14.99      |      21.52      |      43.84      |      16.83      |
|   SE5-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.60       |      2.85       |      24.29      |      16.87      |
|   SE5-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.88       |      1.54       |      22.33      |      6.17       |
|   SE5-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.23       |      3.03       |      23.31      |      4.49       |

> **Note**：  
> 1. The test notes in Section 7.2 apply here;
> 2. Postprocess acceleration does not involve hardware acceleration, and only the test data of the SE5-16 platform and fp32 model are provided here.
> 3. Increasing `conf_thresh`, or using single class NMS (that is, set `#define USE_MULTICLASS_NMS 0` in yolov5s.cpp for cpp examples, or set the class variable `self.multi_label=False`) to accelerate postprocess to higher level.

## 9. FAQ
Please refer to [YOLOv5 Common Problems](./docs/YOLOv5_Common_Problems_EN.md) to see some problems of YOLOv5 inference.For other questions ,please refer to [FAQ](../../docs/FAQ_EN.md) to see some common questions and answers.