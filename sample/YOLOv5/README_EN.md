[简体中文](./README.md) | [English](./README_EN.md)

# YOLOv5

## Catalogue

- [YOLOv5](#yolov5)
  - [Catalogue](#catalogue)
  - [1. Introduction](#1-introduction)
  - [2. Characteristics](#2-characteristics)
    - [2.1 Directory Instructions](#21-directory-instructions)
    - [2.2 SDK Characteristics](#22-sdk-characteristics)
    - [2.3 Algorithm Characteristics](#23-algorithm-characteristics)
  - [3. Data preparation and model compilation](#3-data-preparation-and-model-compilation)
    - [3.1 Data prepration](#31-data-prepration)
    - [3.2 Model Compilation](#32-model-compilation)
  - [4. Example Test](#4-example-test)
  - [5. Precision Test](#5-precision-test)
    - [5.1 Testing Method](#51-testing-method)
    - [5.2 Test Result](#52-test-result)
  - [6. Performance Testing](#6-performance-testing)
    - [6.1 bmrt\_test](#61-bmrt_test)
    - [6.2 Program Performance](#62-program-performance)
  - [7. YOLOv5 cpu opt](#7-yolov5-cpu-opt)
    - [7.1. NMS Optimization Item](#71-nms-optimization-item)
    - [7.2. Precision Test](#72-precision-test)
    - [7.3. Performance Test](#73-performance-test)
  - [8. FAQ](#8-faq)
  
## 1. Introduction
YOLOv5 is a very classical One Stage target detection algorithm based on anchor. Because of its excellent accuracy and speed performance, it has been widely used in engineering practice. This example [​YOLOv5 official open source repository](https://github.com/ultralytics/yolov5) transplants the v6.1 version of the model and algorithm so that it can be inference tested on SOPHON BM1684/BM1684X/BM1688/CV186X.

## 2. Characteristics

### 2.1 Directory Instructions
```bash
├── cpp                   # Store C++ example and its README.
|   ├──README_EN.md  
|   ├──README.md
|   ├──yolov5_bmcv        # C++ example which decoding with FFmpeg, preprocessing with BMCV, inference with BMRT.
|   └──yolov5_sail        # C++ example which decoding with SAIL, preprocessing with SAIL, inference with SAIL.
├── docs                  # Store documents for this sample, such as ONNX export and common problems.
├── pics                  # Store pictures for this sample's documents.
├── python                # Store Python example and its README.
|   ├──README_EN.md
|   ├──README.md
|   ├──yolov5_bmcv.py     # Python example which decoding with SAIL, preprocessing with SAIL.BMCV, inference with SAIL.
|   ├──yolov5_opencv.py   # Python example which decoding with SAIL, preprocessing with BMCV, Inference with SAIL.
|   └──...                # Common functions for python examples.
├── README_EN.md          # English guide for this sample.
├── README.md             # Chinese guide for this sample.
├── scripts               # Store shell scripts such as bmodel compilation, data downloads, auto test.
└── tools                 # Store python scripts such as evalutation, statis comparison.
```

### 2.2 SDK Characteristics
* Support for BM1688/CV186X(SoC), BM1684X(x86 PCIe、SoC、riscv PCIe), BM1684(x86 PCIe、SoC、arm PCIe)
* Support for FP32, FP16 (BM1684X/BM1688/CV186X), INT8 model compilation and inference
* Support C++ inference based on BMCV preprocessing
* Support Python inference based on OpenCV and BMCV preprocessing
* Support single batch and multi-batch model inference
* Support 1 output and 3 output model inference
* Support for picture and video testing
* Support NMS postprocessing acceleration

### 2.3 Algorithm Characteristics
PYTHON CODE
1. `sophon-demo/sample/YOLOv5/python`: Source code is available in this repository, implementing simple demos based on both bmcv and opencv interfaces. These are used for testing model accuracy and are **not recommended for performance evaluation**.
2. `sophon-demo/sample/YOLOv5_opt/python`: Source code is available in this repository, supporting only in `1684x`. It implements YOLOv5 decoding layers and NMS operations using TPU to enhance end-to-end performance.
3. `sophon-sail/sample/python/yolov5_multi_3output_pic.py`: Source code resides in the SDK's sophon-sail folder. It utilizes Python to call C++ encapsulated interfaces, allocating decoding, pre-processing, inference, and post-processing in separate threads to improve overall performance.
4. Sections 8 and 9 of the "TPU-MLIR Quick Start Guide" involve leveraging TPU for pre-processing and post-processing. Please refer to the corresponding documentation and develop corresponding routines to integrate pre-processing and post-processing into the algorithm, thereby improving end-to-end performance.
5. Utilize the Python `multiprocessing` module to invoke time-consuming function via multiple processes, enhancing overall throughput.
6. `sophon-demo/sample/YOLOv5_fuse/python`，fuse pre/post process into bmodel, **significantly increase end2end performance**, to use this example, you shall install the latest libsophon and tpu-mlir.

C++ CODE
1. `sophon-demo/sample/YOLOv5/cpp`: Source code is available in this repository, demonstrating simple demos based on bmcv and sail interfaces for accuracy verification.
2. `sophon-demo/sample/YOLOv5_opt/cpp`: Source code in this repository supports only in `1684x`. It utilizes TPU for YOLOv5 decoding layers and NMS operations, improving end-to-end performance.
3. `sophon-stream/samples/yolov5`: Source code is in the SDK (version V23.10.01 and above) under `sophon-stream`, separating pre-processing, inference, and post-processing into different threads to significantly improve overall performance.
4. `sophon-pipeline/examples/yolov5`: [Source code](https://github.com/sophgo/sophon-pipeline) , implementing the entire algorithmic inference process based on a thread pool to enhance overall performance.
5. Sections 8 and 9 of the "TPU-MLIR Quick Start Guide" involve leveraging TPU for pre-processing and post-processing, thereby improving end-to-end performance.
6. `sophon-demo/sample/YOLOv5_fuse/cpp`，fuse pre/post process into bmodel, **significantly increase end2end performance**, to use this example, you shall install the latest libsophon and tpu-mlir.

> **note**  
> This code supports both triple-output and single-output models. The single-output model demonstrates higher performance, but you should set sensitive layers for quantization. On the other hand, the triple-output model simplifies quantization. **When used for model accuracy validation, it is recommended to opt for the triple-output model.**
 
## 3. Data preparation and model compilation

### 3.1 Data prepration

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**
This sample provides `scripts/download.sh` that can download datasets and models for its test. **If you wish preparing models and datasets by yourself, you can skip this section and refer to [3.2 model compilation](#32-model-compilation) to do model compiling.**
```bash
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

### 3.2 Model Compilation

**If you do not have to compile models, and you are using the datasets and models from [3.1 data prepration](#31-data-prepration), you can skip this section.**

Source model needs to be compiled into BModel to run on SOPHON TPU. Before compilation, it should be exported to onnx model, if your tpu-mlir's version >= v1.3.0(from v23.07.01 SDK on our official site), you can also use torchscript model. Please check out [YOLOv5_Export_Guide](./docs/YOLOv5_Export_Guide_EN.md). In the meantime, prepare datasets for inference and quantization if you need to.

Use TPU-MLIR to compile BModel, refer to [TPU-MLIR Installation](../../docs/Environment_Install_Guide_EN.md#1-tpu-mlir-environmental-installation) to install TPU-MLIR before compilation. After installation, you need to enter this sample's root directory in the TPU-MLIR environment, and use the scripts this sample provides to compile onnx model into BModel. For specific details of commands in these scripts, you can look out in `TPU-MLIR_Technical_Reference_Manual.pdf`(please obtain it from the corresponding version of SDK of [Sophgo official website](https://developer.sophgo.com/site/index.html?categoryActive=material)).

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

## 4. Example Test
- [C++ Example](./cpp/README_EN.md)
- [Python Example](./python/README_EN.md)

## 5. Precision Test
### 5.1 Testing Method

First of all, refer to [C++ example](cpp/README_EN.md#32-image-test-demo) or [Python example](python/README_EN.md#22-image-test-demo) to deduce the dataset to be tested, generate the predicted json file, and pay attention to modifying the dataset (datasets/coco/val2017_1000) and related parameters (conf_thresh=0.001, nms_thresh=0.6).
Then, use the `test generated .py` script under the `tools` directory to compare the json file generated by the test with the test set tag json file, and calculate the evaluation metrics for target detection. The command is as follows:
```bash
# Install pycocotools, skip if it is already installed
pip3 install pycocotools
# Please modify the program path and json file path according to the actual situation
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yolov5s_v6.1_3output_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 5.2 Test Result
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
| SE7-32       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_1b.bmodel      |    0.361 |    0.570 |
| SE7-32       | yolov5_opencv.py   | yolov5s_v6.1_3output_int8_4b.bmodel      |    0.361 |    0.570 |
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

## 6. Performance Testing
### 6.1 bmrt_test
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

### 6.2 Program Performance
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
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      14.10      |      36.40      |     112.48      |     151.18      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      9.82       |      35.82      |      41.96      |     150.27      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      9.60       |      36.77      |      21.98      |     150.78      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      9.44       |      33.33      |      19.38      |     152.33      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.53       |      4.85       |     107.05      |     143.24      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.53       |      4.86       |      36.74      |     143.46      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      4.53       |      4.85       |      16.87      |     143.56      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      4.40       |      4.54       |      14.93      |     149.27      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      5.95       |      1.79       |      97.61      |      22.25      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      5.97       |      1.79       |      27.37      |      22.22      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      5.96       |      1.79       |      7.14       |      22.25      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      5.81       |      1.71       |      7.03       |      21.98      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.97       |      5.02       |     100.10      |      19.79      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      3.97       |      5.00       |      29.84      |      19.77      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.94       |      5.00       |      9.60       |      19.76      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.79       |      4.76       |      9.29       |      19.63      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      9.51       |      36.35      |      67.50      |     150.78      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      9.41       |      35.45      |      32.00      |     150.56      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      9.53       |      36.07      |      20.61      |     150.55      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      9.43       |      32.73      |      17.31      |     152.09      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      4.54       |      4.86       |      62.00      |     143.29      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      4.54       |      4.86       |      27.86      |     143.22      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      4.51       |      4.87       |      15.66      |     143.12      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      4.39       |      4.52       |      12.94      |     149.71      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      5.96       |      1.79       |      52.70      |      22.23      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      5.99       |      1.79       |      18.09      |      22.24      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      5.96       |      1.79       |      6.33       |      22.24      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      5.79       |      1.71       |      5.00       |      21.99      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      3.98       |      5.01       |      55.17      |      19.79      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      3.98       |      5.01       |      20.55      |      19.80      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      3.96       |      5.01       |      8.80       |      19.83      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      3.78       |      4.75       |      7.26       |      19.64      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      20.99      |      36.78      |     112.86      |     151.88      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      20.48      |      36.45      |      42.31      |     151.69      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      19.23      |      34.47      |      20.47      |     149.72      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      19.31      |      33.41      |      18.75      |     154.23      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.11       |      4.73       |     107.17      |     144.27      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.05       |      4.71       |      36.70      |     144.19      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      4.12       |      4.76       |      15.80      |     144.34      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      3.94       |      4.42       |      14.18      |     150.38      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      5.56       |      1.80       |      97.75      |      22.42      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      5.56       |      1.79       |      27.33      |      22.41      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      5.54       |      1.79       |      6.35       |      22.45      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      5.39       |      1.72       |      6.22       |      22.14      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.58       |      4.83       |     100.27      |      20.00      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      3.60       |      4.82       |      29.83      |      19.99      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.62       |      4.82       |      8.84       |      20.04      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.45       |      4.63       |      8.53       |      19.78      |

> **Note**：  
> 1. The time units are all milliseconds (ms), and the statistical time is the average processing time of each image.
> 2. The performance test results are volatile to a certain extent, so it is recommended that the average value should be taken from multiple tests.
> 3. SE5-16/SE7-32's processors are all 8-cores ARM CA53@2.3GHz, SE9-16's processor is 8-cores ARM CA53@1.6GHz and SE9-8 use 6-cores ARM CA53@1.6GHz, performance on PCIe may vary greatly due to different processors.
> 4. The image resolution has a great influence on the decoding time, the reasoning result has a great influence on the post-processing time, different test pictures may be different, and different thresholds have a great influence on the post-processing time.

## 7. YOLOv5 cpu opt

Based on the YOLOv5 mentioned above, this section optimizes the YOLOv5 postprocessing algorithm NMS. The following mainly explains the content and performance accuracy results of NMS optimization.

### 7.1. NMS Optimization Item
* Place the operation that filters the noise anchors before all other operations. Subsequent operations only need to process candidate boxes with significantly reduced numbers
* Remove a large number of sigmoid calculations during anchor filtering by setting a new threshold
* Optimize storage space to reduce traversal of data, and only retain coordinates, confidence, highest category score, and corresponding index of candidate boxes when decoding outputs
* Increase conf_thresh, filtering more noise boxes
* Remove some other redundant calculations

The time bottleneck of the optimized NMS algorithm lies in the size of the output map. Attempting to reduce the height or width or number of channels of the output map can further reduce the NMS computation time.

### 7.2. Precision Test
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

### 7.3. Performance Test
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

## 8. FAQ
Please refer to [YOLOv5 Common Problems](./docs/YOLOv5_Common_Problems_EN.md) to see some problems of YOLOv5 inference.For other questions ,please refer to [FAQ](../../docs/FAQ_EN.md) to see some common questions and answers.