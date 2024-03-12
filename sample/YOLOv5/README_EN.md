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
  * [8.2 Performance and Precision Test](#82-performance-and-precision-test)
* [9. FAQ](#9-faq)
  
## 1. Introduction
YOLOv5 is a very classical One Stage target detection algorithm based on anchor. Because of its excellent accuracy and speed performance, it has been widely used in engineering practice. This example [​YOLOv5 official open source repository](https://github.com/ultralytics/yolov5) transplants the v6.1 version of the model and algorithm so that it can be inference tested on SOPHON BM1684/BM1684X/BM1688.

## 2. Characteristics
### 2.1 SDK Characteristics
* Support for BM1688(SoC)/BM1684X(x86 PCIe、SoC)/BM1684(x86 PCIe、SoC、arm PCIe)
* Support for FP32, FP16 (BM1688/BM1684X), INT8 model compilation and inference
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

This example provides a script for TPU-MLIR to compile FP32 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_fp32bmodel_mlir.sh`, and specify the target platform on which BModel runs (**BM1684/BM1684X/BM1688 is supported**) during execution, such as:

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684 #bm1684x/bm1688
```

Executing the above command will generate the `yolov5s_v6.1_3output_fp32_1b.bmodel` file under a folder like `models/BM1684`, that is, the converted FP32 BModel.

- Generate FP16 BModel

This example provides a script for TPU-MLIR to compile FP16 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_fp16bmodel_mlir.sh`, and specify the target platform on which BModel runs (**BM1684X/BM1688 is supported**) during execution, such as:

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688
```

Executing the above command will generate the `yolov5s_v6.1_3output_fp16_1b.bmodel` file under a folder like`models/BM1684X/`, that is, the converted FP16 BModel.

- Generate INT8 BModel

This example provides a script for quantifying INT8 BModel in the `scripts` directory. Please modify the parameters such as onnx model path, generated model directory and input size shapes in `gen_int8bmodel_mlir.sh`, and enter the target platform of BModel (**BM1684/BM1684X is supported**) during execution, such as:

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684 #bm1684x/bm1688
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
| SE5-16        | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.377         | 0.580    |
| SE5-16        | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 0.344         | 0.553    |
| SE5-16        | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.373         | 0.573    |
| SE5-16        | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.337         | 0.544    |
| SE5-16        | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.375         | 0.572    |
| SE5-16        | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.338         | 0.544    |
| SE5-16        | yolov5_sail.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.375         | 0.572    |
| SE5-16        | yolov5_sail.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.338         | 0.544    |
| SE7-32        | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.377         | 0.580    |
| SE7-32        | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.377         | 0.580    |
| SE7-32        | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 0.363         | 0.572    |
| SE7-32        | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.373         | 0.573    |
| SE7-32        | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.373         | 0.573    |
| SE7-32        | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.356         | 0.563    |
| SE7-32        | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.374         | 0.572    |
| SE7-32        | yolov5_bmcv.pcie | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.374         | 0.572    |
| SE7-32        | yolov5_bmcv.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.357         | 0.562    |
| SE7-32        | yolov5_sail.pcie | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.374         | 0.572    |
| SE7-32        | yolov5_sail.pcie | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.374         | 0.572    |
| SE7-32        | yolov5_sail.pcie | yolov5s_v6.1_3output_int8_1b.bmodel | 0.357         | 0.562    |
| SE9-16        | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.362         | 0.569    |
| SE9-16        | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.362         | 0.569    |
| SE9-16        | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.344         | 0.560    |
| SE9-16        | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.362         | 0.569    |
| SE9-16        | yolov5_sail.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.362         | 0.569    |
| SE9-16        | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 0.344         | 0.560    |
| SE9-16        | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.378         | 0.579    |
| SE9-16        | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.377         | 0.579    |
| SE9-16        | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 0.358         | 0.571    |
| SE9-16        | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 0.374         | 0.573    |
| SE9-16        | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 0.374         | 0.573    |
| SE9-16        | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 0.356         | 0.565    |
> **Test Description**:
> 1. The model mAP of batch_size=4 and batch_size=1 is the same.
> 2. Due to possible differences between SDK versions, it is normal for the mAP error of <0.01 between the actual running results and this table;
> 3. AP@IoU=0.5:0.95 is the corresponding indicator of area=all.
> 4. On a PCIe or SoC platform equipped with the same TPU and SOPHONSDK, the mAP of the same program is the same, SE5 series corresponds to BM1684, SE7 series corresponds to BM1684X, and SE9 series corresponds to BM1688;
> 5. The mAP of the BM1688 num_core=2 model is basically the same as that of the num_core=1 model.

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
| BM1684/yolov5s_v6.1_3output_fp32_1b.bmodel        | 22.6              |
| BM1684/yolov5s_v6.1_3output_int8_1b.bmodel        | 11.5              |
| BM1684/yolov5s_v6.1_3output_int8_4b.bmodel        | 6.4               |
| BM1684X/yolov5s_v6.1_3output_fp32_1b.bmodel       | 20.8              |
| BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel       | 7.2               |
| BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel       | 3.5               |
| BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel       | 3.3               |
| BM1688/yolov5s_v6.1_3output_fp32_1b.bmodel        | 99.1              |
| BM1688/yolov5s_v6.1_3output_fp16_1b.bmodel        | 28.7              |
| BM1688/yolov5s_v6.1_3output_int8_1b.bmodel        | 8.4               |
| BM1688/yolov5s_v6.1_3output_int8_4b.bmodel        | 7.3               |
| BM1688/yolov5s_v6.1_3output_fp32_1b_2core.bmodel  | 64.6              |
| BM1688/yolov5s_v6.1_3output_fp16_1b_2core.bmodel  | 19.3              |
| BM1688/yolov5s_v6.1_3output_int8_1b_2core.bmodel  | 7.6               |
| BM1688/yolov5s_v6.1_3output_int8_4b_2core.bmodel  | 5.3               |

> **Test Description**：  
> 1. The performance test results have a certain volatility.
> 2. The `calculate time` has been converted to the average inference time per picture.
> 3. The test results of SoC and PCIe are basically the same.

### 7.2 Program Performance
Refer to [C++ example](cpp/README_EN.md) or [Python example](python/README_EN.md) to run the program, and check the statistical decoding time, preprocessing time, inference time, post-processing time. The time info which C++/Python example prints have already been converted into processing time per image.

CPP set `--use_cpu_opt=false` or Python not set `--use_cpu_opt` for testing. Use different examples and models to test `datasets/coco/val2017_1000` with `conf_thresh=0.5,nms_thresh=0.5` on different test platforms. The performance test results are shown as follows:
|Test Platform|  Test Program    |            Test model               |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
| SE5-16      | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 14.0     | 27.8          | 33.5          | 115       |
| SE5-16      | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 13.9     | 23.5          | 33.5          | 111       |
| SE5-16      | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel | 13.8     | 24.2          | 28.2          | 115       |
| SE5-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 3.0      | 3.0           | 28.5          | 111       |
| SE5-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 3.0      | 2.4           | 17.4          | 111       |
| SE5-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 2.8      | 2.3           | 11.5          | 115       |
| SE5-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 5.4      | 1.5           | 22.6          | 15.7      |
| SE5-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 5.4      | 1.5           | 11.5          | 15.7      |
| SE5-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 5.2      | 1.6           | 6.2           | 15.5      |
| SE5-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 3.3      | 3.1           | 23.3          | 13.9      |
| SE5-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 3.3      | 1.9           | 12.2          | 13.9      |
| SE5-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 3.1      | 1.8           | 6.9           | 13.8      |
| SE7-32      | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel |  15.26   | 27.31         | 33.20         | 108.40    |
| SE7-32      | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel |  13.95   | 26.98         | 18.88         | 108.68    |
| SE7-32      | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel |  13.93   | 27.52         | 15.07         | 108.64    |
| SE7-32      | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel |  13.81   | 25.16         | 13.23         | 107.75    |
| SE7-32      | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel |  3.08    | 2.34          | 29.12         | 103.93    |
| SE7-32      | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel |  3.07    | 2.32          | 14.78         | 103.74    |
| SE7-32      | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel |  3.05    | 2.33          | 10.87         | 103.78    |
| SE7-32      | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel |  2.92    | 2.16          | 9.78          | 108.15    |
| SE7-32      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel |  4.27    | 0.73          | 21.62         |  15.81    |
| SE7-32      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel |  4.29    | 0.73          | 7.37          |  15.89    |
| SE7-32      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel |  4.30    | 0.73          | 3.47          |  15.90    |
| SE7-32      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b.bmodel |  4.14    | 0.71          | 3.33          |  15.73    |
| SE7-32      | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel |  2.69    | 2.58          | 23.40         |  14.15    |
| SE7-32      | yolov5_sail.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel |  2.69    | 2.59          | 9.11          |  14.17    |
| SE7-32      | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel |  2.70    | 2.58          | 5.22          |  14.16    |
| SE7-32      | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_4b.bmodel |  2.56    | 2.51          | 4.97          |  14.07    |
| SE9-16      | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 21.3     | 36.8          | 113.4         | 151.9     |
| SE9-16      | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel | 22.4     | 36.3          | 42.1          | 152.1     |
| SE9-16      | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 19.5     | 35.6          | 21.8          | 150.6     |
| SE9-16      | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel | 20.1     | 33.2          | 19.5          | 151.6     |
| SE9-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 4.8      | 5.2           | 106.5         | 143.97    |
| SE9-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 4.5      | 5.2           | 36.2          | 143.9     |
| SE9-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 4.5      | 5.2           | 15.6          | 144.2     |
| SE9-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 4.4      | 4.9           | 14.1          | 149.9     |
| SE9-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 5.8      | 2.0           | 98.1          | 22.4      |
| SE9-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 5.8      | 2.0           | 27.5          | 22.4      |
| SE9-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 5.8      | 2.0           | 7.3           | 22.4      |
| SE9-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 5.7      | 1.9           | 7.0           | 22.4      |
| SE9-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 4.3      | 5.3           | 99.6          | 19.9      |
| SE9-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 4.3      | 5.3           | 28.9          | 19.9      |
| SE9-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 4.3      | 5.3           | 8.7           | 19.9      |
| SE9-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 4.3      | 5.3           | 8.2           | 19.9      |

> **Test Description**：  
> 1. The time units are all milliseconds (ms), and the statistical time is the average processing time of each image.
> 2. The performance test results are volatile to a certain extent, so it is recommended that the average value should be taken from multiple tests.
> 3. SE5-16/SE7-32's processors are all 8-core ARM CA53@2.3GHz, SE9-16's processor is 8-core ARM CA53@1.6GHz, performance on PCIe may vary greatly due to different processors.
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

### 8.2. Performance and Precision Test
Use different examples and models to test `datasets/coco/val2017_1000` with `conf_thresh=0.001,nms_thresh=0.6` on different test platforms, c++ example set `--use_cpu_opt=true`, python example set `--use_cpu_opt` to use nms acceleration. The performance and accuracy test results before and after the improvement of the NMS post-processing algorithm are as follows:
| Test Platform|   Test Program   |             Test model              | YOLOv5 postprocess_time | YOLOv5_cpu_opt postprocess_time|AP@IoU=0.5:0.95| 
| ------------ | ---------------- | ----------------------------------- | ------------            | -------------                  | ------------- |
| SE5-16       | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 35.6                    | 22.9                           | 0.375         |
| SE5-16       | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 33.8                    | 20.5                           | 0.339         |
| SE5-16       | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 34.6                    | 21.1                           | 0.375         |
| SE5-16       | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 33.9                    | 18.9                           | 0.339         |
| SE5-16       | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 210.1                   | 98.4                           | 0.341         |
| SE5-16       | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 209.7                   | 99.7                           | 0.336         |

Use different examples and models to test `datasets/coco/val2017_1000` with `conf_thresh=0.01,nms_thresh=0.6` on different test platforms, c++ example set `--use_cpu_opt=true`, python example set `--use_cpu_opt` to use nms acceleration. The performance and accuracy test results before and after the improvement of the NMS post-processing algorithm are as follows:
| Test Platform|   Test Program   |             Test model              | YOLOv5 postprocess_time | YOLOv5_cpu_opt postprocess_time |AP@IoU=0.5:0.95|
| ------------ | ---------------- | ----------------------------------- | ------------            | -------------                   | --------------|
| SE5-16       | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 18.1                    | 7.5                             | 0.373         |
| SE5-16       | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 17.8                    | 7.2                             | 0.337         |
| SE5-16       | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 16.3                    | 5.8                             | 0.373         |
| SE5-16       | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 16.0                    | 5.5                             | 0.337         |
| SE5-16       | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 118.8                   | 22.9                            | 0.339         |
| SE5-16       | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 116.5                   | 22.8                            | 0.334         |

> **Note:** Due to the consistency between the implementation of sail and CPP, there were slight drops after Python calls, but there is a significant improvement in speed.

If using single-class NMS, by setting the macro `USE_MULTICLASS_NMS 0` in the `yolov5.cpp` file or setting YOLOv5 class member variable `self.multi_label=False` in both `yolov5_opencv.py` and `yolov5_bmcv.py` files, it can improve post-processing performance with slight loss of accuracy. Use different examples and models to test `datasets/coco/val2017_1000` with `conf_thresh=0.001,nms_thresh=0.6`, c++ example set `--use_cpu_opt=true`, python example set `--use_cpu_opt` to use nms acceleration. The performance and accuracy test results before and after the improvement of the NMS post-processing algorithm are as follows:
| Test Platform|   Test Program   |             Test model              | YOLOv5 postprocess_time | YOLOv5_cpu_opt postprocess_time |AP@IoU=0.5:0.95|
| ------------ | ---------------- | ----------------------------------- | ------------            | -------------                   | ------------- |
| SE5-16       | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 23.5                    | 10.2                            | 0.369         |
| SE5-16       | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 23.1                    | 9.9                             | 0.332         |
| SE5-16       | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 21.6                    | 8.5                             | 0.369         |
| SE5-16       | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 21.3                    | 8.1                             | 0.332         |
| SE5-16       | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 147.3                   | 32.5                            | 0.335         |
| SE5-16       | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 147.8                   | 32.8                            | 0.330         |

If using single-class NMS, by setting the macro `USE_MULTICLASS_NMS 0` in the `yolov5.cpp` file or setting YOLOv5's class member variable `self.multi_label=False` in both `yolov5_opencv.py` and `yolov5_bmcv.py` files, it can improve post-processing performance with slight loss of accuracy. Use different examples and models to test `datasets/coco/val2017_1000` with `conf_thresh=0.01,nms_thresh=0.6`, c++ example set `--use_cpu_opt=true`, python example set `--use_cpu_opt` to use nms acceleration. The performance and accuracy test results before and after the improvement of the NMS post-processing algorithm are as follows:
| Test Platform|   Test Program   |             Test model              | YOLOv5 postprocess_time | YOLOv5_cpu_opt postprocess_time |AP@IoU=0.5:0.95|
| ------------ | ---------------- | ----------------------------------- | ------------            | -------------                   | ------------- |
| SE5-16       | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 17.6                    | 6.2                             | 0.367         |
| SE5-16       | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 17.5                    | 6.1                             | 0.330         |
| SE5-16       | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 15.8                    | 4.5                             | 0.367         |
| SE5-16       | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 15.7                    | 4.3                             | 0.330         |
| SE5-16       | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 114.7                   | 9.6                             | 0.333         |
| SE5-16       | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 114.2                   | 9.5                             | 0.327         |

> **Test Description**：  
> 1. The time units are all milliseconds (ms), and the statistical time is the average processing time of each image.
> 2. The performance test results are volatile to a certain extent, so it is recommended that the average value should be taken from multiple tests.
> 3. The image resolution has a great influence on the decoding time, the reasoning result has a great influence on the post-processing time, different test pictures may be different, and different thresholds have a great influence on the post-processing time.

## 9. FAQ
Please refer to [YOLOv5 Common Problems](./docs/YOLOv5_Common_Problems_EN.md) to see some problems of YOLOv5 inference.For other questions ,please refer to [FAQ](../../docs/FAQ_EN.md) to see some common questions and answers.