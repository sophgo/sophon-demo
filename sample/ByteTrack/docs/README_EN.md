[简体中文](../README.md) | English
# ByteTrack

**Deploy YOLOX+ByteTrack target tracking using OpenCV and BMCV, including both C++ and Python versions of the program.**

- [ByteTrack](#bytetrack)
	- [1. Introduction](#1-introduction)
	- [2. Features](#2-features)
	- [3. Preparation](#3-preparation)
	- [4. Model Compilation](#4-model-compilation)
		- [4.1 Compilation of BModel using TPU-NNTC](#41-compilation-of-bmodel-using-tpu-nntc)
		- [4.2 Compilation of BModel using TPU-MLIR](#42-compilation-of-bmodel-using-tpu-mlir)
	- [5. Example Test](#5-example-test)
	- [6. Accuracy Testing](#6-accuracy-testing)
		- [6.1 Test Method](#61-test-method)
		- [6.2Automated Testing](#62automated-testing)
		- [6.3 Test Result](#63-test-result)
	- [7. Performance Testing](#7-performance-testing)
		- [7.1 bmrt\_test](#71-bmrt_test)
		- [7.2 Program Performance](#72-program-performance)
	- [8. FAQ](#8-faq)

## 1. Introduction
ByteTrack is a simple, fast and strong multi-object tracker.
Multi-object tracking (MOT) aims at estimating bounding boxes and identities of objects in videos. Most methods obtain identities by associating detection boxes whose scores are higher than a threshold. The objects with low detection scores, e.g. occluded objects, are simply thrown away, which brings non-negligible true object missing and fragmented trajectories. To solve this problem, we present a simple, effective and generic association method, tracking by associating every detection box instead of only the high score ones. For the low score detection boxes, we utilize their similarities with tracklets to recover true objects and filter out the background detections.

**Paper** (https://arxiv.org/abs/2110.06864)

**Source Code** (https://github.com/ifzhang/ByteTrack)

## 2. Features
* Supports BM1684X (x86 PCIe, SoC) and BM1684 (x86 PCIe, SoC, arm PCIe)
* Supports FP32 model compilation and inference
* Supports C++ inference based on BMCV pre-processing
* Supports Python inference based on OpenCV and BMCV pre-processing
* Supports image and video testing

## 3. Preparation
Pytorch's model need to goes through 'torch.jit.trace' before compiling, and the traced model can be used to compile BModel. The method and principle of trace can be found in [torch.jit.trace Guide](../../../docs/torch.jit.trace_Guide.md)。

At the same time, you need to prepare a dataset for testing and, if quantizing the model, a dataset for quantization.

This routine provides the download script 'download.sh' of the relevant model and dataset in the 'scripts' directory, and automatically downloads the pt model, dataset and BModel after running, that is, you can skip Chapter 4 model compilation. You can also use the downloaded pt model and quantization dataset, or prepare the model and dataset yourself, and refer to [4. Model compilation](#4-model-compilation) for model conversion to generate BModel.

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install unzip
cd ./scripts
chmod +x download.sh
./download.sh
```

After execution, the model is saved to 'models/', the test video and  datasets are saved to 'datasets/'.

Downloaded models include:

```
./models
├── BM1684
│   ├── bytetrack_s_fp32_1b.bmodel   # BM1684 FP32 BModel，batch_size=1
├── BM1684X
│   ├── bytetrack_s_st_fp32_1b.bmodel   # BM1684X FP32 BModel，batch_size=1
└── onnx
    └── bytetrack_s.onnx             # onnx model
```

Downloaded data include:

```
./datasets
├── sample.mp4                                # Test video
└── MOT15                                     # MOT15 dataset
    └──  ADL-Rundle-6                         # Extract ADL-Rundle-6 from the train directory in MOT15
          ├── det                             # Detection Comparison
          ├── gt                              # Ground truth
          └── img1                            # Test pictures
```


## 4. Model Compilation

The exported model needs to be compiled into a BModel to run on the Sophon TPU. If using a pre-compiled BModel, you can skip this section. If you are using the BM1684 chip, it is recommended to use TPU-NNTC to compile the BModel. If you are using the BM1684X chip, it is recommended to use TPU-MLIR to compile the BModel.

### 4.1 Compilation of BModel using TPU-NNTC

Before compiling the model, TPU-NNTC needs to be installed. For specific instructions, please refer to [TPU-NNTC环境搭建](../../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建). After installation, you need to enter the sample directory in the TPU-NNTC environment.

- Generate FP32 BModel

To compile a trace of the torchscript model into FP32 BModel using TPU-NNTC, or directly compile the onnx model, refer to the 《TPU-NNTC开发参考手册》 (available from the corresponding version of the SDK on the [算能官网](https://developer.sophgo.com/site/index/material/28/all.html)).

This tutorial provides a script for compiling FP32 BModel using TPU-NNTC in the `scripts` directory. Please modify the model path, the generated model directory, input size shapes, and other parameters in the `gen_fp32bmodel_nntc.sh` script, and specify the target platform (BM1684 or BM1684X) for BModel execution during execution. For example:

```bash
cd scripts/
chmod +x gen_fp32bmodel_nntc.sh
./gen_fp32bmodel_nntc.sh BM1684
```

The above command will generate the `bytetrack_s_fp32_1b.bmodel` file in the models/BM1684/ directory, which is the converted FP32 BModel.

### 4.2 Compilation of BModel using TPU-MLIR

Before compiling the model, TPU-MLIR needs to be installed, which can be referred to [TPU-MLIR环境搭建](../../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建). After installation, enter the demo directory in the TPU-MLIR environment. To compile an ONNX model to a BModel using TPU-MLIR, please refer to section "3. Compiling ONNX Models" in the "TPU-MLIR Quick Start Guide" (obtain the corresponding version of the SDK from the [算能官网](https://developer.sophgo.com/site/index/material/31/all.html)).

- Generating FP32 BModel

In the `scripts` directory, this demo provides a script for compiling an FP32 BModel using TPU-MLIR. Please modify the ONNX model path, the directory for saving the generated model, input size, and other parameters in `gen_fp32bmodel_mlir.sh`. During execution, please specify the target platform for the BModel (BM1684X is supported), such as:

```bash
cd scripts/
chmod +x gen_fp32bmodel_mlir.sh
./gen_fp32bmodel_mlir.sh bm1684x
```

The above command will generate the `bytetrack_s_fp32_1b.bmodel` file in the models/BM1684X/ directory, which is the converted FP32 BModel.

## 5. Example Test

- [C++ Example](./README_CPP_EN.md)
- [Python Example](./README_PY_EN.md)

## 6. Accuracy Testing
### 6.1 Test Method
First, refer to the [C++ example](README_CPP_EN.md#41-test-mot-dataset) or [Python example](README_PY_EN.md#31-test-mot-dataset) to infer the dataset to be tested and generate a txt file containing the target tracking results. Pay attention to modify the dataset (datasets/MOT15/ADL-Rundle-6/img1). Then, use the eval_mot.py script in the tools directory to compare the txt file generated by the test with the txt file of test set labels and calculate a series of evaluation indicators for target tracking. The command is as follows:
```bash
# Install motmetrics
pip3 install motmetrics
# 请根据实际情况修改程序路径和txt文件路径
    python3 ../tools/eval_mot.py \
      --ground_truths=../datasets/MOT15/ADL-Rundle-6/gt/gt.txt \
      --detections=../cpp/bytetrack_bmcv/results/img1_bytetrack_s_fp32_1b_cpp.txt
```

**Results：**
```bash
MOTA = -0.4791375524056698
     num_frames      IDF1       IDP       IDR      Rcll     Prcn    GT  MT  PT  ML    FP    FN  IDsw   FM      MOTA      MOTP
acc         525  0.049857  0.058351  0.043522  0.159114  0.21333  5009   0   7  17  2939  4212   258  562 -0.479138  0.342534
```

### 6.2Automated Testing

This automated test script needs to be performed on an x86 host or Sophi SoC device with a PCIe accelerator card.

Rely on the python package 'motmetrics'
```bash
pip3 install motmetrics
```

After preparing the BModel with test data:

```bash
cd scripts
chmod +x ./auto_test.sh
./auto_test.sh ${platform} ${target} ${tpu_id} ${sail_dir}
```

where 'platform' refers to the platform (x86 or soc), 'target' is the chip model (BM1684 or BM1684X), 'tpu_id' specifies the ID of the TPU (viewed using BM-SMI), 'sail_dir' is the installation path of SAIL. If the final output is 'Failed:', execution failed, otherwise it indicates success.

For example,

```bash
./auto_test.sh soc BM1684 0 /opt/sophon/sophon-sail
```

On x86, 'auto_test.sh' includes the compilation and operation of C++ programs in the cpp folder and the running of all Python programs in the Python folder, as well as the operation of MOT metrics scripts.

On soc, auto_test.sh includes the operation of C++ programs in the cpp folder and the operation of all Python programs in the Python folder, as well as the operation of MOT metrics scripts.


To execute this script on x86, refer to [x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)，then run this script, where ${sail_dir} builds the resulting sophon-sail installation path for the above environment, usually /opt/sophon/sophon-sail.

To execute this script on SoC, you first need to cross-compile the ARM program, refer to [交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#31-交叉编译环境搭建)，then move the generated executable to the cpp folder. After that, set the environment variables.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
```

Run this script again, where ${sail_dir} is the build_soc/sophon-sail folder created for the above environment.

### 6.3 Test Result
Here we use the target detection model `bytetrack_s_fp32_1b.bmodel` and the `ADL-Rundle-6` dataset. Record MOTA as an accuracy indicator. The accuracy test results are as follows:

|Test Platform|    Test Program     |          Test model        | MOTA |
| ------------|   ----------------  | -------------------------- | ---- |
| BM1684 SoC  | bytetrack_bmcv.soc  | bytetrack_s_fp32_1b.bmodel | 47.9 |
| BM1684 SoC  | bytetrack_opencv.py | bytetrack_s_fp32_1b.bmodel | 44.1 |
| BM1684 SoC  | bytetrack_bmcv.py   | bytetrack_s_fp32_1b.bmodel | 37.1 |

## 7. Performance Testing
### 7.1 bmrt_test
Use bmrt_test to test the theoretical performance of the model:”
```bash
# Please modify the path of the bmodel to be tested according to the actual situation.
bmrt_test --bmodel models/BM1684/bytetrack_s_fp32_1b.bmodel
```
The calculate time in the test results is the inference time of the model.
The theoretical inference time of each model was tested and the results are as follows:

|           Test model               |  calculate time(ms) |
| -----------------------------      |  -----------------  |
| BM1684/bytetrack_s_fp32_1b.bmodel  |      40.50         |

> **Test Description**:

1. Performance test results have a certain degree of fluctuation;
2. `calculate time` has been converted to the average inference time per image

### 7.2 Program Performance
Refer to the C++ example or Python example to run the program and view the statistics of preprocessing time, inference time, and post-processing time.

On different test platforms, using different examples and models to test `datasets/MOT5/ADL-Rundle-6/img1`, the performance test results are as follows:

|Test Platform|     Test Program    |         Test model         |preprocess_time|inference_time|postprocess_time|track_time| overall_time|
| ----------- | ------------------- |  ------------------------- | ------------- | ------------- | ------------ |  --------- | ---------- |
| BM1684 soc  | bytetrack_opencv.py | bytetrack_s_fp32_1b.bmodel |     214.70    |     54.59     |     7.87     |    10.26   |   359.31   |
| BM1684 soc  | bytetrack_bmcv.py   | bytetrack_s_fp32_1b.bmodel |     30.19     |     41.50     |     7.66     |     9.39   |    99.91   |
| BM1684 soc  | bytetrack_bmcv.soc  | bytetrack_s_fp32_1b.bmodel |     10.84     |     40.56     |     0.19     |     0.78   |    52.37   |


> **Test Description**：
1. The time unit is milliseconds (ms), preprocess_time, inference_time, and postprocess_time are the processing times of the YOLOX detector. track_time is the time for the bytetrack algorithm to update the tracker. overall_time is the time to process one frame of an image;
2. Performance test results have a certain degree of fluctuation and it is recommended to take the average value after multiple tests;
3. The main control CPU of BM1684/1684X SoC are all 8-core ARM A53 42320 DMIPS @2.3GHz, and there may be a large difference in performance on PCIe due to different CPUs;
4. Image resolution has a significant impact on decoding time;

## 8. FAQ
Please refer to the [FAQ](../../../docs/FAQ.md) for some common questions and answers.