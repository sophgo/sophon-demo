# BERT

##Table of Contents

* [1. Introduction](#1-Introduction)
* [2. Features](#2-Features)
* [3. Prepare Model and Data](#3-Prepare-Models-and-Data)
* [4. Model Compilation](#4-Model-Compilation)
  * [4.1 TPU-NNTC Compile BModel](#41-tpu-nntc-Compilation-BModel)
  * [4.2 TPU-MLIR Compile BModel](#42-tpu-mlir-Compilation-BModel)
* [5. Routine Testing](#5-Routine-Testing)
* [6. Accuracy-Testing](#6-Accuracy-testing)
  * [6.1 Test-Methods](#61-Test-methods)
  * [6.2 Test Results](#62-Test-Results)
* [7. Performance Test](#7-Performance-Test)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 Program Running Performance](#72-Program-Running-Performance)
* [8. FAQ](#8-faq)

## 1. introduction
The full name of BERT is Bidirectional Encoder Representation from Transformers, which is a pre trained language representation model. It emphasizes that traditional unidirectional language models or shallow concatenation of two unidirectional language models are no longer used for pre training, but a new masked language model (MLM) is adopted to generate deep bidirectional language representations. When the BERT paper was published, it was mentioned that a new state of the art result was obtained in 11 NLP (Natural Language Processing) tasks, which was astounding.
A simple training framework that recreates bert4keras in PyTorch. bert4torch
## 2. Features
* Supports BM1684X (x86 PCIe, SoC) and BM1684 (x86 PCIe, SoC)
* Support FP32, FP16 (BM1684X) model compilation and inference
* Support for Sail based C++inference
* Support Python inference based on sail
* Support single batch and multi batch model inference
* Support text testing

## 3. Prepare models and data
If you are using the BM1684 chip, it is recommended to use TPU-NNTC to compile the BModel. The Python model should be exported as a torchscript model or onnx model before compilation; If you are using the BM1684X chip, it is recommended to compile the BModel using TPU-MLIR, and the Python model should be exported as an onnx model before compilation. For details, please refer to [BERT-Model-Export](./docs/BERT4torch_Exportonnx_Guide.md).

At the same time, you need to prepare a dataset for testing, and if quantifying the model, you also need to prepare a dataset for quantification.

This routine provides the download script 'download.sh' for the relevant model and data in the 'scripts' directory. You can also prepare the model and dataset yourself and refer to [4-Model-Compilation](#4-Model-Compilation) for model conversion.

```bash
#Install unzip. If it is already installed, please skip it. For non ubuntu systems, use yum or other methods depending on the situation
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

The downloaded models include:
```
./models/
├── BM1684
│   └── bert4torch_output_fp32_1b.bmodel
│   └── bert4torch_output_fp32_8b.bmodel
├── BM1684X
│   └── bert4torch_output_fp32_1b.bmodel
│   └── bert4torch_output_fp32_8b.bmodel
│   └── bert4torch_output_fp16_1b.bmodel
│   └── bert4torch_output_fp16_8b.bmodel
├── pre_train
│   └── vocab.txt
└── torch
    └── bert4torch_jit.pt
```

The downloaded data includes:

```
./datasets/china-people-daily-ner-corpus
├── example.dev                                              # 验证集
├── example.test                                             # 测试集
└── example.train                                            # 训练集
```

## 4. Model Compilation
The exported model needs to be compiled into BModel to run on SOPHON TPU. If you use the downloaded BModel, you can skip this section. If you use the BM1684 chip, it is recommended to use TPU-NNTC to compile BModel; If you are using the BM1684X chip, it is recommended to compile the BModel using TPU-MLIR.

### 4.1 TPU-NNTC Compilation BModel
Before model compilation, it is necessary to install TPU-NNTC. For details, please refer to the [TPU-NNTC-environment-setup](../../docs/Environment_install_Guide_EN.md#1-TPU-NNTC-Environmental-Installation). After installation, you need to enter the routine directory in the TPU-NNTC environment.

-Generate FP32 BModel

Use TPU-NNTC to compile the torch script model after trace into FP32 BModel. For specific methods, please refer to the "BMNETP Usage" section of the "TPU-NNTC Development Reference Manual" (please refer to the "Calculation Official Website")(https://developer.sophgo.com/site/index/material/28/all.html)Obtained from the corresponding version of the SDK.

This routine provides the TPU-NNTC compilation script for FP32 BModel in the 'scripts' directory. Please pay attention to modifying  the parameters such as the path of the torchscript model, the directory of the generative model, and the input size shapes in 'gen_fp32bmodel_nntc.sh', and the target platform for the BModel to run (supporting BM1684 and BM1684X) are specified during execution, such as:

```bash
./scripts/gen_fp32bmodel_nntc.sh BM1684
```

Executing the above command will generate 'models/BM1684/bert4torch_output_fp32_1b.bmodel'file, which is the converted FP32 BModel.


### 4.2 TPU-MLIR Compilation BModel
Before model compilation, it is necessary to install TPU-MLIR. For details, please refer to [TPU MLIR Environment Construction](../../docs/Environment_install_Guide_EN.md#2-TPU-MLIR-Environmental-Installation). After installation, you need to enter the routine directory in the TPU-MLIR environment. Use TPU-MLIR to compile the onnx model into BModel. For specific methods, please refer to "3 Compile the ONNX model "(please refer to the [Calculus official website](https://developer.sophgo.com/site/index/material/31/all.html )Obtained from the corresponding version of the SDK).

-Generate FP32 BModel

This routine provides the TPU-MLIR compilation script for FP32 BModel in the 'scripts' directory. Please pay attention to modifying the Onnx model path, generative model directory, input size shapes and other parameters in 'gen_fp32bmodel_mlir.sh', and specify the target platform for BModel to run (BM1684X is supported) during execution, such as:

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

Executing the above command will generate 'models/BM1684X/bert4torch_output_fp32_1b.bmodel'file, which is the converted FP32 BModel.

-Generate FP16 BModel

This routine provides the TPU-MLIR compilation script for FP16 BModel in the 'scripts' directory. Please pay attention to modifying the Onnx model path, generative model directory, input size shapes and other parameters in 'gen_fp16bmodel_mlir.sh', and specify the target platform for BModel to run (BM1684X is supported) during execution, such as:

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

Executing the above command will generate  under 'models/BM1684X/bert4torch_output_fp16_1b.bmodel'file, which is the converted FP16 BModel.


## 5. Routine testing
-[C++ Routine](./cpp/README_EN.md)
-[Python Routine](./Python/README_EN.md)
## 6. Accuracy testing
### 6.1 Test methods

Firstly, refer to the [C++ routine](cpp/README_EN.md#32-test-text) or the [Python routine](Python/README_EN.md#22-test-text) to infer the dataset to be tested, generate the predicted txt file, and pay attention to modifying the dataset (datasets/Chinese people daily ner corpus).
Then, use The 'eval_people.py' script in the 'tools' directory compares the txt file generated by the test with the txt file labeled by the test set to calculate the evaluation indicators for target detection. The command is as follows:
```bash
#Install seqeval, if already installed, please skip
pip3 install seqeval
#Please modify the program path and JSON file path according to the actual situation
python3 tools/eval_people.py --test_path ../datasets/china-people-daily-ner-corpus/example.test --input_path ../python/results/bert4torch_output_fp16_8b.bmodel_sail_python_result.txt
```
### 6.2 test results
On the China people daily ner corpus dataset, the accuracy test results are as follows:
|Test platform | test program     | test model                          | f1            |accuary   |
| ------------ | ---------------- | ----------------------------------- | ------------- | -------- |
| BM1684 PCIe  | bert_sail.pcie   | bert4torch_output_fp32_1b.bmodel    | 0.9203        | 0.9914   |
| BM1684 PCIe  | bert_sail.pcie   | bert4torch_output_fp32_8b.bmodel    | 0.9185        | 0.9914   |
| BM1684X PCIe | bert_sail.pcie   | bert4torch_output_fp32_1b.bmodel    | 0.9130        | 0.9907   |
| BM1684X PCIe | bert_sail.pcie   | bert4torch_output_fp32_8b.bmodel    | 0.9130        | 0.9907   |
| BM1684X PCIe | bert_sail.pcie   | bert4torch_output_fp16_1b.bmodel    | 0.9121        | 0.9907   |
| BM1684X PCIe | bert_sail.pcie   | bert4torch_output_fp16_8b.bmodel    | 0.9120        | 0.9907   |
| BM1684 PCIe  | bert_sail.py     | bert4torch_output_fp32_1b.bmodel    | 0.9173        | 0.9915   |
| BM1684 PCIe  | bert_sail.py     | bert4torch_output_fp32_8b.bmodel    | 0.9163        | 0.9915   |
| BM1684X PCIe | bert_sail.py     | bert4torch_output_fp32_1b.bmodel    | 0.9161        | 0.9914   |
| BM1684X PCIe | bert_sail.py     | bert4torch_output_fp32_8b.bmodel    | 0.9224        | 0.9917   |
| BM1684X PCIe | bert_sail.py     | bert4torch_output_fp16_1b.bmodel    | 0.9219        | 0.9915   |
| BM1684X PCIe | bert_sail.py     | bert4torch_output_fp16_8b.bmodel    | 0.9191        | 0.9915   |

>**Test Instructions**:

1. The test results have a certain degree of volatility; Basically within 0.1

## 7. performance test
### 7.1 bmrt_test
Using bmrt_ heoretical performance of the test model:
```bash
#Please modify the bmodel path and devid parameters to be tested according to the actual situation
bmrt_test --bmodel models/BM1684/bert4torhc_output_fp32_1b.bmodel
```
The 'calculate time' in the test results is the model inference time, and a multi batch size model should be divided by the corresponding batch size to determine the theoretical inference time for each image.
Test the theoretical reasoning time of each model, and the results are as follows:
|                  test model                 | calculate time(ms) |
| ------------------------------------------- | ----------------- |
| BM1684/bert4torch_output_fp32_1b.bmodel     | 170.848           |
| BM1684/bert4torch_output_fp32_8b.bmodel     | 1173.222          |
| BM1684X/bert4torch_output_fp32_1b.bmodel    | 91.473            |
| BM1684X/bert4torch_output_fp16_1b.bmodel    | 8.643             |
| BM1684X/bert4torch_output_fp32_8b.bmodel    | 699.825           |
| BM1684X/bert4torch_output_fp16_8b.bmodel    | 45.805            |

>**Test Instructions**:
1. The performance test results have certain volatility;
2. 'calculate time' has been converted to the average inference time per image;
The test results of SoC and PCIe are basically consistent.
### 7.2 Program Running Performance
Refer to the [C++ routine](cpp/README_EN.md) or [Python routine](Python/README_EN.md) to run the program and view the statistical decoding time, preprocessing time, inference time, and post-processing time. The preprocessing time, inference time, and post-processing time printed by C++routines are the processing time of the entire batch, which needs to be divided by the corresponding batch size to determine the processing time of each image.

Using different routines and models to test 'datasets/val2017' on different testing platforms_ 1000`，conf_ thresh=0.5，nms_ Threshold=0.5, and the performance test results are as follows:

|Test Platform| Test Program     | Test Model                          |tot_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | --------- | --------- | --------- |
| BM1684 SoC  | bert_sail.py     | bert4torch_output_fp32_1b.bmodel    | 307.64   | 3.48      | 171.87    | 132.26    |
| BM1684 SoC  | bert_sail.py     | bert4torch_output_fp32_8b.bmodel    | 168.89   | 3.46      | 147.06    | 18.359    |
| BM1684 SoC  | bert_sail.soc    | bert4torch_output_fp32_1b.bmodel    | 19.52    | 19.11     | 0.35      | 0.022     |
| BM1684 SoC  | bert_sail.soc    | bert4torch_output_fp32_8b.bmodel    | 20.21    | 19.34     | 0.830     | 0.021     |
| BM1684X SoC | bert_sail.py     | bert4torch_output_fp32_1b.bmodel    | 225.16   | 3.50      | 92.25     | 129.39    |
| BM1684X SoC | bert_sail.py     | bert4torch_output_fp32_8b.bmodel    | 109.60   | 3.5       | 87.76     | 18.36     |
| BM1684X SoC | bert_sail.py     | bert4torch_output_fp16_1b.bmodel    | 141.59   | 3.5       | 9.50      | 128.57    |
| BM1684X SoC | bert_sail.py     | bert4torch_output_fp16_8b.bmodel    | 27.64    | 3.4       | 5.84      | 18.325    |
| BM1684X SoC | bert_sail.soc    | bert4torch_output_fp32_1b.bmodel    | 19.45    | 19.14     | 0.028     | 0.022     |
| BM1684X SoC | bert_sail.soc    | bert4torch_output_fp32_8b.bmodel    | 19.28    | 19.15     | 0.078     | 0.021     |
| BM1684X SoC | bert_sail.soc    | bert4torch_output_fp16_1b.bmodel    | 19.87    | 19.59     | 0.218     | 0.020     |
| BM1684X SoC | bert_sail.soc    | bert4torch_output_fp16_8b.bmodel    | 19.73    | 19.62     | 0.642     | 0.019     |

>**Test Instructions**:
1. The time unit is milliseconds (ms), and the statistical time is the average processing time of each text;
2. The performance test results have certain volatility, and it is recommended to take the average value after multiple tests;
3. The main control CPUs of the BM1684/1684X SoC are all 8-core ARM A53 42320 DMIPS @ 2.3GHz, and the performance on PCIe may vary significantly due to different CPUs;

## 8. FAQ
Please refer to [FAQ](../../docs/FAQ_EN.md) for some common questions and answers.