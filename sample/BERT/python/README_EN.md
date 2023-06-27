#Python routines

##Table of Contents

* [1. Environmental Preparation](#1-Environmental-preparation)
    * [1.1 x86/arm PCIe platform](#11-x86/arm-pcie-platform)
    * [1.2 SoC Platform](#12-soc-Platform)
* [2. Inference testing](#2-Inference-testing)
    * [2.1 Parameter Description](#21-Parameter-Description)
    * [2.2 Test Text](#22-Test-Text)
    * [2.3 Dev Text](#23-Dev-Text)
    * [2.4 Test Dataset](#24-Test-Dataset)

A series of Python routines are provided in the Python directory, as follows:

|Serial Number | Python Routine | Description|
| ---- | ---------------- | ----------------------------------- |
| 1 | bert_ Sail. py | Using SAIL Reasoning|

## 1. Environmental preparation
### 1.1 x86/arm PCIe platform

If you have installed a PCIe accelerator card (such as the SC series accelerator card) on the x86/arm platform and used it to test this routine, you need to install libsophon, sophon opencv, sophon ffmpeg, and sophon mail, For specific details, please refer to [Development and Running Environment Construction of x86 psie Platform](../../../docs/Environment_INSTALL_Guide_EN.md#3-x86-PCIe-Platform-Development-and-Runtime-Environment-Construction) or [Development and Running Environment Construction of Arm pcie-Platform](../../../docs/Environment_INSTALL_Guide_EN.md#5-Arm-pcie-Platform-Development-and-Runtime-Environment-Construction).

In addition, you may also need to install other third-party libraries:
```bash
pip3 install bert4torch
```

### 1.2 SoC Platform

If you use the SoC platform (such as SE, SM series Edge device) and test this routine with it, the corresponding libsophon, sophon opencv, and sophon ffmpeg runtime library packages have been pre installed under '/opt/sophon/' after the reboot. You also need to cross compile and install Sophon ail, please refer to [Cross Compile and Install Sophon ail](../../../docs/Environment_install_Guide_EN.md#42-Cross-compiling-and-sophon-sail-Installation) for details.

In addition, you may also need to install other third-party libraries:
```bash
pip3 install bert4torch
```

## 2. Inference testing
Python routines do not need to be compiled and can be run directly. The testing parameters and running methods for PCIe and SoC platforms are the same.
### 2.1 Parameter Description
```bash
usage: bert_sail.py [--input INPUT] [--bmodel BMODEL] [--dev_id DEV_ID]
--input: Test data, which can be input as text or the entire text file
--bmodel: The bmodel path used for inference, which defaults to using the network of stage 0 for inference
--dev_id: tpu device id used for inference
--if_crf: whether to enable the crf layer
--dict_path: pre trained model dictionary


```
### 2.2 Test Text
The text test example is as follows
```bash
cd python
python3 bert_sail.py --input ../datasets/china-people-daily-ner-corpus/test.txt --bmodel ../models/BM1684/bert4torch_output_fp32_1b.bmodel --dev_id 0 
```

After the test is completed, output the results

### 2.3 Dev Text
The text test example is as follows, which supports online command line testing.
Enter a piece of text, such as, Zhang Fei, Liu Bei, Guan Yu, Taoyuan, Sanjieyi.
Extract entities {('Zhang Fei', 'PER'), ('Guan Yu', 'PER'), ('Taoyuan', 'LOC'), ('Liu Bei', 'PER')}
```bash
python3 bert_sail.py --input dev --bmodel .. /models/BM1684/bert4torch_output_fp32_1b.bmodel --dev_id 0 
```
The command line enters a text and outputs the result

### 2.4 Test Dataset
The example of image testing is as follows, which supports testing the entire image folder.
```bash
python3 python/bert_sail.py --input datasets/test --bmodel models/BM1684//bert4torch_output_fp32_1b.bmodel --dev_id 0 
```
After the test is completed, the predicted results are saved in 'results/bert4torch_output_fp32_1b.bmodel_test_sail_python_result.txt', information such as prediction results and inference time will also be printed.