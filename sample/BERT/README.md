# BERT

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
  * [4.1 TPU-NNTC编译BModel](#41-tpu-nntc编译bmodel)
  * [4.2 TPU-MLIR编译BModel](#42-tpu-mlir编译bmodel)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-faq)
  
## 1. 简介
​BERT的全称为Bidirectional Encoder Representation from Transformers，是一个预训练的语言表征模型。它强调了不再像以往一样采用传统的单向语言模型或者把两个单向语言模型进行浅层拼接的方法进行预训练，而是采用新的masked language model（MLM），以致能生成深度的双向语言表征。BERT论文发表时提及在11个NLP（Natural Language Processing，自然语言处理）任务中获得了新的state-of-the-art的结果，令人目瞪口呆。
A simple training framework that recreates bert4keras in PyTorch. bert4torch
## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC)
* 支持FP32、FP16(BM1684X)模型编译和推理
* 支持基于sail的C++推理
* 支持基于sail的Python推理
* 支持单batch和多batch模型推理
* 支持文本测试
 
## 3. 准备模型与数据
如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel，Pytorch模型在编译前要导出成torchscript模型或onnx模型；如果您使用BM1684X芯片，建议使用TPU-MLIR编译BModel，Pytorch模型在编译前要导出成onnx模型。具体可参考[BERT模型导出](./docs/BERT4torch_Exportonnx_Guide.md)。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
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
下载的数据包括：
```
./datasets/china-people-daily-ner-corpus
├── example.dev                                              # 验证集
├── example.test                                             # 测试集
└── example.train                                            # 训练集
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。如果您使用BM1684芯片，建议使用TPU-NNTC编译BModel；如果您使用BM1684X芯片，建议使用TPU-MLIR编译BModel。

### 4.1 TPU-NNTC编译BModel
模型编译前需要安装TPU-NNTC，具体可参考[TPU-NNTC环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。安装好后需在TPU-NNTC环境中进入例程目录。

- 生成FP32 BModel

使用TPU-NNTC将trace后的torchscript模型编译为FP32 BModel，具体方法可参考《TPU-NNTC开发参考手册》的“BMNETP 使用”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

​本例程在`scripts`目录下提供了TPU-NNTC编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_nntc.sh`中的torchscript模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
./scripts/gen_fp32bmodel_nntc.sh BM1684
```

​执行上述命令会在`models/BM1684/`下生成`bert4torch_output_fp32_1b.bmodel`文件，即转换好的FP32 BModel。


### 4.2 TPU-MLIR编译BModel
模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#2-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

注意：mlir版本需要>=0625的版本为v1.2.2。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`bert4torch_output_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684X），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`下生成`bert4torch_output_fp16_1b.bmodel`文件，即转换好的FP16 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试文本)或[Python例程](python/README.md#22-测试文本)推理要测试的数据集，生成预测的txt文件，注意修改数据集(datasets/china-people-daily-ner-corpus)。  
然后，使用`tools`目录下的`eval_people.py`脚本，将测试生成的txt文件与测试集标签txt文件进行对比，计算出NER的评价指标，命令如下：
```bash
# 安装seqeval，若已安装请跳过
pip3 install seqeval
# 请根据实际情况修改程序路径和txt文件路径
python3 tools/eval_people.py --test_path ../datasets/china-people-daily-ner-corpus/example.test --input_path ../python/results/bert4torch_output_fp16_8b.bmodel_sail_python_result.txt
```
### 6.2 测试结果
在china-people-daily-ner-corpus数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |              测试模型               |f1             |accuary   |
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


> **测试说明**：  
1. 测试结果具有一定的波动性；基本在0.01之内


## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/bert4torch_output_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                   | calculate time(ms) |
| ------------------------------------------- | ----------------- |
| BM1684/bert4torch_output_fp32_1b.bmodel     | 170.848           |
| BM1684/bert4torch_output_fp32_8b.bmodel     | 146.653           |
| BM1684X/bert4torch_output_fp32_1b.bmodel    | 91.473            |
| BM1684X/bert4torch_output_fp16_1b.bmodel    | 8.643             |
| BM1684X/bert4torch_output_fp32_8b.bmodel    | 87.478            |
| BM1684X/bert4torch_output_fp16_8b.bmodel    | 5.726            |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/china-people-daily-ner-corpus/example.test`，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型                |tot_time|preprocess_time|inference_time|postprocess_time| 
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

> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为平均每个文本处理的时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异； 

## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。
如果执行python例程bert_sail.py时报错"ImportError: /home/ljtang/miniconda3/lib/libstdc+.so.6: version `GLIBCXX_3.4.30' not found (required by /home/ljtang/miniconda3/envs/torch1.13/lib/python3.10/site-packages/sophon/sail.cpython-310-x86_64-linux-gnu.so)"，无法导入sophon.sail，使用”strings /usr/lib/x86_64-linux-gnu/libstdc+.so.6 | grep GLIBCXX“查询到兼容版本，使用命令”export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH“解决问题
在解决完以上问题后，如果执行eval_people.py时报错”ImportError: xxx/miniconda3/envs/torch1.13/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by xxx/miniconda3/envs/torch1.13/lib/python3.10/site-packages/sophon/sail.cpython-310-x86_64-linux-gnu.so)“，在eval_people.py的” from python.bert_sail import dataset“之前增加”import sophon.sail as sail“解决问题