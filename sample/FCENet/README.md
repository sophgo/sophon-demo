# FCENet

# 目录
* [1. 简介](#1.-简介)
* [2. 特性](#2.-特性)
* [3. 准备模型与数据](#3.-准备模型与数据)
* [4. 模型编译](#4.-模型编译)
	* [4.1. TPU-NNTC编译BModel](#4.1.-TPU-NNTC编译BModel)
* [5. 例程测试](#5.-例程测试)
* [6. 精度与性能测试](#6.-精度与性能测试)
	* [6.1. 精度测试](#6.1.-精度测试)
	* [6.2. 性能测试](#6.2.-性能测试)
	* [6.3. 测试结果](#6.3.-测试结果)





## 1. 简介
FCENet (Fourier Contour Embedding for Arbitrary-Shaped Text Detection) 通过预测一种基于傅里叶变换的任意形状文本包围框表示，从而实现了自然场景文本检测中对于高度弯曲文本实例的检测精度的提升

Paper:
> [Fourier Contour Embedding for Arbitrary-Shaped Text Detection](https://arxiv.org/abs/2104.10442)
> Yiqin Zhu and Jianyong Chen and Lingyu Liang and Zhanghui Kuang and Lianwen Jin and Wayne Zhang
> CVPR, 2021

本例程对[PaddleOCR中训练好的FCENet](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_det_fcenet_en.md)的模型进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

## 2. 特性
- 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC)
- 支持FP32模型编译和推理
- 支持基于OpenCV和BMCV预处理的Python推理
- 支持单batch和多batch模型推理

## 3. 准备模型与数据
Paddle模型编译BModel前需要参考Paddle官方提供的方式导出模型[Paddle模型export](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_det_fcenet_en.md).本例程在`scripts`目录下提供了原始模型、编译模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
chmod +x ./scripts/*
./scripts/download.sh
```
执行后，模型保存至`models`，测试数据集下载并解压至`datasets/`

下载的模型包括：
```bash
./models
├── BM1684
│   ├── fcenet_fp32_b1.bmodel          # 用于BM1684的FP32 BModel，batch_size=1
│   ├── fcenet_fp32_b4.bmodel          # 用于BM1684的FP32 BModel，batch_size=4
├── BM1684X
│   ├── fcenet_fp32_b1.bmodel          # 用于BM1684X的FP32 BModel，batch_size=1
│   ├── fcenet_fp32_b4.bmodel          # 用于BM1684X的FP32 BModel，batch_size=4
└── paddle
    ├── det_r50_dcn_fce_ctw_v2.0_train # Paddle原始模型
    │   ├── best_accuracy.pdparams
    │   ├── best_accuracy.states
    │   └── train.log
    └── inference
        └── det_fce                    # Paddle导出模型
            ├── inference.pdiparams
            ├── inference.pdiparams.info
            └── inference.pdmodel
```

测试数据集ctw1500包括：
```bash
./datasets
└── ctw1500
    └── imgs
        ├── test    
        ├── test_opencv_read_write     # ctw1500测试图片文件夹
        ├── test.txt                   # 测试图片标签
        ├── training
        └── training.txt
```

## 4. 模型编译

Paddle导出模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

### 4.1. TPU-NNTC编译BModel

模型编译前需要安装TPU-NNTC，具体可参考[TPU-NNTC环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。安装好后需在TPU-NNTC环境中进入例程目录。

#### 4.1.1. 生成FP32 BModel

使用TPU-NNTC将Paddle模型编译为FP32 BModel，具体方法可参考《TPU-NNTC开发参考手册》的"BMPADDLE 使用"(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`gen_fp32bmodel.sh`中的Paddle模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
./scripts/gen_fp32bmodel.sh BM1684X
```

执行上述命令会在`models/BM1684X/`下生成`fcenet_fp32_b1.bmodel、fcenet_fp32_b4.bmodel`文件，即转换好的FP32 BModel。


## 5. 例程测试
* [Python例程](python/README.md)

## 6. 精度与性能测试

### 6.1. 精度测试

本例程在`tools`目录下提供了`eval.py`脚本，可以将预测结果文件与测试集标签文件进行对比，计算出指标。具体的测试命令如下：
```bash
# 请根据实际情况修改文件路径
python3 tools/eval.py --gt_path datasets/ctw1500/imgs/test.txt --pred_path python/results/fcenet_fp32_b1.bmodel_test_opencv_read_write_opencv_python_result.json 
```
执行完成后，会打印出指标：
```bash
INFO:root:thr 0.3:precision:0.83344 recall:0.83317 hmean:0.83331
INFO:root:thr 0.4:precision:0.84242 recall:0.82927 hmean:0.83579
INFO:root:thr 0.5:precision:0.84980 recall:0.82244 hmean:0.83589
INFO:root:thr 0.6:precision:0.85963 recall:0.81854 hmean:0.83858
INFO:root:thr 0.7:precision:0.86773 recall:0.81073 hmean:0.83826
INFO:root:thr 0.8:precision:0.87883 recall:0.80195 hmean:0.83863
INFO:root:thr 0.9:precision:0.89701 recall:0.78179 hmean:0.83545
INFO:root:hmean:0.8386328855636798
```
### 6.2. 性能测试

可以使用bmrt_test测试模型的理论性能：
```bash
bmrt_test --bmodel {path_of_bmodel}
```

### 6.3. 测试结果

在BM1684X PCIe上，不同例程、不同模型的精度和性能测试结果如下：


在BM1684X SoC上，不同例程、不同模型的精度和性能测试结果如下：


在BM1684 PCIe上，不同例程、不同模型的精度和性能测试结果如下：


在BM1684 SoC上，不同例程、不同模型的精度和性能测试结果如下：


```
bmrt_test: 使用bmrt_test计算出来的每张图的理论推理时间；
infer_time: 程序运行时每张图的实际推理时间；
QPS: 程序每秒钟全流程处理的图片数。
```

> **测试说明**：  
1. 性能测试的结果具有一定的波动性。