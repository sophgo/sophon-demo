# CenterNet Python例程

## 目录

[TOC]

## 1. 目录说明

 python目录下提供了python例程以供参考使用，目录结构如下：

```bash
.
├── base_detector.py
├── ctdet.py
├── debugger.py
├── centernet_bmcv.py  # 测试程序
└── README.md
```

## 2. 测试

### 2.1 PCIe模式

#### 2.1.1 环境配置

硬件环境：x86平台，并安装了含有1684或1684X的PCIE加速卡

软件环境：libsophon、sophon-mw、sophon-sail，可以通过[算能官网](https://developer.sophgo.com/site/index/material/21/all.html)下载安装对应版本

安装python opencv

```bash
pip3 install opencv-python-headless==4.3.0.36
```

> 运行之前需要访问sophon-sail github仓库编译安装sail包

Python代码无需编译，无论是x86 SC平台还是arm SE5平台配置好环境之后就可直接运行。

#### 2.1.2 测试

可执行程序默认有一套参数，也可以根据具体情况传入对应参数，参数说明如下：

```bash
usage: centernet_bmcv.py [-h] [--input INPUT] [--loops LOOPS] [--tpu_id TPU_ID] [--bmodel BMODEL] [--class_path CLASS_PATH]

Demo for CenterNet

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT
  --loops LOOPS
  --tpu_id TPU_ID
  --bmodel BMODEL
  --class_path CLASS_PATH
```

 demo中支持图片测试，按照实际情况传入路径参数即可。另外，模型支持fp32bmodel、int8bmodel，可以通过传入模型路径参数进行测试：

```bash
# 1batch
python3 centernet_bmcv.py --bmodel=../data/models/BM1684/centernet_fp32_1b.bmodel --input=../data/ctdet_test.jpg
# 执行完毕后，在./results生成centernet_result_20xx-xx-xx-xx-xx-xx_b_x.jpg格式的图片
# 图片上检测出11个目标

# 4batch
python3 centernet_bmcv.py --bmodel=../data/models/BM1684/centernet_int8_4b.bmodel --input=../data/ctdet_test.jpg
# 执行完毕后，在./results生成centernet_result_20xx-xx-xx-xx-xx-xx_b_x.jpg格式的图片
# 按照量化结果差异，图片上检测出11-13个目标，均属正常范围
```

1. 如果是fp32的模型，图片有11个框
2. 如果是int8的模型，按照量化结果差异，图片上检测出11-13个目标，均属正常范围

>  **使用SAIL模块的注意事项：**对于INT8 BModel来说，当输入输出为int8时，含有scale，需要在处理时将输入输出乘以相应的scale。使用SAIL接口推理时，当sail.Engine.process()接口输入为numpy时，SAIL内部会自动乘以scale，用户无需操作；而输入为Tensor时，需要手动在数据送入推理接口前乘以scale。
> 这是因为Tensor作为输入的话，一般图像来源就是bm_image，这样就可以直接调用vpp进行scale等操作，所以推理之前由用户乘以scale更高效；而在python接口中，当numpy作为输入的话，推理之前没办法调用vpp，sail内部使用SSE指令进行了加速。

### 2.2 SoC模式

#### 2.2.1 环境配置

硬件环境：SoC arm平台

软件环境：libsophon、sophon-mw、sophon-sail，可以通过[算能官网](https://developer.sophgo.com/site/index/material/21/all.html)下载安装对应版本

#### 2.2.2  测试

 demo中支持图片测试，将编译好的可执行文件、bmodel、测试图片传输到soc模式的算能设备中，按照实际情况传入路径参数即可。另外，模型支持fp32bmodel、int8bmodel，可以通过传入模型路径参数进行测试。
 例如：在BM1684设备上运行fp32_1batch的模型，需拷贝`sophon-demo/sample/CenterNet/data/models/BM1684/centernet_fp32_1b.bmodel`, 在BM1684X设备上运行int8_4batch的模型，需拷贝`sophon-demo/sample/CenterNet/data/models/BM1684X/centernet_int8_4b.bmodel`,

> 将python整个文件夹拷贝到SE5中，和bmodel和jpg文件同一目录下

```bash
# 测试图片目标检测,SoC mode,SoC环境下运行
cd python
# 1batch
python3 centernet_bmcv.py --bmodel=../centernet_fp32_1b.bmodel --input=../ctdet_test.jpg --class_path=../coco_classes.txt
# 4batch
python3 centernet_bmcv.py --bmodel=../centernet_int8_4b.bmodel --input=../ctdet_test.jpg --class_path=../coco_classes.txt
```

成功后，在当前目录下生成和`centernet_result_20xx-xx-xx-xx-xx-xx_b_x.jpg`图片
