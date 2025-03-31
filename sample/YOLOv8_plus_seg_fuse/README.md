# YOLOv8_plus_seg_fuse

## 目录

- [YOLOv8\_plus\_seg\_fuse](#yolov8_plus_seg_fuse)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 模型编译](#32-模型编译)
  - [4. 例程测试](#4-例程测试)
  - [5. 精度测试](#5-精度测试)
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt\_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  - [8. FAQ](#8-faq)
  
## 1. 简介
YOLOv8_plus​_seg_fuse例程将[YOLOv8_plus_seg例程](../YOLOv8_plus_seg/README.md)的部分后处理使用TPU来计算，大大提高了后处理速度，目前支持使用BM1684X/BM1688的INT8模型推理。

## 2. 特性

### 2.1 目录结构说明
```bash
├── cpp                   # 存放C++例程及其README
|   ├──README.md      
|   ├──yolov8_bmcv        # C++例程
├── docs                  # 存放本例程专用文档，如ONNX导出、移植常见问题等
├── pics                  # 存放README等说明文档中用到的图片
├── python                # 存放Python例程及其README
|   ├──README.md 
|   ├──yolov8_bmcv.py     # Python例程
|   └──...                # Python例程共用功能的封装。
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译、数据下载、自动测试等shell脚本
└── tools                 # 存放精度测试、性能比对等python脚本
```

### 2.2 SDK特性
* 支持BM1688(SoC)和BM1684X(x86 PCIe、SoC、riscv PCIe)
* 支持INT8模型编译和推理
* 支持C++、Python推理
* 支持图片和视频测试

## 3. 数据准备与模型编译

### 3.1 数据准备

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

```bash
chmod -R +x scripts/
./scripts/download.sh --all 
```

`download.sh`默认只下载`datasets`，`models`可以通过指定参数分平台下载，参数如下：
```bash
--all     # 下载所有模型
--BM1684X # 下载BM1684X的bmodel
--BM1688  # 下载BM1688的bmodel
--onnx    # 下载onnx
```

下载的模型包括：
```bash
models/
├── BM1684X # 在BM1684X上运行的模型
│   ├── yolov8s_int8_1b.bmodel
├── BM1688 # 在BM1688上运行的模型
│   ├── yolov8s_int8_1b.bmodel
│   ├── yolov8s_int8_1b_2core.bmodel
├── onnx
    ├── yolov8s_qtable # 量化yolov8s-seg.onnx时，需要混合精度的层
    ├── yolov8s-seg.onnx
```
下载的数据包括：
```bash
./datasets
├── test                                      # 测试图片
├── test_car_person_1080P.mp4                 # 测试视频
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
└── coco                                      
    ├── val2017_1000                               # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── instances_val2017_1000.json                # coco val2017_1000数据集关键点标签文件，用于计算精度评价指标 
```

### 3.2 模型编译

**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，如果您使用的TPU-MLIR版本>=v1.3.0（即官网v23.07.01），也可以直接使用torchscript模型。具体可参考[模型导出](./docs/YOLOv8_Export_Guide.md)。​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，本例程需要使用特定的mlir版本，请通过如下命令下载：
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_plus_seg_fuse/tpu-mlir_v1.16.beta.0-44-g98f098494-20250324.tar.gz
```
下载完成之后可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)中的发布包安装方式来配置mlir环境。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X/BM1688**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1688
```

​上述脚本会在`models/BM1684X`等文件夹下生成转换好的INT8 BModel。

注：这里用到了混合精度量化，需要将一些层设为敏感层，相应的qtable在此前`download.sh`下载的`models/onnx`文件夹里。如果您需要量化自己微调过的模型，可以参考[量化指南](../../docs/Calibration_Guide.md#13-特定模型优化技巧)中的方法，从我们提供的qtable倒推出自己模型需要的qtable。

## 4. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 5. 精度测试
敬请期待，目前暂不支持精度测试。

## 6. 性能测试
### 6.1 bmrt_test
本例程使用动态输出，使用bmrt_test测试模型理论性能的方式不准确，因此这里不提供bmrt_test的数据。

### 6.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.25，nms_thresh=0.7，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | -------- | --------- | --------- | --------- |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b.bmodel  |      9.15       |      3.23       |      19.33      |      25.84      |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b.bmodel  |      4.00       |      1.19       |      18.10      |      4.97       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s_seg_fuse_int8_1b_2core.bmodel|      5.14       |      3.22       |      14.89      |      25.60      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_seg_fuse_int8_1b_2core.bmodel|      3.48       |      1.19       |      13.64      |      5.02       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。