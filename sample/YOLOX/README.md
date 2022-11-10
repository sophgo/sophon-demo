# YOLOx

## 目录
- [YOLOx](#yolox)
  - [目录](#目录)
  - [1.简介](#1简介)
  - [2. 数据集](#2-数据集)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4.模型编译](#4模型编译)
    * [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    * [4.2 生成INT8 BModel](#42-生成int8-bmodel)
  - [5. 部署测试](#5-部署测试)
    - [5.1 使用脚本对示例代码进行自动测试](#51-使用脚本对示例代码进行自动测试)
    - [5.2 C++例程部署测试](#52-c例程部署测试)
    - [5.3 Python例程部署测试](#53-python例程部署测试)

## 1.简介

YOLOx由旷世研究提出,是基于YOLO系列的改进。

**论文地址** (https://arxiv.org/abs/2107.08430)

**官方源码地址** (https://github.com/Megvii-BaseDetection/YOLOX)


## 2. 数据集

[MS COCO](http://cocodataset.org/#home),是微软构建的一个包含分类、检测、分割等任务的大型的数据集.

> MS COCO提供了一些[API](https://github.com/cocodataset/cocoapi),方便对数据集的使用和模型评估,您可以使用pip安装` pip3 install pycocotools`,并使用COCO提供的API进行下载.

## 3. 准备模型与数据

Pytorch的模型在编译前要经过`torch.jit.trace`，trace后的模型才能用于编译BModel。trace的方法和原理可参考[torch.jit.trace参考文档](../docs/torch.jit.trace_Guide.md)。

同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，运行后自动下载pt模型，数据集以及BModel，即可以跳过第4章节模型编译。您也可以使用下载得到的pt模型和量化数据集，或者自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换，生成BModel。
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install unzip
cd ./scripts
chmod +x download.sh
./download.sh
```
执行后，模型保存至`data/models`，测试视频保存至`data/video`，图片数据集下载并解压至`data/image/`，图片数据集集标签文件保存至`data/ground_truths`
其中，/data/image/lmdb为量化数据集，/data/image/val2017为测试集。

```
下载的模型包括：
/data/models/torch/yolox_s.pt: trace后的模型
/data/models/BM1684/yolox_s_fp32_1b.bmodel: 用于BM1684的FP32 BModel，batch_size=1
/data/models/BM1684/yolox_s_fp32_4b.bmodel: 用于BM1684的FP32 BModel，batch_size=4
/data/models/BM1684/yolox_s_int8_1b.bmodel: 用于BM1684的INT8 BModel，batch_size=1
/data/models/BM1684/yolox_s_int8_4b.bmodel: 用于BM1684的INT8 BModel，batch_size=4
/data/models/BM1684X/yolox_st_fp32_1b.bmodel: 用于BM1684X的FP32 BModel，batch_size=1
/data/models/BM1684X/yolox_s_fp32_4b.bmodel: 用于BM1684X的FP32 BModel，batch_size=4
/data/models/BM1684X/yolox_s_int8_1b.bmodel: 用于BM1684X的INT8 BModel，batch_size=1
/data/models/BM1684X/yolox_s_int8_4b.bmodel: 用于BM1684X的INT8 BModel，batch_size=4

下载的数据集包括：
/data/image/lmdb：用于量化的lmdb数据集
/data/image/val2017：用于测试的图片数据集

下载的标签文件包括：
/data/ground_truths/instances_val2017.json ：测试图片集val2017的标签文件

下载的测试视频包括：
/data/video/1080_1.mp4：1080p测试视频
```

## 4. 模型编译

trace后的pytorch模型需要编译成BModel才能在SOPHON TPU上运行，如果直接使用上一步下载好的BModel可跳过本节。

您可以使用上一步下载的，位于data/models/torch/yolox_s.pt的pt模型，以及位于/data/image/lmdb的量化数据集，也可以使用自己trace完成的pt模型，和量化数据集。

模型编译前需要安装TPU-NNTC(>=3.1.0)，具体可参考[tpu-nntc环境搭建](../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。

### 4.1 生成FP32 BModel

pytorch模型编译为FP32 BModel，具体方法可参考TPU-NNTC开发参考手册。

本例程在`scripts`目录下提供了编译FP32 BModel的脚本。

```bash
cd ./scripts
chmod +x gen_fp32bmodel.sh
./gen_fp32bmodel.sh
```

执行上述命令会在`data/models/`下生成BM1684和BM1684X下的1_batch和4_batch的fp32 bmodel文件，即转换好的FP32 BModel。


### 4.2 生成INT8 BModel

不量化模型可跳过本节。

pytorch模型的量化方法可参考TPU-NNTC开发参考手册。

本例程在`scripts`目录下提供了量化INT8 BModel的脚本。

```bash
cd ./scripts
chmod +x gen_int8bmodel.sh
./gen_int8bmodel.sh
```

执行上述命令会在`data/models/`下生成BM1684和BM1684X下的1_batch和4_batch的int8 bmodel文件，即转换好的int8 BModel。

> **模型量化建议：**   
1.尝试不同的iterations进行量化可能得到较明显的精度提升；  
2.对输入输出层保留浮点计算可能得到较明显的精度提升。

## 5. 部署测试

### 5.1 使用脚本对示例代码进行自动测试

此自动测试脚本需要在挂载有PCIe加速卡的x86主机或者sophon soc设备内进行

准备好BModel与测试数据后：

```bash
cd scripts
chmod +x auto_test.sh
./auto_test.sh ${platform} ${target} ${tpu_id} ${sail_dir}
```
其中platform指所在平台（x86 or soc），target是芯片型号（BM1684 or BM1684X），tpu_id指定tpu的id（使用bm-smi查看），sail_dir是sail的安装路径。如果最终输出 `Failed:`则表示执行失败，否则表示成功。并且在根路径下生成mAP文件夹，其中保存着mAP结果。\
例如 auto_test.sh x86 BM1684 0 /opt/sophon/sophon-sail \

在x86上，auto_test.sh包括了cpp文件夹下c++程序的编译，运行和python文件夹下所有python程序的运行，以及mAP计算脚本的运行。\
在soc上，auto_test.sh包括了cpp文件夹下c++程序的运行和python文件夹下所有python程序的运行，以及mAP计算脚本的运行。

在x86上执行此脚本，首先参见[x86-pcie平台的开发和运行环境搭建](../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)，然后运行此脚本，其中${sail_dir}为上述环境搭建得到的sophon-sail安装路径，通常为/opt/sophon/sophon-sail。

在soc上执行此脚本，首先需要在x86平台交叉编译出arm程序(参见[交叉编译环境搭建](../docs/Environment_Install_Guide.md#31-交叉编译环境搭建).)，然后把生成的可执行文件移动到cpp文件夹下。之后设置环境变量
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
```
 再运行此脚本，其中${sail_dir}为上述环境搭建得到的build_soc/sophon-sail文件夹。



### 5.2 C++例程部署测试

详细步骤参考cpp下Readme.md

### 5.3 Python例程部署测试

详细步骤参考python下Readme.md
