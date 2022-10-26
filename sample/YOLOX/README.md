# YOLOx

## 目录
- [YOLOx](#yolox)
  - [目录](#目录)
  - [1.简介](#1简介)
  - [2. 数据集](#2-数据集)
  - [3. 准备模型与数据](#3-准备模型与数据)
    - [3.1 准备模型](#31-准备模型)
      - [**3.1.1 下载yolovx源码**](#311-下载yolovx源码)
      - [**3.1.2 导出JIT模型**](#312-导出jit模型)
    - [3.2 准备量化集](#32-准备量化集)
      - [**3.2.1 准备量化图片**](#321-准备量化图片)
      - [**3.2.2 使用不同的resize方法对图片进行扩展**](#322-使用不同的resize方法对图片进行扩展)
    - [3.2.3 生成lmdb数据](#323-生成lmdb数据)
  - [4.模型转换](#4模型转换)
    - [4.1 使用脚本快速生成BMODEL](#41-使用脚本快速生成bmodel)
    - [4.2 生成FP32 BModel](#42-生成fp32-bmodel)
      - [**4.2.1 生成FP32 BModel**](#421-生成fp32-bmodel)
      - [**4.2.2 查看FP32 BModel**](#422-查看fp32-bmodel)
    - [4.3 生成INT8 BModel](#43-生成int8-bmodel)
      - [**4.3.1 生成FP32 UModel**](#431-生成fp32-umodel)
      - [**4.3.2 生成INT8 UModel**](#432-生成int8-umodel)
      - [**4.3.3 生成INT8 BModel**](#433-生成int8-bmodel)
      - [**4.3.4 查看INT8 BModel**](#434-查看int8-bmodel)
  - [5. 部署测试](#5-部署测试)
    - [5.1 环境配置](#51-环境配置)
    - [5.2 使用脚本对示例代码进行自动测试](#52-使用脚本对示例代码进行自动测试)
    - [5.3 C++例程部署测试](#53-c例程部署测试)
    - [5.4 Python例程部署测试](#54-python例程部署测试)

## 1.简介

YOLOx由旷世研究提出,是基于YOLO系列的改进。

**论文地址** (https://arxiv.org/abs/2107.08430)

**官方源码地址** (https://github.com/Megvii-BaseDetection/YOLOX)


## 2. 数据集

[MS COCO](http://cocodataset.org/#home),是微软构建的一个包含分类、检测、分割等任务的大型的数据集.

> MS COCO提供了一些[API](https://github.com/cocodataset/cocoapi),方便对数据集的使用和模型评估,您可以使用pip安装` pip3 install pycocotools`,并使用COCO提供的API进行下载.

## 3. 准备模型与数据

您可以按以下步骤准备yolox的pt模型，生成bmodel，下载数据集，也可以运行scripts/downloads.sh来下载pt模型，准备好的BM1684和BM1684X的fp32和int8 bmodel，val2017测试集和groundtruth，以及准备好的量化数据集。

### 3.1 准备模型  

- 因为上述docker已经安装了pytorch，但是版本较yolox版本要求的版本低一些，所以此步骤不建议在docker内进行，最好在物理机上直接进行

- YOLOx模型的模型参数  
  
| Model                                       | size | mAP<sup>val<br>0.5:0.95 | mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) | FLOPs<br>(G) |                           weights                            |
| ------------------------------------------- | :--: | :---------------------: | :----------------------: | :----------------: | :-----------: | :----------: | :----------------------------------------------------------: |
| [YOLOX-s](./exps/default/yolox_s.py)        | 640  |          40.5           |           40.5           |        9.8         |      9.0      |     26.8     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
| [YOLOX-m](./exps/default/yolox_m.py)        | 640  |          46.9           |           47.2           |        12.3        |     25.3      |     73.8     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
| [YOLOX-l](./exps/default/yolox_l.py)        | 640  |          49.7           |           50.1           |        14.5        |     54.2      |    155.6     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
| [YOLOX-x](./exps/default/yolox_x.py)        | 640  |          51.1           |           51.5           |        17.3        |     99.1      |    281.9     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |
| [YOLOX-Darknet53](./exps/default/yolov3.py) | 640  |          47.7           |           48.0           |        11.1        |     63.7      |    185.3     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth) |


#### **3.1.1 下载yolovx源码**

 ```bash
 # 下载yolox源码
 git clone https://github.com/Megvii-BaseDetection/YOLOX
 # 切换到yolox工程目录
 cd YOLOX
 # 安装依赖
 pip3 install -r requirements.txt
 ```

#### **3.1.2 导出JIT模型**

 SophonSDK中的PyTorch模型编译工具BMNETP只接受PyTorch的JIT模型（TorchScript模型）.

 Pytorch的模型在编译前要经过`torch.jit.trace`，trace后的模型才能用于编译BModel。trace的方法和原理可参考[torch.jit.trace参考文档](../docs/torch.jit.trace_Guide.md)。

 JIT（Just-In-Time）是一组编译工具，用于弥合PyTorch研究与生产之间的差距。它允许创建可以在不依赖Python解释器的情况下运行的模型，并且可以更积极地进行优化。在已有PyTorch的Python模型（基类为torch.nn.Module）的情况下，通过torch.jit.trace就可以得到JIT模型,如`torch.jit.trace(python_model, torch.rand(input_shape)).save('jit_model')`。BMNETP暂时不支持带有控制流操作（如if语句或循环）的JIT模型，因此不能使用`torch.jit.script`，而要使用`torch.jit.trace`，它仅跟踪和记录张量上的操作，不会记录任何控制流操作。这部分操作YOLOX已经为我们写好，只需运行如下命令即可导出符合要求的JIT模型：

- YOLOX-s
  ```bash
    python3 tools/export_torchscript.py -n yolox-s -c ${PATH_TO_YOLOX_MODEL}/yolox_s.pth --output-name ${PATH_TO_YOLOX_MODEL}/yolox_s.trace.pt
  ```
- YOLOX-m
  ```bash
    python3 tools/export_torchscript.py -n yolox-m -c ${PATH_TO_YOLOX_MODEL}/yolox_m.pth --output-name ${PATH_TO_YOLOX_MODEL}/yolox_m.trace.pt
  ```
- YOLOX-l
  ```bash
    python3 tools/export_torchscript.py -n yolox-l -c ${PATH_TO_YOLOX_MODEL}/yolox_l.pth --output-name ${PATH_TO_YOLOX_MODEL}/yolox_l.trace.pt
  ```
- YOLOX-x
  ```bash
    python3 tools/export_torchscript.py -n yolox-x -c ${PATH_TO_YOLOX_MODEL}/yolox_x.pth --output-name ${PATH_TO_YOLOX_MODEL}/yolox_x.trace.pt
  ```
- YOLOX-Darknet53
  ```bash
    python3 tools/export_torchscript.py -n yolov3 -c ${PATH_TO_YOLOX_MODEL}/yolox_darknet.pth --output-name ${PATH_TO_YOLOX_MODEL}/yolox_darknet.trace.pt
  ```

上述脚本会在 `${PATH_TO_YOLOX_MODEL}` 下生成相应的JIT模型

### 3.2 准备量化集

此步骤需要在开发主机的docker内进行，不量化模型可以跳过本节。

#### **3.2.1 准备量化图片**

示例从coco数据集中随机选取了部分图片，保存在docker内的路径为：${OST_DATA_PATH}

#### **3.2.2 使用不同的resize方法对图片进行扩展**

- 使用opencv做resize and padding

  ```bash
  python3 ./tools/image_resize.py --ost_path=${OST_DATA_PATH} --dst_path=${RESIZE_DATA_PATH} --dst_width=640 --dst_height=640
  ```
  结果图片将保存在`${RESIZE_DATA_PATH}`中


### 3.2.3 生成lmdb数据

  ```bash
    python3 ./tools/convert_imageset.py \
        --imageset_rootfolder=${RESIZE_DATA_PATH} \
        --imageset_lmdbfolder=${LMDB_PATH} \
        --resize_height=640 \
        --resize_width=640 \
        --shuffle=True \
        --bgr2rgb=False \
        --gray=False
  ```
  结果lmdb将保存的`${LMDB_PATH}`中


## 4.模型转换

模型转换的过程需要在x86下的docker开发环境中完成。fp32模型的运行验证可以在挂载有PCIe加速卡的x86-docker开发环境中进行，也可以在盒子中进行，且使用的原始模型为JIT模型。下面以YOLOX-s为例，介绍如何完成模型的转换。

模型编译前需要安装tpu-nntc，具体可参考[tpu-nntc环境搭建](../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。

### 4.1 使用脚本快速生成BMODEL

```bash
  ./gen_fp32bmodel.sh
  ./gen_int8bmodel.sh
```

在../data/models/下分别生成适用于BM1684和BM1684X的fp32，int8的1batch和4batch bmodel。
最终输出`Passed: convert to bmodel`表示转换成功,否则转换失败

### 4.2 生成FP32 BModel

#### **4.2.1 生成FP32 BModel**

  ```bash
  python3 -m bmnetp --net_name=yolox_s --target=BM1684 --opt=1 --cmp=true --shapes="[1,3,640,640]" --model=${OST_MODEL_NAME} --outdir=${OUTPUT_MODEL_PATH} --dyn=false
  ```
  其中 `${OST_MODEL_NAME}` 表示原始模型的路径及名称,结果会在`${OUTPUT_MODEL_PATH}`文件夹下面生成,文件夹内的compilation.bmodel即为fp32 bmodel


#### **4.2.2 查看FP32 BModel**

  此步骤可以在开发的docker内进行,也可以在盒子上进行

  ```bash
  tpu_model --info ${BModel_NAME}
  ```
  使用`tpu_model --info`查看的模型具体信息如下：

  ```bash
  bmodel version: B.2.2
  chip: BM1684
  create time: Tue Mar 29 12:04:18 2022

  ==========================================
  net 0: [yolox_s]  static
  ------------
  stage 0:
  input: x.1, [1, 3, 640, 640], float32, scale: 1
  output: 15, [1, 8400, 85], float32, scale: 1
  ```

### 4.3 生成INT8 BModel

此过程需要在x86下的docker开发环境中完成，不量化模型可跳过本节。

INT8 BModel的生成需要经历中间格式UModel，即：原始模型→FP32 UModel→INT8 UModel→INT8 BModel。

#### **4.3.1 生成FP32 UModel**

  执行以下命令，将依次调用以下步骤中的脚本，生成INT8 BModel：

  ```bash
  python3 gen_fp32_umodel.py \
    --trace_model=${OST_MODEL_NAME} \
    --data_path=${LMDB_PATH}/data.mdb \
    --dst_width=640 \
    --dst_height=640
  ```
  结果将在`${OST_MODEL_NAME}`的文件夹下面创建一个以`${OST_MODEL_NAME}`模型名称命名的文件夹`${UMODEL_PATH}`，文件夹内存放的是fp32 umodel。


#### **4.3.2 生成INT8 UModel**

  ```bash
  calibration_use_pb \
    quantize \
    -model=${UMODEL_PATH}/*_bmnetp_test_fp32.prototxt \
    -weights=${UMODEL_PATH}/*_bmnetp.fp32umodel \
    -iterations=100 \
    -bitwidth=TO_INT8
  ```
  ```注意：不同的模型的bmnetp_test_fp32.prototxt和bmnetp.fp32umodel文件名称不同,实际使用时需要替换命令行中的*```

  ```int8 umodel将保存在${UMODEL_PATH}文件夹下```

#### **4.3.3 生成INT8 BModel**

  ```bash
  bmnetu 
    -model=${UMODEL_PATH}/*_bmnetp_deploy_int8_unique_top.prototxt \
    -weight=${UMODEL_PATH}*_bmnetp.int8umodel \
    -max_n=4 \
    -prec=INT8 \
    -dyn=0 \
    -cmp=1 \
    -target=BM1684 \
    -outdir=${OUTPUT_BMODEL_PATH}
  ```
  ```注意：不同的模型的bmnetp_deploy_int8_unique_top.prototxt和bmnetp.int8umodel文件名称不同,实际使用时需要替换命令行中的*```

  ```命令参数中max_n表示生成模型的batchsize,结果bmodel将保存在${OUTPUT_BMODEL_PATH}下```

#### **4.3.4 查看INT8 BModel**

此步骤可以在开发的docker内进行,也可以在盒子上进行

  ```bash
    tpu_model --info ${BModel_NAME}
  ```
使用`tpu_model --info`查看的模型具体信息如下：

```bash
bmodel version: B.2.2
chip: BM1684
create time: Wed Mar 30 19:24:41 2022

==========================================
net 0: [yolox_s.trace_bmnetp]  static
------------
stage 0:
input: x.1, [4, 3, 640, 640], int8, scale: 0.498161
output: 15, [4, 8400, 85], float32, scale: 1
```

## 5. 部署测试

请注意根据您使用的模型，所有例程都使用sail进行推理，内部对batch size和int8 or fp32做了自适应。

### 5.1 环境配置

您需要安装libsophon、sophon-opencv，sophon-ffmpeg，sophon-sail。libsophon的安装可参考[LIBSOPHON使用手册]()，sophon-opencv和sophon-ffmpeg的安装可参考[multimedia开发参考手册]()。sophon-sail的安装可参考[sail开发参考手册]()

### 5.2 使用脚本对示例代码进行自动测试

此自动测试脚本需要在挂载有PCIe加速卡的x86-pcie-docker或者sophon soc设备内进行

配置好环境变量安装好对应版本的sail之后执行：

```bash
cd scripts
./auto_test.sh ${platform} ${target} ${tpu_id} ${sail_dir} ${soc_sdk}
```
例如 auto_test.sh x86 BM1684 0 /opt/sophon/sophon-sail soc-sdk \
auto_test.sh包括了cpp文件夹下c++程序的编译，运行和python文件夹下所有python程序的运行，以及mAP计算脚本的运行。\
如果已经在x86平台交叉编译过soc程序，则需要把生成的可执行文件移动到cpp文件夹下，并且注释掉184行build_cpp $platform $sail_dir $SDK_dir 

其中platform指所在平台（x86 or soc），target是芯片型号（BM1684 or BM1684X），tpu_id指定tpu的id（使用bm-smi查看），sail_dir是sail的安装路径，soc_sdk是soc_sdk的路径。如果最终输出 `Failed:`则表示执行失败，否则表示成功。并且在根路径下生成mAP文件夹，其中保存着mAP结果。

### 5.3 C++例程部署测试

详细步骤参考cpp下Readme.md

### 5.4 Python例程部署测试

详细步骤参考python下Readme.md
