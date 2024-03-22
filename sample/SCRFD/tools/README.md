[简体中文](./README.md) | [English](./README_EN.md)
# SCRFD 精度评测脚本使用指南

## 目录
* [1. 简介](#1-简介)
* [2. 数据及环境准备](#2-数据及环境准备)
    * [2.1 数据集说明](#21-数据集说明)
    * [2.2 结果数据格式](#22-结果数据格式)
    * [2.3 环境准备及共享对象文件编译](#23-环境准备及共享对象文件编译)
* [3. 数据生成](#3-数据生成)
* [4. 测试方法](#4-测试方法)


## 1. 简介

本项目评测代码为基于WIDER FACE验证集的测评代码，由于其源代码为matlab版本不便于运行，故而基于其源代码修改。

**数据集地址** (http://shuoyang1213.me/WIDERFACE/)

**官方源码下载地址** (http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip)


本项目的目录所包含的文件如下：
```bash
.
├── box_overlaps.pyx              # Cython 语言的源代码文件，C拓展模块
├── evaluation.py                 # 评估脚本
├── files                         
│   └── download.png              
├── ground_truth                  # ground_truth数据目录
│   ├── wider_easy_val.mat        
│   ├── wider_face_val.mat        
│   ├── wider_hard_val.mat        
│   └── wider_medium_val.mat      
├── prediction_dir                # 预测结果存储目录
│   ├── 0--Parade
│       ├──  0_Parade_marchingband_1_20.txt   # 预测结果
│   ...
│   └── 61--Street_Battle
├── README.md                     # 项目说明
└── setup.py                      # 编译Cython，生成拓展模块
```

## 2. 数据及环境准备
### 2.1 数据集说明
为方便大家下载，本项目已将数据集的下载集成在 `./scripts/download.sh` 脚本内，
您只需运行上述脚本即可。根据下载的数据集，请将其按照下面的文件夹格式放置：
```bash
.
├── datasets
... ├── face_det.mp4
    ├── test
    │   ├── men.jpg
    │   └── selfie.jpg
    └── WIDER_val
        └── images
            ├── 0--Parade
                ├──  0_Parade_marchingband_1_20.jpg
                ...
            ...
            └── 61--Street_Battle
```
其中，0--Parade是不同场景的文件夹，`WIDER FACE` 总共有61种场景，
如上目录所示，其中`0_Parade_marchingband_1_20.jpg` 是某场景下的图片。

### 2.2 结果数据格式
生产的数据会自动保存在： `./prediction_dir` 目录下，对应的 groud truth 数据
会在执行完 `./scripts/download.sh` 脚本后，自动解压到 `./ground_truth`目录下。

生成以下预测结果：  
```bash
.
├── prediction_dir
│   ├── 0--Parade
        ├── 0_Parade_marchingband_1_20.txt
        ├── 0_Parade_marchingband_1_74.txt
    ...
    └── 61--Street_Battle
```
`0--Parade` 表示不同场景类别的文件夹，而 `WIDER FACE` 数据集包含总计 61 种不同的场景类别。文件 `0_Parade_marchingband_1_20.txt` 包含了特定图片的预测结果，其格式如下所示：
```
image_name
the number fo faces  #  共计检测出多少张人脸
x, y, w, h, confidence  #  注意：x,y是检测框左上角的坐标，w,h为检测框的宽高，confidence是置信度
```
### 2.3 环境准备及共享对象文件编译
此外您还需要配置opencv等其他第三方库：
```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
然后，您需要编译对应的Cython源文件的C拓展模块，请参考下面指令执行：
```bash
cd tools
python3 setup.py build_ext --inplace
```

## 3. 数据生成
执行数据验证集的推理，并将结果存储在 `prediction_dir` 目录下。本案例提供两种不同模式的结果生成，
分别是 `scrfd_opencv.py` 和 `scrfd_bmcv.py`，您可以根据需要自行选择对应的程序。

scrfd_opencv.py 和 scrfd_bmcv.py的参数一致，以 scrfd_opencv.py 为例：
```bash
usage: scrfd_opencv.py [--input INPUT_PATH] [--bmodel BMODEL] [--dev_id DEV_ID]
                        [--conf_thresh CONF_THRESH] [--nms_thresh NMS_THRESH]
--input: 测试数据路径，请输入整个图片文件夹的路径；
--bmodel: 用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--dev_id: 用于推理的tpu设备id；
--conf_thresh: 置信度阈值，设置为0.02；
--nms_thresh: nms阈值，设置为0.45；
--eval: 是否为精度评测使用,仅可在测试图片时使用，默认为False;
```
请注意，下面的数据生成共计3226张图片，预计需要花费10分钟左右。
```bash
cd ./sophon-demo/sample/SCRFD/
python3 ./python/scrfd_bmcv.py --input ./datasets/WIDER_val/ --bmodel ./models/BM1684X/scrfd_10g_kps_fp32_1b.bmodel  --dev_id 0 --conf_thresh 0.02 --nms_thresh 0.45 --eval True
```

## 4. 测试方法

经过以上的环境准备和数据准备，您可以运行下面的命令来对模型进行评估，我们提供了两种不同的测试方式，您可以自主选择。

evaluation.py的参数如下：
```bash
usage: evaluation.py [--pred prediction_path] [--gt ground_truth_path] [--all store_true]
--pred: 预测结果的路径，请输入整个图片文件夹的路径; 
--gt: ground_truth路径，请输入整个图片文件夹路径;
--all: 是否进行综合评测，会将三种难度（easy,medium,hard）的数据集图片进行，默认不开启;
```

```bash
cd ./sophon-demo/sample/SCRFD/tools
python3 evaluation.py --pred ./prediction_dir --gt ground_truth #--all
```

