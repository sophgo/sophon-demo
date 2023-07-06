# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 文本检测推理测试](#21-文本检测推理测试)
    * [2.2 文本方向分类推理测试](#22-文本方向分类推理测试)
    * [2.3 文本方向分类推理测试](#23-文本识别分类推理测试)
    * [2.4 全流程推理测试](#24-全流程推理测试)

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程              | 说明                                    | 功能 |
| ----   | ----------------       | ---------------------------             |-    |
| 1      | ppocr_det_opencv.py    | 使用OpenCV解码、OpenCV前处理、SAIL推理   |文本检测|
| 2      | ppocr_cls_opencv.py    | 使用OpenCV解码、OpenCV前处理、SAIL推理   |文本方向分类|
| 3      | ppocr_rec_opencv.py    | 使用OpenCV解码、OpenCV前处理、SAIL推理   |文本识别|
| 4      | ppocr_system_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理   |全流程测试|

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

需要执行以下命令安装所需的python库:
```bash
pip3 install -r requirements.txt
```

如果在运行python例程的过程中遇到 "`No module named '_ctypes'`"的问题需要
```bash
apt-get install libffi-dev
# 进入python安装路径内重新编译安装python
cd python<version>
make && make install
```
### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

需要执行以下命令安装所需的python库:
```bash
pip3 install -r requirements.txt
```

如果在运行python例程的过程中遇到 "`No module named '_ctypes'`"的问题需要
```bash
apt-get install libffi-dev
# 进入python安装路径内重新编译安装python
cd python<version>
make && make install
```

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
## 2.1 文本检测推理测试：
ppocr_det_opencv.py参数说明如下：
```bash
usage: ppocr_det_opencv.py [-h] [--dev_id DEV_ID] [--input INPUT] [--bmodel_det BMODEL_DET]

optional arguments:
  -h, --help            show this help message and exit
  --dev_id DEV_ID       tpu card id
  --input INPUT         input image directory path
  --bmodel_det BMODEL_DET
                        bmodel path
```

文本检测测试实例如下：
```bash
# 程序会自动根据文件夹中的图片数量来选择1batch或者4batch，优先选择4batch推理。
python3 ppocr_det_opencv.py --input ../datasets/cali_set_det --bmodel_det ../models/BM1684X/ch_PP-OCRv3_det_fp32.bmodel --dev_id 0
```
执行完成后，会将预测图片保存在`results/det_results`文件夹下。

## 2.2 文本方向分类推理测试：
ppocr_cls_opencv.py参数说明如下：
```bash
usage: ppocr_cls_opencv.py [-h] [--dev_id DEV_ID] [--input INPUT] [--bmodel_cls BMODEL_CLS]
                           [--cls_thresh CLS_THRESH] [--label_list LABEL_LIST]

optional arguments:
  -h, --help            show this help message and exit
  --dev_id DEV_ID       tpu card id
  --input INPUT         input image directory path
  --bmodel_cls BMODEL_CLS
                        classifier bmodel path
  --cls_thresh CLS_THRESH
  --label_list LABEL_LIST
```

文本方向分类测试实例如下：
```bash
# 程序会自动根据文件夹中的图片数量来选择1batch或者4batch，优先选择4batch推理。
python3 ppocr_cls_opencv.py --input ../datasets/cali_set_rec --bmodel_cls ../models/BM1684X/ch_PP-OCRv3_cls_fp32.bmodel --dev_id 0 --cls_thresh 0.9 --label_list 0,180
```

## 2.3 文本识别分类推理测试：
ppocr_rec_opencv.py参数说明如下：
```bash
usage: ppocr_rec_opencv.py [-h] [--dev_id DEV_ID] [--input INPUT] [--bmodel_rec BMODEL_REC]
                           [--img_size IMG_SIZE] [--char_dict_path CHAR_DICT_PATH]
                           [--use_space_char USE_SPACE_CHAR]

optional arguments:
  -h, --help            show this help message and exit
  --dev_id DEV_ID       tpu card id
  --input INPUT         input image directory path
  --bmodel_rec BMODEL_REC
                        recognizer bmodel path
  --img_size IMG_SIZE   You should set inference size [width, height] manually if using
                        multi-stage bmodel.
  --char_dict_path CHAR_DICT_PATH
  --use_space_char USE_SPACE_CHAR
```

文本识别测试实例如下：
```bash
# 程序会自动根据文件夹中的图片数量来选择1batch或者4batch，优先选择4batch推理。
python3 ppocr_rec_opencv.py --input ../datasets/cali_set_rec --bmodel_rec ../models/BM1684X/ch_PP-OCRv3_rec_fp32.bmodel --dev_id 0 --img_size [[640,48],[320,48]] --char_dict_path ../datasets/ppocr_keys_v1.txt
```

## 2.4 全流程推理测试：
ppocr_system_opencv.py参数说明如下：
```bash
usage: ppocr_system_opencv.py [-h] [--input INPUT] [--dev_id DEV_ID]
                              [--batch_size BATCH_SIZE] [--bmodel_det BMODEL_DET]
                              [--det_limit_side_len DET_LIMIT_SIDE_LEN]
                              [--bmodel_rec BMODEL_REC] [--img_size IMG_SIZE]
                              [--char_dict_path CHAR_DICT_PATH]
                              [--use_space_char USE_SPACE_CHAR] [--rec_thresh REC_THRESH]
                              [--use_angle_cls] [--bmodel_cls BMODEL_CLS]
                              [--label_list LABEL_LIST] [--cls_thresh CLS_THRESH]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         input image directory path
  --dev_id DEV_ID       tpu card id
  --batch_size BATCH_SIZE
                        img num for a ppocr system process launch.
  --bmodel_det BMODEL_DET
                        detector bmodel path
  --det_limit_side_len DET_LIMIT_SIDE_LEN
  --bmodel_rec BMODEL_REC
                        recognizer bmodel path
  --img_size IMG_SIZE   You should set inference size [width,height] manually if using
                        multi-stage bmodel.
  --char_dict_path CHAR_DICT_PATH
  --use_space_char USE_SPACE_CHAR
  --rec_thresh REC_THRESH
  --use_angle_cls
  --bmodel_cls BMODEL_CLS
                        classifier bmodel path
  --label_list LABEL_LIST
  --cls_thresh CLS_THRESH
```

测试实例如下：
```bash
python3 ppocr_system_opencv.py --input=../datasets/train_full_images_0 \
                           --batch_size=4 \
                           --bmodel_det=../models/BM1684X/ch_PP-OCRv3_det_fp32.bmodel \
                           --bmodel_cls=../models/BM1684X/ch_PP-OCRv3_cls_fp32.bmodel \
                           --bmodel_rec=../models/BM1684X/ch_PP-OCRv3_rec_fp32.bmodel \
                           --dev_id=0 \
                           --img_size [[640,48],[320,48]] \
                           --char_dict_path ../datasets/ppocr_keys_v1.txt \
                           --use_angle_cls
```

执行完成后，会打印预测的字段，同时会将预测的可视化结果保存在`results/inference_results`文件夹下，推理结果会保存在`results/ppocr_system_results_b4.json`下。
