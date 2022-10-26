

# Python例程
- [Python例程](#python例程)
  - [1. 目录](#1-目录)
  - [2. 环境](#2-环境)
    - [2.1 PCIE模式](#21-pcie模式)
    - [2.2 SOC模式](#22-soc模式)
  - [3.  测试](#3--测试)
    - [3.1 使用说明](#31-使用说明)
    - [3.2 精度结果](#32-精度结果)
      - [3.2.1 fp32bmodel精度](#321-fp32bmodel精度)
      - [3.2.2 int8bmodel精度](#322-int8bmodel精度)
## 1. 目录

​	目录结构说明如下，主要使用`yolov5_bmcv.py`、`yolov5_opencv.py`：

```
.
├── __init__.py
├── yolov5_bmcv.py         # 前处理采用bmcv
├── yolov5_opencv.py       # 前处理采用opencv
├── yolov5_onnx.py
├── yolov5_trace_pt.py     # 测试trace模型（非必须）
└── yolov5_utils           # 依赖

```



## 2. 环境

​	支持以下环境运行本程序。

### 2.1 PCIE模式

**硬件：**x86平台，并安装了168X PCIE加速卡，168X作为从设备。

**软件：**

1. libsophon、sophon-opencv、sophon-ffmpeg，相应成果物可以联系技术支持获取或者通过官网获取
2. sophon-sail的安装可参考工具包中说明文档 

### 2.2 SOC模式

**硬件：**SE5 SE6盒子等，168X作为主控。

**软件：**

1. 设备出厂一般会具备运行必备的环境，如果存在问题，可通过技术支持（或者官网）获取对应版本，进行固件升级。
2. sophon-sail的安装可参考工具包中说明文档



## 3.  测试

### 3.1 使用说明

python程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash

usage: yolov5_opencv.py [-h] [--model MODEL] [--dev_id DEV_ID]
                    [--conf_thresh CONF_THRESH] [--nms_thresh NMS_THRESH]
                    [--is_video IS_VIDEO] [--input_path INPUT_PATH]
                    [--output_dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         bmodel path
  --dev_id DEV_ID       device id
  --conf_thresh CONF_THRESH
                        confidence threshold
  --nms_thresh NMS_THRESH
                        nms threshold
  --is_video IS_VIDEO   input is video?
  --input_path INPUT_PATH
                        input image path
  --output_dir OUTPUT_DIR
                        output image directory
```

​	demo中支持单图、文件夹、视频测试，按照实际情况传入参数即可，默认是单图。另外，模型支持fp32bmodel、int8bmodel，可以通过传入模型路径参数进行测试：

```bash
# 测试单张图片
python3 yolov5_opencv.py
```



### 3.2 精度结果

注意精度测试需要依赖`pycocotools`工具包，安装命令：

```
pip3 install pycocotools
```



#### 3.2.1 fp32bmodel精度

​	采用coco2017val数据集，使用`tools/evaluate_coco.py`脚本进行测试，得到精度结果如下（BM1684X、fp32bmodel、PCIE模式，SOC模式数据一致）：

- opencv

evaluate_coco.py中YOLOv5基于`yolov5_opencv.py`，需要修改的地方如下：

```
class YoloTest(object):
def __init__(self, json_path, image_dir, save_json_path):
......
        model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        # model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        # model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        # model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        
        self.yolov5 = YOLOv5_opencv(model_path=model_path,
                             device_id = 0,
                             conf_thresh=0.001,
                             nms_thresh=0.65)
        print("use opencv, model:{}".format(model_path))
        
        # self.yolov5 = YOLOv5_bmcv(model_path=model_path,
        #                      device_id = 0,
        #                      conf_thresh=0.001,
        #                      nms_thresh=0.65)
        # print("use bmcv, model:{}".format(model_path))
```

运行脚本如下：

```
cd simple/YOLOv5/tools
python3 evaluate_coco.py
```

精度结果如下：

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.567
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.403
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.214
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.423
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.409
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.733
```

- bmcv

evaluate_coco.py中YOLOv5基于yolov5_bmcv.py，需要修改的地方如下：

```
class YoloTest(object):
def __init__(self, json_path, image_dir, save_json_path):
......
        model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        # model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        # model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        # model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        
        # self.yolov5 = YOLOv5_opencv(model_path=model_path,
        #                      device_id = 0,
        #                      conf_thresh=0.001,
        #                      nms_thresh=0.65)
        # print("use opencv, model:{}".format(model_path))
        
        self.yolov5 = YOLOv5_bmcv(model_path=model_path,
                             device_id = 0,
                             conf_thresh=0.001,
                             nms_thresh=0.65)
        print("use bmcv, model:{}".format(model_path))
```

运行脚本如下：

```
cd simple/YOLOv5/tools
python3 evaluate_coco.py
```

精度结果如下：

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.363
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.555
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.200
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.410
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.512
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.576
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.636
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.724
```



​	采用coco2017val数据集，使用`tools/evaluate_coco.py`脚本进行测试，得到精度结果如下（BM1684、fp32bmodel、PCIE模式，SOC模式数据一致）：

- opencv

evaluate_coco.py中YOLOv5基于`yolov5_opencv.py`，需要修改的地方如下：

```
class YoloTest(object):
def __init__(self, json_path, image_dir, save_json_path):
......
        # model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        # model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        # model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        
        self.yolov5 = YOLOv5_opencv(model_path=model_path,
                             device_id = 0,
                             conf_thresh=0.001,
                             nms_thresh=0.65)
        print("use opencv, model:{}".format(model_path))
        
        # self.yolov5 = YOLOv5_bmcv(model_path=model_path,
        #                      device_id = 0,
        #                      conf_thresh=0.001,
        #                      nms_thresh=0.65)
        # print("use bmcv, model:{}".format(model_path))
```

运行脚本如下：

```
cd simple/YOLOv5/tools
python3 evaluate_coco.py
```

精度结果如下：

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.567
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.403
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.214
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.423
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.409
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.733
```

- bmcv

evaluate_coco.py中YOLOv5基于yolov5_bmcv.py，需要修改的地方如下：

```
class YoloTest(object):
def __init__(self, json_path, image_dir, save_json_path):
......
        # model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        # model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        # model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        
        # self.yolov5 = YOLOv5_opencv(model_path=model_path,
        #                      device_id = 0,
        #                      conf_thresh=0.001,
        #                      nms_thresh=0.65)
        # print("use opencv, model:{}".format(model_path))
        
        self.yolov5 = YOLOv5_bmcv(model_path=model_path,
                             device_id = 0,
                             conf_thresh=0.001,
                             nms_thresh=0.65)
        print("use bmcv, model:{}".format(model_path))
```

运行脚本如下：

```
cd simple/YOLOv5/tools
python3 evaluate_coco.py
```

精度结果如下：

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.363
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.555
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.200
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.410
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.481
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.512
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.401
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.636
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.725
```



#### 3.2.2 int8bmodel精度

​	采用coco2017val数据集，使用`tools/evaluate_coco.py`脚本进行测试，得到精度结果如下（BM1684X、int8bmodel、PCIE模式，SOC模数数据基本一致）：

- opencv

evaluate_coco.py中YOLOv5基于`yolov5_opencv.py`，需要修改的地方如下：

```
class YoloTest(object):
def __init__(self, json_path, image_dir, save_json_path):
......
        # model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        # model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        # model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        
        self.yolov5 = YOLOv5_opencv(model_path=model_path,
                             device_id = 0,
                             conf_thresh=0.001,
                             nms_thresh=0.65)
        print("use opencv, model:{}".format(model_path))
        
        # self.yolov5 = YOLOv5_bmcv(model_path=model_path,
        #                      device_id = 0,
        #                      conf_thresh=0.001,
        #                      nms_thresh=0.65)
        # print("use bmcv, model:{}".format(model_path))
```

运行脚本如下：

```
cd simple/YOLOv5/tools
python3 evaluate_coco.py
```

精度结果如下：

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.550
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.205
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.572
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.389
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.636
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.724
```

- bmcv

evaluate_coco.py中YOLOv5基于yolov5_bmcv.py，需要修改的地方如下：

```
class YoloTest(object):
def __init__(self, json_path, image_dir, save_json_path):
......
        # model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        # model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        # model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        
        # self.yolov5 = YOLOv5_opencv(model_path=model_path,
        #                      device_id = 0,
        #                      conf_thresh=0.001,
        #                      nms_thresh=0.65)
        # print("use opencv, model:{}".format(model_path))
        
        self.yolov5 = YOLOv5_bmcv(model_path=model_path,
                             device_id = 0,
                             conf_thresh=0.001,
                             nms_thresh=0.65)
        print("use bmcv, model:{}".format(model_path))
```

运行脚本如下：

```
cd simple/YOLOv5/tools
python3 evaluate_coco.py
```

精度结果如下：

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.344
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.536
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.191
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.561
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.380
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.624
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.708
```



​	采用coco2017val数据集，使用`tools/evaluate_coco.py`脚本进行测试，得到精度结果如下（BM1684、int8bmodel、PCIE模式，SOC模数数据基本一致）：

- opencv

evaluate_coco.py中YOLOv5基于`yolov5_opencv.py`，需要修改的地方如下：

```
class YoloTest(object):
def __init__(self, json_path, image_dir, save_json_path):
......
        # model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        # model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        # model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        
        self.yolov5 = YOLOv5_opencv(model_path=model_path,
                             device_id = 0,
                             conf_thresh=0.001,
                             nms_thresh=0.65)
        print("use opencv, model:{}".format(model_path))
        
        # self.yolov5 = YOLOv5_bmcv(model_path=model_path,
        #                      device_id = 0,
        #                      conf_thresh=0.001,
        #                      nms_thresh=0.65)
        # print("use bmcv, model:{}".format(model_path))
```

运行脚本如下：

```
cd simple/YOLOv5/tools
python3 evaluate_coco.py
```

精度结果如下：

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.538
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.202
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.395
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.446
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.561
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.622
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.704
```

- bmcv

evaluate_coco.py中YOLOv5基于yolov5_bmcv.py，需要修改的地方如下：

```
class YoloTest(object):
def __init__(self, json_path, image_dir, save_json_path):
......
        # model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        # model_path = "../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        # model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel"
        model_path = "../data/models/BM1684/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel"
        
        # self.yolov5 = YOLOv5_opencv(model_path=model_path,
        #                      device_id = 0,
        #                      conf_thresh=0.001,
        #                      nms_thresh=0.65)
        # print("use opencv, model:{}".format(model_path))
        
        self.yolov5 = YOLOv5_bmcv(model_path=model_path,
                             device_id = 0,
                             conf_thresh=0.001,
                             nms_thresh=0.65)
        print("use bmcv, model:{}".format(model_path))
```

运行脚本如下：

```
cd simple/YOLOv5/tools
python3 evaluate_coco.py
```

精度结果如下：

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.331
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.522
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.355
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.379
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.436
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.551
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.372
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.610
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.696
```

