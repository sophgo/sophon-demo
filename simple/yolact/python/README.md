# YOLACT Python例程

python目录下提供了一系列Python例程，具体情况如下：

| #    | 样例文件           | 说明                                                    |
| ---- | ------------------ | ------------------------------------------------------- |
| 1    | yolact_bmcv.py     | 使用SAIL解码、BMCV前处理、SAIL推理、OpenCV后处理        |
| 2    | yolact_opencv.py   | 使用OpenCV解码、OpenCV前处理、SAIL推理、OpenCV后处理    |
| 3    | yolact_trace_pt.py | 使用OpenCV解码、OpenCV前处理、PyTorch推理、OpenCV后处理 |
| 4    | yolact_onnx.py     | 使用OpenCV解码、OpenCV前处理、ONNX推理、OpenCV后处理    |

python目录结构如下：

```bash
python
├── configs
│   ├── yolact_base.cfg				# yolact_base模型配置文件
│   ├── yolact_darknet53.cfg		# yolact_darknet53模型配置文件
│   ├── yolact_im700.cfg			# yolact_im700模型配置文件
│   └── yolact_resnet50.cfg			# yolact_resnet50模型配置文件
├── __init__.py
├── yolact_bmcv.py					
├── yolact_onnx.py					
├── yolact_opencv.py
├── yolact_trace_pt.py
└── yolact_utils					
    ├── __init__.py
    ├── onnx_inference.py
    ├── postprocess_numpy.py		# numpy后处理
    ├── preprocess_bmcv.py			# bmcv前处理
    ├── preprocess_numpy.py			# numpy前处理
    ├── sophon_inference.py
    └── utils.py
```

## 1. x86 PCIe平台

### 1.1 环境准备

如果您在x86平台安装了PCIe加速卡，并使用它测试本例程。本例程需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，版本要求如下表所示。libsophon的安装可参考《LIBSOPHON使用手册》，sophon-opencv和sophon-ffmpeg的安装可参考《multimedia开发参考手册》，sophon-sail的安装可参考《sophon-sail使用手册》。注：需要获取《LIBSOPHON使用手册》、《multimedia开发参考手册》和《sophon-sail使用手册》，请联系技术支持。

| 依赖          | 版本    |
| ------------- | ------- |
| libsophon     | >=0.3.0 |
| sophon-opencv | >=0.2.4 |
| sophon-ffmpeg | >=0.2.4 |
| sophon-sail   | >=3.0.1 |

### 1.2 测试命令

python例程不需要编译，可以直接运行。yolact_opencv.py和yolact_bmcv.py的命令参数相同，以yolact_bmcv.py为例，参数说明如下：

```bash
usage: yolact_bmcv.py [-h] [--cfgfile CFGFILE] [--model MODEL] [--dev_id DEV_ID] [--conf_thresh CONF_THRESH] [--nms_thresh NMS_THRESH] [--keep KEEP] [--is_video IS_VIDEO] [--input_path INPUT_PATH] [--output_dir OUTPUT_DIR] [--video_detect_frame_num VIDEO_DETECT_FRAME_NUM]

optional arguments:
  -h, --help            show this help message and exit
  --cfgfile CFGFILE     model config file					# 模型.cfg配置文件
  --model MODEL         bmodel path							# 模型bmodel文件
  --dev_id DEV_ID       device id							# 设备id
  --conf_thresh CONF_THRESH
                        confidence threshold				# conf阈值，默认为0.5
  --nms_thresh NMS_THRESH
                        nms threshold						# nms阈值，默认为0.5
  --keep KEEP           keep top-k							# top-k数量，默认为100
  --is_video IS_VIDEO   input is video?						# 输入数据是否为视频，0：输入为图像，1：输入为视频。默认为0
  --input_path INPUT_PATH
                        input path							# 输入图像或视频路径 	
  --output_dir OUTPUT_DIR
                        output image directory				# 结果图像保存的文件夹路径，默认保存在results/results_{script}文件夹下。{script}为bmcv或opencv
  --video_detect_frame_num VIDEO_DETECT_FRAME_NUM
                        detect frame number of video		# 当检测视频时，检测和保存结果的视频帧数，默认为10
```

请根据目标平台、模型精度、选择相应的bmodel，测试示例如下：

```bash
cd ${YOLACT}/python
# yolact_cv.py使用方法与yolact_bmcv.py一致，如果使用yolact_cv.py，结果将保存在results/results_cv目录下；如果使用yolact_bmcv.py，结果将保存在results/results_bmcv目录下。

# 以yoloact base 1684X为例
# image
python3 yolact_bmcv.py --cfgfile configs/yolact_base.cfg --model ../data/models/BM1684X/yolact_base_54_800000_fp32_b1.bmodel --input_path ../data/images/

# video
python3 yolact_bmcv.py --cfgfile configs/yolact_base.cfg --model ../data/models/BM1684X/yolact_base_54_800000_fp32_b1.bmodel --is_video 1 --input_path ../data/videos/road.mp4 --video_detect_frame_num 10

# PCIe模式下如果需要使用yolact_trace_pt.py、yolact_onnx.py测试，请自行安装pytorch、onnx环境
# 如果使用yolact_trace_pt.py测试，<model>为JIT模型路径，结果将保存在results/results_trace_pt目录下
# 如果使用yolact_onnx.py,<model>为ONNX模型路径，结果将保存在results/results_onnx目录下
```

## 2. SoC平台

### 2.1 环境准备

如果您使用SoC平台测试本例程，您需要安装sophon-sail(>=3.0.1)，具体可参考《sophon-sail用手册》。注：需要获取《sophon-sail用手册》，请联系技术支持。

### 2.2 测试命令

SoC平台的测试方法与x86 PCIe平台相同，请参考[1.2 测试命令](#12-测试命令)。

将python文件夹和data文件夹拷贝到SE5中同一目录下，测试示例如下：

```bash
cd ${YOLACT}/python
# yolact_cv.py使用方法与yolact_bmcv.py一致，如果使用yolact_cv.py，结果将保存在results/results_cv目录下；如果使用yolact_bmcv.py，结果将保存在results/results_bmcv目录下。

# 以yoloact base 1684X为例
# image
python3 yolact_bmcv.py --cfgfile configs/yolact_base.cfg --model ../data/models/BM1684X/yolact_base_54_800000_fp32_b1.bmodel --input_path ../data/images/

# video
python3 yolact_bmcv.py --cfgfile configs/yolact_base.cfg --model ../data/models/BM1684X/yolact_base_54_800000_fp32_b1.bmodel --is_video 1 --input_path ../data/videos/road.mp4 --video_detect_frame_num 10

# SoC模式下不具备pytorch、onnx环境，不建议使用yolact_trace_pt.py、yolact_onnx.py测试
```

## 3. 其他

> **使用SAIL模块的注意事项：**对于INT8 BModel来说，当输入输出为int8时，含有scale，需要在处理时将输入输出乘以相应的scale。使用SAIL接口推理时，当sail.Engine.process()接口输入为numpy时，SAIL内部会自动乘以scale，用户无需操作；而输入为Tensor时，需要手动在数据送入推理接口前乘以scale。
>
> 这是因为Tensor作为输入的话，一般图像来源就是bm_image，这样就可以直接调用vpp进行scale等操作，所以推理之前由用户乘以scale更高效；而在python接口中，当numpy作为输入的话，推理之前没办法调用vpp，sail内部使用SSE指令进行了加速。

