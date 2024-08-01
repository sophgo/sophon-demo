[简体中文](./README.md) | [English](./README_EN.md)

## SOPHON-DEMO介绍


## 简介
SOPHON-DEMO基于SOPHONSDK接口进行开发，提供一系列主流算法的移植例程。包括基于TPU-NNTC和TPU-MLIR的模型编译与量化，基于BMRuntime的推理引擎移植，以及基于BMCV/OpenCV的前后处理算法移植。

SOPHONSDK是算能科技基于其自主研发的深度学习处理器所定制的深度学习SDK，涵盖了神经网络推理阶段所需的模型优化、高效运行时支持等能力，为深度学习应用开发和部署提供易用、高效的全栈式解决方案。目前可兼容BM1684/BM1684X/BM1688(CV186X)。

## 目录结构与说明
SOPHON-DEMO提供的例子从易到难分为`tutorial`、`sample`、`application`三个模块，`tutorial`模块存放一些基础接口的使用示例，`sample`模块存放一些经典算法在SOPHONSDK上的串行示例，`application`模块存放一些典型场景的典型应用。

| tutorial                                                                 | 说明                                                                      |
| ----------------------------------------------------                     | ------------------------------------------------------------              |
| [resize](./tutorial/resize/README.md)                                    | resize接口。针对图像做缩放操作                                               |
| [crop](./tutorial/crop/README.md)                                        | crop接口，从输入图片中抠出需要用的图片区域                                    |
| [crop_and_resize_padding](./tutorial/crop_and_resize_padding/README.md)  | 将图片指定位置指定大小部分图片抠出，缩放后填充到大图中，空余部分填充指定像素数值  |
| [ocv_jpgbasic](./tutorial/ocv_jpubasic/README.md)                        | 使用sophon-opencv硬件加速实现图片编解码                                      |
| [ocv_vidbasic](./tutorial/ocv_vidbasic/README.md)                        | 使用sophon-opencv硬件加速实现视频解码，并将视频记录为png或jpg格式              |
| [blend](./tutorial/blend/README.md)                                      | 融合拼接两张图                                                              |
| [stitch](./tutorial/stitch/README.md)                                    | 拼接两张图片                                                                |
| [avframe_ocv](./tutorial/avframe_ocv/README.md)                          | avframe到cv::mat的转换例程                                                  |
| [ocv_avframe](./tutorial/ocv_avframe/README.md)                          | bgr mat到yuv420p avframe的转换例程                                          |
| [bm1688_2core2task_yolov5](./tutorial/bm1688_2core2task_yolov5/README.md)| 使用bm1688的双核双任务推理部署的yolov5                                       |
| [mmap](./tutorial/mmap/README.md)                                        | mmap接口，映射TPU内存到CPU                                                  |
| [video_encode](./tutorial/video_encode/README.md)                        | 视频编码和推流                                                              |

| sample                                                          | 算法类别          | 编程语言    | BModel         |
|---                                                            |---               |---          | ---           |
| [LPRNet](./sample/LPRNet/README.md)                           | 车牌识别          | C++/Python | FP32/FP16/INT8 |
| [ResNet](./sample/ResNet/README.md)                           | 图像分类          | C++/Python | FP32/FP16/INT8 |
| [RetinaFace](./sample/RetinaFace/README.md)                   | 人脸检测          | C++/Python | FP32/FP16/INT8 |
| [SCRFD](./sample/SCRFD/README.md)                             | 人脸检测          | C++/Python | FP32/FP16/INT8 |
| [segformer](./sample/segformer/README.md)                     | 语义分割          | C++/Python | FP32/FP16      |
| [SAM](./sample/SAM/README.md)                                 | 语义分割          | Python     | FP32/FP16      |
| [yolact](./sample/yolact/README.md)                           | 实例分割          | C++/Python | FP32/FP16/INT8 |
| [YOLOv8_seg](./sample/YOLOv8_seg/README.md)                   | 实例分割          | C++/Python | FP32/FP16/INT8 |
| [YOLOv9_seg](./sample/YOLOv9_seg/README.md)                   | 实例分割          | C++/Python | FP32/FP16/INT8 |
| [PP-OCR](./sample/PP-OCR/README.md)                           | OCR              | C++/Python | FP32/FP16      | 
| [OpenPose](./sample/OpenPose/README.md)                       | 人体关键点检测    | C++/Python | FP32/FP16/INT8 |
| [C3D](./sample/C3D/README.md)                                 | 视频动作识别      | C++/Python | FP32/FP16/INT8 |
| [DeepSORT](./sample/DeepSORT/README.md)                       | 多目标跟踪        | C++/Python | FP32/FP16/INT8 |
| [ByteTrack](./sample/ByteTrack/README.md)                     | 多目标跟踪        | C++/Python | FP32/FP16/INT8 |
| [CenterNet](./sample/CenterNet/README.md)                     | 目标检测、姿态识别 | C++/Python | FP32/FP16/INT8 |
| [YOLOv5](./sample/YOLOv5/README.md)                           | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [YOLOv34](./sample/YOLOv34/README.md)                         | 目标检测          | C++/Python | FP32/INT8      |
| [YOLOX](./sample/YOLOX/README.md)                             | 目标检测          | C++/Python | FP32/INT8      |
| [SSD](./sample/SSD/README.md)                                 | 目标检测          | C++/Python | FP32/INT8      |
| [YOLOv7](./sample/YOLOv7/README.md)                           | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [YOLOv8_det](./sample/YOLOv8_det/README.md)                   | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [YOLOv5_opt](./sample/YOLOv5_opt/README.md)                   | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [YOLOv5_fuse](./sample/YOLOv5_fuse/README.md)                 | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [YOLOv9_det](./sample/YOLOv9_det/README.md)                   | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [YOLOv10](./sample/YOLOv10/README.md)                         | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [ppYOLOv3](./sample/ppYOLOv3/README.md)                       | 目标检测          | C++/Python | FP32/FP16/INT8 |
| [ppYoloe](./sample/ppYoloe/README.md)                         | 目标检测          | C++/Python | FP32/FP16      |
| [WeNet](./sample/WeNet/README.md)                             | 语音识别          | C++/Python | FP32/FP16      | 
| [Whisper](./sample/Whisper/README.md)                         | 语音识别          | Python     | FP16           | 
| [Seamless](./sample/Seamless/README.md)                       | 语音识别          | Python     | FP32/FP16      | 
| [BERT](./sample/BERT/README.md)                               | 语言模型          | C++/Python | FP32/FP16      | 
| [ChatGLM2](./sample/ChatGLM2/README.md)                       | 语言模型          | C++/Python | FP16/INT8/INT4 | 
| [Llama2](./sample/Llama2/README.md)                           | 语言模型          | C++/Python | FP16/INT8/INT4 |
| [ChatGLM3](./sample/ChatGLM3/README.md)                       | 语言模型          | Python     | FP16/INT8/INT4 | 
| [Qwen](./sample/Qwen/README.md)                               | 语言模型          | Python     | FP16/INT8/INT4 | 
| [Qwen1_5](./sample/Qwen1_5/README.md)                         | 语言模型          | Python     | FP16/INT8/INT4 | 
| [MiniCPM](./sample/MiniCPM/README.md)                         | 语言模型          | C++        | INT8/INT4      | 
| [Baichuan2](./sample/Baichuan2/README.md)                     | 语言模型          | Python     | INT8/INT4      | 
| [ChatGLM4](./sample/ChatGLM4/README.md)                       | 语言模型          | Python     | INT8/INT4      | 
| [StableDiffusionV1.5](./sample/StableDiffusionV1_5/README.md) | 图像生成          | Python     | FP32/FP16      |
| [StableDiffusionXL](./sample/StableDiffusionXL/README.md)     | 图像生成          | Python     | FP32/FP16      |
| [GroundingDINO](./sample/GroundingDINO/README.md)             | 多模态目标检测     | Python     | FP16           |
| [Real-ESRGAN](./sample/Real-ESRGAN/README.md)                 | 超分辨            | C++/Python | FP32/FP16/INT8 |
| [P2PNet](./sample/P2PNet/README.md)                           | 人群计数          | C++/Python | FP32/FP16/INT8 |
| [CLIP](./sample/CLIP/README.md)                               | 图文生成          | Python     | FP16           |
| [SuperGlue](./sample/SuperGlue/README.md)                     | 特征匹配          | C++        | FP32/FP16      |

| application                                                    | 应用场景                  | 编程语言    | 
|---                                                             |---                       |---          | 
| [VLPR](./application/VLPR/README.md)                           | 多路车牌检测+识别          | C++/Python  | 
| [YOLOv5_multi](./application/YOLOv5_multi/README.md)           | 多路目标检测               | C++         | 
| [YOLOv5_multi_QT](./application/YOLOv5_multi_QT/README.md)     | 多路目标检测+QT_HDMI显示   | C++         | 

## 版本说明
| 版本    | 说明 | 
|---     |---   |
| 0.2.3  | 完善和修复文档、代码问题，sample模块新增例程StableDiffusionXL、ChatGLM4、Seamless、YOLOv10，tutorial模块新增mmap、video_encode例程。 |
| 0.2.2  | 完善和修复文档、代码问题，部分例程补充CV186X支持，sample模块新增例程Whisper、Real-ESRGAN、SCRFD、P2PNet、MiniCPM、CLIP、SuperGlue、YOLOv5_fuse、YOLOv8_seg、YOLOv9_seg、Baichuan2等例程，tutorial模块新增avframe_ocv、ocv_avframe、bm1688_2core2task_yolov5例程。 |
| 0.2.1  | 完善和修复文档、代码问题，部分例程补充CV186X支持，YOLOv5适配SG2042，sample模块新增例程GroundingDINO、Qwen1_5，StableDiffusionV1_5新支持多种分辨率，Qwen、Llama2、ChatGLM3添加web和多会话模式。tutorial模块新增blend和stitch例程 |
| 0.2.0  | 完善和修复文档、代码问题，新增application和tutorial模块，新增例程ChatGLM3和Qwen，SAM添加web ui，BERT、ByteTrack、C3D适配BM1688，原YOLOv8更名为YOLOv8_det并且添加cpp后处理加速方法，优化常用例程的auto_test，更新TPU-MLIR安装方式为pip |
| 0.1.10 | 修复文档、代码问题，新增ppYoloe、YOLOv8_seg、StableDiffusionV1.5、SAM，重构yolact，CenterNet、YOLOX、YOLOv8适配BM1688，YOLOv5、ResNet、PP-OCR、DeepSORT补充BM1688性能数据，WeNet提供C++交叉编译方法 |
| 0.1.9	 | 修复文档、代码问题，新增segformer、YOLOv7、Llama2例程，重构YOLOv34，YOLOv5、ResNet、PP-OCR、DeepSORT、LPRNet、RetinaFace、YOLOv34、WeNet适配BM1688，OpenPose后处理加速，chatglm2添加编译方法和int8/int4量化。|
| 0.1.8  | 完善修复文档、代码问题，新增BERT、ppYOLOv3、ChatGLM2，重构YOLOX，PP-OCR添加beam search，OpenPose添加tpu-kernel后处理加速，更新SFTP下载方式。|
| 0.1.7	 | 修复文档等问题，一些例程支持BM1684 mlir，重构PP-OCR、CenterNet例程，YOLOv5添加sail支持 |
| 0.1.6	 | 修复文档等问题，新增ByteTrack、YOLOv5_opt、WeNet例程 |
| 0.1.5	 | 修复文档等问题，新增DeepSORT例程，重构ResNet、LPRNet例程 |
| 0.1.4	 | 修复文档等问题，新增C3D、YOLOv8例程 |
| 0.1.3	 | 新增OpenPose例程，重构YOLOv5例程（包括适配arm PCIe、支持TPU-MLIR编译BM1684X模型、使用ffmpeg组件替换opencv解码等） |
| 0.1.2	 | 修复文档等问题，重构SSD相关例程，LPRNet/cpp/lprnet_bmcv使用ffmpeg组件替换opencv解码 |
| 0.1.1	 | 修复文档等问题，使用BMNN相关类重构LPRNet/cpp/lprnet_bmcv |
| 0.1.0	 | 提供LPRNet等10个例程，适配BM1684X(x86 PCIe、SoC)，BM1684(x86 PCIe、SoC) |

## 环境依赖
SOPHON-DEMO主要依赖TPU-MLIR、TPU-NNTC、LIBSOPHON、SOPHON-FFMPEG、SOPHON-OPENCV、SOPHON-SAIL，对于BM1684/BM1684X SOPHONSDK，其版本要求如下：
|SOPHON-DEMO|TPU-MLIR  |TPU-NNTC |LIBSOPHON|SOPHON-FFMPEG|SOPHON-OPENCV|SOPHON-SAIL| SOPHONSDK   |
|-------- |------------| --------|---------|---------    |----------   | ------    | --------  |
| 0.2.3  | >=1.8       | >=3.1.7 | >=0.5.0 | >=0.7.3     | >=0.7.3     | >=3.7.0   | >=v24.04.01|
| 0.2.2  | >=1.8       | >=3.1.7 | >=0.5.0 | >=0.7.3     | >=0.7.3     | >=3.7.0   | >=v23.10.01|
| 0.2.1  | >=1.7       | >=3.1.7 | >=0.5.0 | >=0.7.3     | >=0.7.3     | >=3.7.0   | >=v23.10.01|
| 0.2.0  | >=1.6       | >=3.1.7 | >=0.5.0 | >=0.7.3     | >=0.7.3     | >=3.7.0   | >=v23.10.01|
| 0.1.10 | >=1.2.2     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.7.0   | >=v23.07.01|
| 0.1.9  | >=1.2.2     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.7.0   | >=v23.07.01|
| 0.1.8  | >=1.2.2     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.6.0   | >=v23.07.01|
| 0.1.7  | >=1.2.2     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.6.0   | >=v23.07.01|
| 0.1.6  | >=0.9.9     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.4.0   | >=v23.05.01|
| 0.1.5  | >=0.9.9     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.4.0   | >=v23.03.01|
| 0.1.4  | >=0.7.1     | >=3.1.5 | >=0.4.4 | >=0.5.1     | >=0.5.1     | >=3.3.0   | >=v22.12.01|
| 0.1.3  | >=0.7.1     | >=3.1.5 | >=0.4.4 | >=0.5.1     | >=0.5.1     | >=3.3.0   |    -      |
| 0.1.2  | Not support | >=3.1.4 | >=0.4.3 | >=0.5.0     | >=0.5.0     | >=3.2.0   |    -      |
| 0.1.1  | Not support | >=3.1.3 | >=0.4.2 | >=0.4.0     | >=0.4.0     | >=3.1.0   |    -      |
| 0.1.0  | Not support | >=3.1.3 | >=0.3.0 | >=0.2.4     | >=0.2.4     | >=3.1.0   |    -      |

对于BM1688/CV186AH SOPHONSDK，其版本要求如下：
|SOPHON-DEMO|TPU-MLIR  |LIBSOPHON|SOPHON-FFMPEG|SOPHON-OPENCV|SOPHON-SAIL| SOPHONSDK   |
|-------- |------------|---------|---------    |----------   | ------    | --------  |
| 0.2.3  | >=1.8       | >=0.4.9 | >=1.6.0     | >=1.6.0     | >=3.8.0   | >=v1.7.0  |
| 0.2.2  | >=1.8       | >=0.4.9 | >=1.6.0     | >=1.6.0     | >=3.8.0   | >=v1.6.0  |
| 0.2.1  | >=1.7       | >=0.4.9 | >=1.5.0     | >=1.5.0     | >=3.8.0   | >=v1.5.0  |
| 0.2.0  | >=1.6       | >=0.4.9 | >=1.5.0     | >=1.5.0     | >=3.7.0   | >=v1.5.0  |

> **注意**：
> 1. 不同例程对版本的要求可能存在差异，具体以例程的README为准，可能需要安装其他第三方库。
> 2. BM1688/CV186X与BM1684X/BM1684对应的sdk不是同一套，在官网上已作区分，请注意。

## 技术资料

请通过算能官网[技术资料](https://developer.sophgo.com/site/index.html)获取相关文档、资料及视频教程。

## 社区

算能社区鼓励开发者多交流，共学习。开发者可以通过以下渠道进行交流和学习。

算能社区网站：https://www.sophgo.com/

算能开发者论坛：https://developer.sophgo.com/forum/index.html


## 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](./CONTRIBUTING_CN.md)。

## 许可证
[Apache License 2.0](./LICENSE)