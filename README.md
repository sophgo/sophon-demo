[简体中文](./README.md) | [English](./README_EN.md)

## sophon-demo介绍


## 简介
Sophon Demo基于SophonSDK接口进行开发，提供一系列主流算法的移植例程。包括基于TPU-NNTC和TPU-MLIR的模型编译与量化，基于BMRuntime的推理引擎移植，以及基于BMCV/OpenCV的前后处理算法移植。

SophonSDK是算能科技基于其自主研发的深度学习处理器所定制的深度学习SDK，涵盖了神经网络推理阶段所需的模型优化、高效运行时支持等能力，为深度学习应用开发和部署提供易用、高效的全栈式解决方案。目前可兼容BM1684/BM1684X/BM1688。

## 目录结构与说明
| 目录                                        | 算法类别         | 编程语言    | BModel | 支持多batch | 预处理库 |
|---                                          |---              |---          | ---     |---          |---      |
| [LPRNet](./sample/LPRNet/README.md)         | 车牌识别          | C++/Python | FP32/FP16/INT8 | YES | BMCV/OpenCV |
| [ResNet](./sample/ResNet/README.md)         | 图像分类          | C++/Python | FP32/FP16/INT8 | YES | BMCV/OpenCV |
| [RetinaFace](./sample/RetinaFace/README.md) | 人脸检测          | C++/Python | FP32           | YES | BMCV/OpenCV |
| [segformer](./sample/segformer/README.md)   | 语义分割           | C++/Python | FP32/FP16      | YES | BMCV/OpenCV |
| [yolact](./sample/yolact/README.md)         | 实例分割          | Python     | FP32           | YES | BMCV/OpenCV |
| [PP-OCR](./sample/PP-OCR/README.md)         | OCR              | C++/Python | FP32/FP16      | YES | BMCV/OpenCV |
| [OpenPose](./sample/OpenPose/README.md)     | 人体关键点检测     | C++/Python | FP32/FP16/INT8 | YES | BMCV/OpenCV |
| [C3D](./sample/C3D/README.md)               | 视频动作识别       | C++/Python | FP32/FP16/INT8 | YES | BMCV/OpenCV |
| [DeepSORT](./sample/DeepSORT/README.md)     | 多目标跟踪         | C++/Python | FP32/FP16/INT8 | YES | BMCV/OpenCV |
| [ByteTrack](./sample/ByteTrack/README.md)   | 多目标跟踪         | C++/Python | FP32/FP16/INT8 | YES | BMCV/OpenCV |
| [CenterNet](./sample/CenterNet/README.md)   | 目标检测、姿态识别 | C++/Python | FP32/FP16/INT8 | YES | BMCV |
| [YOLOv5](./sample/YOLOv5/README.md)         | 目标检测          | C++/Python | FP32/FP16/INT8 | YES | BMCV/OpenCV |
| [YOLOv34](./sample/YOLOv34/README.md)       | 目标检测          | C++/Python | FP32/INT8      | NO  | BMCV/OpenCV |
| [YOLOX](./sample/YOLOX/README.md)           | 目标检测          | C++/Python | FP32/INT8      | YES | BMCV/OpenCV |
| [SSD](./sample/SSD/README.md)               | 目标检测          | C++/Python | FP32/INT8      | YES | BMCV/OpenCV |
| [YOLOv7](./sample/YOLOv7/README.md)         | 目标检测           | C++/Python | FP32/FP16/INT8 | YES | BMCV/OpenCV|
| [YOLOv8](./sample/YOLOv8/README.md)         | 目标检测           | C++/Python | FP32/FP16/INT8 | YES | BMCV/OpenCV |
| [YOLOv5_opt](./sample/YOLOv5_opt/README.md) | 目标检测           | C++/Python | FP32/FP16/INT8 | YES | BMCV/OpenCV|
| [BERT](./sample/BERT/README.md)             | 语言模型           | C++/Python | FP32/FP16      | YES | BMCV|
| [ChatGLM2](./sample/chatglm2/README.md)     | 语言模型           | C++/Python | FP16/INT8/INT4 | YES | BMCV|
| [Llama2](./sample/Llama2/README.md)         | 语言模型           | C++        | FP16/INT8/INT4 | YES | -|

## 版本说明
| 版本    | 说明 | 
|---     |---   |
| 0.1.9	 | 修复文档等问题，新增segformer、YOLOv7、Llama2例程，重构YOLOv34，YOLOv5、ResNet、PP-OCR、DeepSORT适配BM1688，OpenPose后处理加速，chatglm2添加编译方法和int8/int4量化。|
| 0.1.8  | 完善修复文档、代码问题，新增BERT、ppYOLOv3、ChatGLM2，重构YOLOX，PP-OCR添加beam search，OpenPose添加tpu-kernel后处理加速，更新SFTP下载方式。|
| 0.1.7	 | 修复文档等问题，一些例程支持BM1684 mlir，重构PP-OCR、CenterNet例程，YOLOv5添加sail支持 |
| 0.1.6	 | 修复文档等问题，新增ByteTrack、YOLOv5_opt例程 |
| 0.1.5	 | 修复文档等问题，新增DeepSORT例程，重构ResNet、LPRNet例程 |
| 0.1.4	 | 修复文档等问题，新增C3D、YOLOv8例程 |
| 0.1.3	 | 新增OpenPose例程，重构YOLOv5例程（包括适配arm PCIe、支持TPU-MLIR编译BM1684X模型、使用ffmpeg组件替换opencv解码等） |
| 0.1.2	 | 修复文档等问题，重构SSD相关例程，LPRNet/cpp/lprnet_bmcv使用ffmpeg组件替换opencv解码 |
| 0.1.1	 | 修复文档等问题，使用BMNN相关类重构LPRNet/cpp/lprnet_bmcv |
| 0.1.0	 | 提供LPRNet等10个例程，适配BM1684X(x86 PCIe、SoC)，BM1684(x86 PCIe、SoC) |

## 环境依赖
Sophon Demo主要依赖tpu-mlir、tpu-nntc、libsophon、sophon-ffmpeg、sophon-opencv、sophon-sail，其版本要求如下：
|sophon-demo|tpu-mlir |tpu-nntc |libsophon|sophon-ffmpeg|sophon-opencv|sophon-sail| 发布日期   |
|--------|------------| --------|---------|---------    |----------   | ------    | --------  |
| 0.1.9 | >=1.2.2     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.7.0   | >=23.10.01|
| 0.1.8 | >=1.2.2     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.6.0   | >=23.07.01|
| 0.1.7 | >=1.2.2     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.6.0   | >=23.07.01|
| 0.1.6 | >=0.9.9     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.4.0   | >=23.05.01|
| 0.1.5 | >=0.9.9     | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.4.0   | >=23.03.01|
| 0.1.4 | >=0.7.1     | >=3.1.5 | >=0.4.4 | >=0.5.1     | >=0.5.1     | >=3.3.0   | >=22.12.01|
| 0.1.3 | >=0.7.1     | >=3.1.5 | >=0.4.4 | >=0.5.1     | >=0.5.1     | >=3.3.0   |    -      |
| 0.1.2 | Not support | >=3.1.4 | >=0.4.3 | >=0.5.0     | >=0.5.0     | >=3.2.0   |    -      |
| 0.1.1 | Not support | >=3.1.3 | >=0.4.2 | >=0.4.0     | >=0.4.0     | >=3.1.0   |    -      |
| 0.1.0 | Not support | >=3.1.3 | >=0.3.0 | >=0.2.4     | >=0.2.4     | >=3.1.0   |    -      |
> **注意**：不同例程对版本的要求可能存在差异，具体以例程的README为准，可能需要安装其他第三方库。

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