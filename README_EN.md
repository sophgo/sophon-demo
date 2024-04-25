[简体中文](./README.md) | [English](./README_EN.md)

## Introduction to SOPHON-DEMO

## Introduction
SOPHON-DEMO is developed based on the SOPHONSDK interface and provides a series of samples for mainstream algorithms. It includes model compilation and quantization based on TPU-NNTC and TPU-MLIR, inference engine porting based on BMRuntime, and pre and post-processing algorithm migration based on BMCV/OpenCV.

SOPHONSDK is a custom deep learning SDK of SOPHGO based on its self-developed AI processor, covering model optimization, efficient runtime support, and other capabilities required for the inference phase of neural networks, providing an easy-to-use and efficient full-stack solution for deep learning application development and deployment. It is currently compatible with BM1684/BM1684X/BM1688(CV186X).

## Directory Structure and Description.
The examples provided by SOPHON-DEMO are divided into three modules: `tutorial`, `sample`, and `application`, the `tutorial` module stores some examples of basic interfaces, the `sample` module stores some serial examples of classic algorithms on SOPHONSDK, and the `application` module stores some typical applications in typical scenarios.

| tutorial                                                                  | code    |
|---                                                                        |---         |
| [resize](./tutorial/resize/README.md)                                     | C++/Python |
| [crop](./tutorial/crop/README.md)                                         | C++/Python |
| [crop_and_resize_padding](./tutorial/crop_and_resize_padding/README.md)   | C++/Python |
| [ocv_jpubasic](./tutorial/ocv_jpubasic/README.md) | C++/Python |
| [ocv_vidbasic](./tutorial/vid_jpubasic/README.md) | C++/Python |

| contents                                                      | category                           | code       |  BModel       |
|---                                                            |---                                 |---          | ---           |
| [LPRNet](./sample/LPRNet/README.md)                           | License Plate Recognition          | C++/Python | FP32/FP16/INT8 |
| [ResNet](./sample/ResNet/README.md)                           | Image classification               | C++/Python | FP32/FP16/INT8 |
| [RetinaFace](./sample/RetinaFace/README.md)                   | Face detection                     | C++/Python | FP32           |
| [segformer](./sample/segformer/README.md)                     | Semantic segmentation              | C++/Python | FP32/FP16      |
| [SAM](./sample/SAM/README.md)                                 | Semantic segmentation              | Python     | FP32/FP16      |
| [yolact](./sample/yolact/README.md)                           | Instance segmentation              | C++/Python | FP32/FP16/INT8 |
| [YOLOv8_seg](./sample/YOLOv8_seg/README.md)                   | Instance segmentation              | C++/Python | FP32/FP16/INT8 |
| [PP-OCR](./sample/PP-OCR/README.md)                           | OCR                                | C++/Python | FP32/FP16      |
| [OpenPose](./sample/OpenPose/README.md)                       | Keypoint detection                 | C++/Python | FP32/FP16/INT8 |
| [C3D](./sample/C3D/README.md)                                 | Video recognition                  | C++/Python | FP32/FP16/INT8 |
| [DeepSORT](./sample/DeepSORT/README.md)                       | Object tracking                    | C++/Python | FP32/FP16/INT8 |
| [ByteTrack](./sample/ByteTrack/README.md)                     | Object tracking                    | C++/Python | FP32/FP16/INT8 |
| [CenterNet](./sample/CenterNet/README.md)                     | Object Detection + pose estimation | C++/Python | FP32/FP16/INT8 |
| [YOLOv5](./sample/YOLOv5/README.md)                           | Object Detection                   | C++/Python | FP32/FP16/INT8 |
| [YOLOv34](./sample/YOLOv34/README.md)                         | Object Detection                   | C++/Python | FP32/INT8      |
| [YOLOX](./sample/YOLOX/README.md)                             | Object Detection                   | C++/Python | FP32/INT8      |
| [SSD](./sample/SSD/README.md)                                 | Object Detection                   | C++/Python | FP32/INT8      |
| [YOLOv7](./sample/YOLOv7/README.md)                           | Object Detection                   | C++/Python | FP32/FP16/INT8 |
| [YOLOv8_det](./sample/YOLOv8_det/README.md)                   | Object Detection                   | C++/Python | FP32/FP16/INT8 |
| [YOLOv5_opt](./sample/YOLOv5_opt/README.md)                   | Object Detection                   | C++/Python | FP32/FP16/INT8 |
| [ppYOLOv3](./sample/ppYOLOv3/README.md)                       | Object Detection                   | C++/Python | FP32/FP16/INT8 |
| [ppYoloe](./sample/ppYoloe/README.md)                         | Object Detection                   | C++/Python | FP32/FP16      |
| [WeNet](./sample/WeNet/README.md)                             | Speech Recognition                 | C++/Python | FP32/FP16      |
| [BERT](./sample/BERT/README.md)                               | Language                           | C++/Python | FP32/FP16      |
| [ChatGLM2](./sample/ChatGLM2/README.md)                       | Language                           | C++/Python | FP16/INT8/INT4 |
| [Llama2](./sample/Llama2/README.md)                           | Language                           | C++        | FP16/INT8/INT4 |
| [ChatGLM3](./sample/ChatGLM3/README.md)                       | Language                           | Python     | FP16/INT8/INT4 | 
| [Qwen](./sample/Qwen/README.md)                               | Language                           | Python     | FP16/INT8/INT4 | 
| [StableDiffusionV1.5](./sample/StableDiffusionV1_5/README.md) | Image Generation                   | Python     | FP32/FP16      |

| application                                                    | scenarios                 | code    | 
|---                                                             |---                       |---          | 
| [VLPR](./application/VLPR/README.md)                           | Multi-streams Vehicle License Plate Recognition | C++/Python  | 
| [YOLOv5_multi](./application/YOLOv5_multi/README.md)           | Multi-streams Object Detection       | C++         | 
| [YOLOv5_multi_QT](./application/YOLOv5_multi_QT/README.md)     | Multi-streams Object Detection + QT_HDMI display    | C++         | 

## Release Notes
| version | description | 
|---|---|
| 0.2.0  |  Fix documentation and other issues, release application/tutorial modules, release new sample ChatGLM3/Qwen, add a web ui in SAM, BERT/ByteTrack/C3D support BM1688, YOLOv8 is renamed to YOLOv8_det and add cpp postproces acceleration, optimize auto_test in commonly used samples, upgrade TPU-MLIR installation to pip |
| 0.1.10 | Fix documentation and other issues, add ppYoloe/YOLOv8_seg/StableDiffusionV1.5/SAM, refactor yolact, CenterNet/YOLOX/YOLOv8 support BM1688, YOLOv5/ResNet/PP-OCR/DeepSORT add BM1688 performance statis, WeNet provide C++ cross compile option. |
| 0.1.9	 | Fix documentation and other issues, add segformer/YOLOv7/Llama2, refactor YOLOv34/YOLOv5/ResNet/PP-OCR/DeepSORT/LPRNet/RetinaFace/YOLOv34/WeNet support BM1688, OpenPose postprocess acceleration, chatglm2 support int8/int4 and add compile method in readme.|
| 0.1.8  | Fix documentation and other issues, added BERT/ppYOLOv3/ChatGLM2, refactor YOLOX, added beam search to PP-OCR, added tpu-kernel post-processing acceleration to OpenPose, and updated the SFTP download method.|
| 0.1.7	 | Fix documentation and other issues, some demos support BM1684 mlir, refactor PP-OCR/CenterNet, sail support YOLOv5. |
| 0.1.6	 | Fix documentation and other issues, add ByteTrack/YOLOv5_opt/WeNet samples. |
| 0.1.5	 | Fix documentation and other issues, add DeepSORT sample, refactor ResNet/LPRNet samples. |
| 0.1.4 | Fix documentation and other issues, add C3D and YOLOv8 samples |
| 0.1.3 | Add OpenPose sample, refactor YOLOv5 sample (including supporting arm PCIe, supporting TPU-MLIR to compile BM1684X model, using ffmpeg component to replace opencv decoding, etc.) |
| 0.1.2 | Fix documentation and other issues, refactor SSD related samples, LPRNet/cpp/lprnet_bmcv use ffmpeg component to replace opencv decoding |
| 0.1.1 | Fix documentation and other issues, refactor LPRNet/cpp/lprnet_bmcv with BMNN related classes | 0.1.0 | Fix documentation and other issues, refactor LPRNet/cpp/lprnet_bmcv with BMNN related classes.
| 0.1.0 | Provide LPRNet and other 10 samples, support BM1684X (x86 PCIe, SoC), BM1684 (x86 PCIe, SoC) |

## Environment dependencies
SOPHON-DEMO mainly depends on TPU-MLIR, TPU-NNTC, LIBSOPHON, SOPHON-FFMPEG, SOPHON-OPENCV, SOPHON-SAIL, whose version requirements are as follows:  
|SOPHON-DEMO|TPU-MLIR |TPU-NNTC |LIBSOPHON|SOPHON-FFMPEG|SOPHON-OPENCV|SOPHON-SAIL| Release Date |
|--------|------------| --------|---------|---------    |----------   | ------    | ----------    |
| 0.2.0  | >=1.6       | >=3.1.7 | >=0.5.0 | >=0.7.3     | >=0.7.3     | >=3.7.0   | >=23.10.01|
| 0.1.10 | >=1.2.2    | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.7.0   | >=23.07.01  |
| 0.1.9  | >=1.2.2    | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.7.0   | >=23.07.01  |
| 0.1.8  | >=1.2.2    | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.6.0   | >=23.07.01  |
| 0.1.7  | >=1.2.2    | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.6.0   | >=23.07.01  |
| 0.1.6  | >=0.9.9    | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.4.0   | >=23.05.01  |
| 0.1.5  | >=0.9.9    | >=3.1.7 | >=0.4.6 | >=0.6.0     | >=0.6.0     | >=3.4.0   | >=23.03.01  |
| 0.1.4  | >=0.7.1    | >=3.1.5 | >=0.4.4 | >=0.5.1     | >=0.5.1     | >=3.3.0   | >=22.12.01  |
| 0.1.3  | >=0.7.1    | >=3.1.5 | >=0.4.4 | >=0.5.1     | >=0.5.1     | >=3.3.0   |    -        |
| 0.1.2  | Not support| >=3.1.4 | >=0.4.3 | >=0.5.0     | >=0.5.0     | >=3.2.0   |    -        |
| 0.1.1  | Not support| >=3.1.3 | >=0.4.2 | >=0.4.0     | >=0.4.0     | >=3.1.0   |    -        |
| 0.1.0  | Not support| >=3.1.3 | >=0.3.0 | >=0.2.4     | >=0.2.4     | >=3.1.0   |    -        |
> **Note**: 
> 1. The version requirements may vary from sample to sample, depending on the README of the routine, and other third-party libraries may need to be installed.
> 2. SDK for BM1688(CV186X) is not the same as BM1684/BM1684X, it is not released on our official site yet, you can contact us to get BM1688 SDK if you are interested in BM1688.

## Technical Data

Please get the related documents, materials and video tutorials through [Technical Materials](https://developer.sophgo.com/site/index.html) on the official website of SOPHGO.

## Community

The SOPHGO community encourages developers to communicate and learn together. Developers can communicate and learn through the following channels.

SOPHGO community website: https://www.sophgo.com/

SOPHGO Developer Forum: https://developer.sophgo.com/forum/index.html


## Contribution

Contributions are welcome. For more details, please refer to our [Contributor Wiki](./CONTRIBUTING_EN.md).

## License
[Apache License 2.0](./LICENSE)