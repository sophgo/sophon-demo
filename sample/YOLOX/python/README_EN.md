[简体中文](./README.md) | [English](./README_EN.md)

# Python Demo

## Content

* [Python Demo](#python-demo)
    * [1. Environments Preparation](#1-environments-preparation)
        * [1.1 x86/arm PCIe Platform](#11-x86arm-pcie-platform)
        * [1.2 SoC Platform](#12-soc-platform)
    * [2. Inference Test](#2-inference-test)
        * [2.1 Parameter Description](#21-parameter-description)
        * [2.2 Image Test Demo](#22-image-test-demo)
        * [2.3 Video Test Demo](#23-video-test-demo)

A series of Python demos are provided under the python directory, the details are as follows:

| No. |  Python Demo      | Description                     |
| ---- | ---------------- | -----------------------------------  |
| 1    | yolox_opencv.py | Decoding and preprocessing with OpenCV, Inference with SAIL |
| 2    | yolox_bmcv.py   | Decoding with SAIL, preprocessing with BMCV, Inference with SAIL |

## 1. Environments Preparation
### 1.1 x86/arm PCIe Platform

If you have installed a PCIe accelerator card (such as SC series accelerator card) on the x86/arm platform and use it to test these demos, you need to install libsophon, sophon-opencv, sophon-ffmpeg and sophon-sail. For specific steps, please refer to [Construction of Development and Runtime Environment of x86-PCIe Platform](../../../docs/Environment_Install_Guide_EN.md#3-x86-pcie-platform-development-and-runtime-environment-construction) or [Construction of Development and Runtime Environment of arm-PCIe Platform](../../../docs/Environment_Install_Guide_EN.md#5-arm-pcie-platform-development-and-runtime-environment-construction).

In addition, you may need to install other third-party libraries:
```bash
pip3 install 'opencv-python-headless<4.3'
```

### 1.2 SoC Platform

If you use the SoC platform (such as SE, SM series edge devices), the corresponding libsophon, sophon-opencv and sophon-ffmpeg runtime packages have been pre-installed under `/opt/sophon/` after resetting the operating system. You also need to cross compile and install sophon-sail, for details, please refer to [Cross Compile and Install sophon-sail](../../../docs/Environment_Install_Guide_EN.md#42-cross-compiling-and-sophon-sail-installation)。

You may need to install other third-party libraries:
```bash
pip3 install 'opencv-python-headless<4.3'
```

If you want to use sophon-opencv directly, you can directly set the following environment variables without performing the above steps.
```bash
export PYTHONPATH=/opt/sophon/sophon-opencv-latest/opencv-python:$PYTHONPATH
```

## 2. Inference Test
The python demo does not need to be compiled and can be run directly. The test parameters and operation methods of PCIe platform and SoC platform are the same.
### 2.1 Parameter Description
The parameters of yolox_opencv.py and yolox_bmcv.py are the same. Here we take yolox_opencv.py as an example:
```bash
usage: yolox_opencv.py [--input INPUT_PATH] [--bmodel BMODEL] [--dev_id DEV_ID]
                        [--conf_thresh CONF_THRESH] [--nms_thresh NMS_THRESH]
--input: The test data path. User can enter the path of the entire picture folder or video path;
--bmodel: bmodel path for inference. By default, the stage 0 network is used for inference;
--dev_id: tpu device id for inference;
--conf_thresh: confidence threshold;
--nms_thresh: Non-Maximum Suppression threshold.
```
### 2.2 Image Test Demo
The image test demo is as follows. It supports testing the entire image folder. 
```bash
python3 python/yolox_opencv.py --input datasets/test --bmodel models/BM1684/yolox_s_fp32_1b.bmodel --dev_id 0 --conf_thresh 0.5 --nms_thresh 0.5
```
After the test, the predicted image will be saved in `results/images`, the predicted result will be saved in `results/yolox_s_fp32_1b.bmodel_test_opencv_python_result.json`, and information such as predicted results and inference time will be printed at the same time.


![res](../pics/zidane_python_opencv.jpg)

### 2.3 Video Test Demo
The video test demo is as follows, which supports testing of video streams. 
```bash
python3 python/yolox_opencv.py --input datasets/test_car_person_1080P.mp4 --bmodel models/BM1684/yolox_s_fp32_1b.bmodel --dev_id 0 --conf_thresh 0.5 --nms_thresh 0.5
```
After the test, the predicted results will be drawn in `results/test_car_person_1080P.avi`, and information such as predicted results and inference time will be printed at the same time.
