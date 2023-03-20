# FAQ

## Directory

* [1 Environment installation related Problems](#1-environment-installation-related-problems)

* [2 Model derived Problems](#2-model-derived-related-problems)
* [3 Issues related to Model compilation and Quantization](#3-issues-related-to-model-compilation-and-quantization)
* [4 Algorithm migration related problems](#4-algorithm-migration-related-problems)
* [5 Accuracy test related Problems](#5-accuracy-test-related-issues)
* [6 Performance Test Related Issues](#6-performance-test-related-issues)
* [7 Other questions](#7-other-questions)

We have listed some common problems that users and developers will encounter during the development process and corresponding solutions. If you find any problems, please feel free to contact us or create a related issue. Any questions or solutions you put forward are very welcomed.


## 1 Environment installation related problems
### 1.1 The installation of development and running environments of sophon-demo can be found in [related documentation](./Environment_Install_Guide.md).

### 1.2 How Do I Use an SD Card to update Firmware in SoC Mode
Download and decompress SophonSDK after v22.09.02, find sdcard swiping package in sophon-img folder, refer to [relative document](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/faq/html/devices/SOC/soc_firmware_update.html#id6) to refresh the firmware.


## 2 Model derived related problems
### 2.1 How to jit.trace the Pytorch Model
Some open source repositories provide scripts for model exports. If not provided, you need to learn from   [relative document](./torch.jit.trace_Guide.md) or [Pytorch official document](https://pytorch.org/docs/stable/jit.html) and write 'jit.trace' script. 'jit.trace' is usually done in torch environment and does not require a SophonSDK installation.
- 'jit.trace' method of YOLOv5 model is available in [related documentation](../sample/YOLOv5/docs/YOLOv5_Export_Guide.md)

## 3 Issues related to model compilation and quantization
### 3.1 Problems related to quantization using TPU-NNTC
For details, see [documentation](./Calibration_Guide.md).

### 3.2 Long time to compile large models (such as resnet260) using TPU-NNTC
It is possible that the resnet260 compilation took so long because the second search took too long. When compiling and optimizing, we conducted a group search on the results after the initial layergroup. This search result did not bring much benefit to the network performance like resnet, but would increase the time of compiling and optimizing. In order to solve such model problems, we increase the upper limit of secondary search and control the upper limit of secondary search through the following environment variables:

```bash
export BMCOMPILER_GROUP_SEARCH_COUNT=1
```

## 4 Algorithm migration related problems
### 4.1 Relationship between Sophon OpenCV and native OpenCV and BMCV
Sophon OpenCV inherits and optimizes native OpenCV, modifies native mat, increases device memory, and adds hardware acceleration support for some interfaces (such as imread, videocapture, videowriter, resize, convert, etc.), keeping the interface consistent with native OpenCV operations. The specific information of modifying please see the [relevant documentation](https://doc.sophgo.com/sdk-docs/v22.12.01/docs_latest_release/docs/sophon-mw/guide/html/1_guide.html).

BMCV is a set of acceleration library provided by us, based on hardware VPP and TPU for image processing and some mathematical operations. It is a library of C interface. At the bottom of our modified Sophon OpenCV, hardware-accelerated operations use the interface of BMCV as well.

Compared with native OpenCV, Sophon OpenCV adopts different upsample algorithms for decoding using hardware-accelerated units in chips. The decoding and pre-processing methods of Sophon OpenCV are different from those of native OpenCV to some degree, which may affect the final prediction results, but usually does not have a significant impact on robust models.

### 4.2 How do OpenCV-based Python routines call Sophon OpenCV for acceleration
Using SophonSDK after v22.09.02, whether in PCIe or SoC mode, Python routines based on OpenCV use the installed native OpenCV by default. Sophon OpenCV can be used by setting environment variables:
```bash
export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
```
Note that the use of sophon-opencv may lead to differences in reasoning results.

### 4.3 OpenCV-based C++ routines using Sophon OpenCV or native OpenCV
The OpenCV-based C++ routine sets the OpenCV_DIR path in CMakeLists.txt, linking Sophon OpenCV's relevant headers and library files. To invoke native OpenCV, install native OpenCV and modify the link path.

### 4.4 The C++ routine decoding video using ff_decode displays "vid reset unlock flock failed", but normal operation is not affected
This is because the cache is not cleared in time, and the inference result is not affected. To solve the problem, run the following command:
```bash
sudo rm /tmp/vid_*
```

### 4.5 The C++ routine uses ff_decode to decode the input picture with the suffix uppercase times "not support pic format, only support jpg and png"
The current solution is to change the suffix of the input image to lower case. In the future, the image format will be determined directly by the image input, rather than by the name.

## 5 Accuracy test related issues
### 5.1 The inference result of FP32 BModel is inconsistent with that of the original model
Under the premise that the pre and post processing is aligned with the original algorithm, the maximum error between the accuracy of FP32 BModel and the original model is usually less than 0.001, which will not affect the final prediction results. Refer to [related documentation](./FP32BModel_Precise_Alignment.md) for precision alignment of the FP32BModel.

## 6 Performance test related issues
### 6.1 Part of FP32 BModel BM1684X performance is lower than BM1684
The local memory of the BM1684X is half of that of the BM1684. The model with a large number of parameters may not be able to fit, which results in a large increase in the number of accessing ddr by gdma.

## 7 Other questions