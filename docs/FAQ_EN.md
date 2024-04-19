[简体中文](./FAQ.md) | [English](./FAQ_EN.md)

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
### 1.1 The installation of development and running environments of sophon-demo can be found in [related documentation](./Environment_Install_Guide_EN.md).

### 1.2 How Do I Use an SD Card to update Firmware in SoC Mode
Download and decompress SophonSDK after v22.09.02, find sdcard swiping package in sophon-img folder, refer to [relative document](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/faq/html/devices/SOC/soc_firmware_update.html#id6) to refresh the firmware.


## 2 Model derived related problems
### 2.1 How to jit.trace the Pytorch Model
**You should use the torch vision of`1.8.0+cpu`to avoid model compilation failures due to the pytorch version.** Some open source repositories provide scripts for model exports. If not provided, you need to learn from   [relative document](./torch.jit.trace_Guide_EN.md) or [Pytorch official document](https://pytorch.org/docs/stable/jit.html) and write 'jit.trace' script. 'jit.trace' is usually done in torch environment and does not require a SophonSDK installation.
- 'jit.trace' method of YOLOv5 model is available in [related documentation](../sample/YOLOv5/docs/YOLOv5_Export_Guide_EN.md)

## 3 Issues related to model compilation and quantization
### 3.1 Problems related to quantization using TPU-NNTC
For details, see [documentation](./Calibration_Guide_EN.md).

### 3.2 Long time to compile large models (such as resnet260) using TPU-NNTC
It is possible that the resnet260 compilation took so long because the second search took too long. When compiling and optimizing, we conducted a group search on the results after the initial layergroup. This search result did not bring much benefit to the network performance like resnet, but would increase the time of compiling and optimizing. In order to solve such model problems, we increase the upper limit of secondary search and control the upper limit of secondary search through the following environment variables:

```bash
export BMCOMPILER_GROUP_SEARCH_COUNT=1
```

## 4 Algorithm migration related problems
### 4.1 Relationship between Sophon OpenCV and native OpenCV and BMCV
Sophon OpenCV inherits and optimizes native OpenCV, modifies native mat, increases device memory, and adds hardware acceleration support for some interfaces (such as imread, videocapture, videowriter, resize, convert, etc.), keeping the interface consistent with native OpenCV operations. The specific information of modifying please see the [relevant documentation](https://doc.sophgo.com/sdk-docs/v22.12.01/docs_latest_release/docs/sophon-mw/guide/html/1_guide.html).

BMCV is a set of acceleration library provided by us, based on hardware VPP and TPU for image processing and some mathematical operations. It is a library of C interface. At the bottom of our modified Sophon OpenCV, hardware-accelerated operations use the interface of BMCV as well.

Compared with native OpenCV, Sophon OpenCV adopts different upsample algorithms for decoding using hardware-accelerated units in TPU. The decoding and pre-processing methods of Sophon OpenCV are different from those of native OpenCV to some degree, which may affect the final prediction results, but usually does not have a significant impact on robust models.

### 4.2 How do OpenCV-based Python demos call Sophon OpenCV for acceleration
Using SophonSDK after v22.09.02, whether in PCIe or SoC mode, Python demos based on OpenCV use the installed native OpenCV by default. Sophon OpenCV can be used by setting environment variables:
```bash
export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
```
Note that the use of sophon-opencv may lead to differences in reasoning results.

### 4.3 OpenCV-based C++ demos using Sophon OpenCV or native OpenCV
The OpenCV-based C++ demo sets the OpenCV_DIR path in CMakeLists.txt, linking Sophon OpenCV's relevant headers and library files. To invoke native OpenCV, install native OpenCV and modify the link path.

### 4.4 The C++ demo decoding video using ff_decode displays "vid reset unlock flock failed", but normal operation is not affected
This is because the cache is not cleared in time, and the inference result is not affected. To solve the problem, run the following command:
```bash
sudo rm /tmp/vid_*
```

## 5 Accuracy test related issues
### 5.1 The inference result of FP32 BModel is inconsistent with that of the original model
Under the premise that the pre and post processing is aligned with the original algorithm, the maximum error between the accuracy of FP32 BModel and the original model is usually less than 0.001, which will not affect the final prediction results. Refer to [related documentation](./FP32BModel_Precise_Alignment_EN.md) for precision alignment of the FP32BModel.

## 6 Performance test related issues
### 6.1 Part of FP32 BModel BM1684X performance is lower than BM1684
The local memory of the BM1684X is half of that of the BM1684. The model with a large number of parameters may not be able to fit, which results in a large increase in the number of accessing ddr by gdma.

### 6.2 Inference time of int8bmodel demo based on opencv-python is not faster than fp32bmodel
The input data type of int8bmodel is int8 with a scale factor that is not equal to 1. The opencv-python demo based on numpy.array as input needs to perform a multiplication operation with the scale factor inside the inference interface. On the other hand, the scale factor of the input layer of fp32bmodel is 1, and there is no need to perform multiplication with the scale factor inside the inference interface. This part of the time may offset the time saved by model inference optimization. You can add `sail.set_print_flag(1)` in the code to print the specific inference time of the interface.

If you want to use the opencv-python demo for deployment, it is recommended to keep the input and output layers of int8bmodel as floating-point calculations.

### 6.3 Can sophon-demo be used for performance testing?
It is not recommended. Sophon-demo provides a series of transplant demos for mainstream algorithms, and users can perform model algorithm transplantation and accuracy testing based on sophon-demo. However, the preprocessing/inference/post-processing of sophon-demo is serial, and even if multiple processes are opened, it is difficult to fully utilize the performance of TPU.  can run preprocessing/inference/post-processing on three threads respectively, maximizing parallelism. Therefore, it is recommended to use sophon-pipeline for performance testing.

### 6.4 Performance testing results are lower than README
If the performance drops more than 10% compared to the README, you need to confirm the product model. The sophon-demo demos are all based on standard products (such as SE-16) for testing. If you are using a low-end product (such as SE5-8), performance degradation is normal. The int8 computing power of SE5-16 is 17.6 TOPS, while that of SE5-8 is 10.6 TOPS, so there may be a loss of 2/5 of the performance.

If you are also using a standard product and encounter performance degradation issues, you can report the issue to the Sophon team or create an issue on GitHub.

### 6.5 What does the performance test table do?
The performance test table provided by sophon-demo generally includes four parts: decode\preprocess\inference\postprocess. This is how algorithm deployment is mostly divided in the industry. For Sophgo AI processors, these four parts can all be concurrent because they each rely on different devices. For example, decode relies on VPU, preprocess relies on VPP, inference relies on TPU, and postprocess relies on the central processor. These components can work concurrently without being affected individually. The performance test table provided by sophon-demo lists this information and can intuitively analyze the bottlenecks of the current algorithm and the theoretical maximum number of processing times per second it can achieve.


## 7 Other questions
### 7.1 `Unkown CMake command "add_compile_definitions".` when compiling C++ programs.
This is because your cmake do not have `add_compile_definitions` function, you can change it to `add_definitions` or upgrade your cmake to 3.12 or higher.

### 7.2 Some images have data overlapped, some have no data on the frame, and some images have no frame
It is normal for there to be a small number of missed and false detections in the image and video test results. Our samples only needs to ensure that the accuracy of the transplanted model and the source model can be aligned, and the accuracy index test should prevail to judge the model effect.

### 7.3 `bm_ion_alloc failed` occured during program running.
First, check if there is any device memory leak, such as alloc a piece of device memory but do not release after its life cycle.
If you can confirm no device memory leak happened, then this issue is probably because a device memory heap being too small, use these commands to see each heap's memory usage.
```bash
#on SE5/SE7 series use:
sudo cat /sys/kernel/debug/ion/bm_vpu_heap_dump/summary
sudo cat /sys/kernel/debug/ion/bm_vpp_heap_dump/summary
sudo cat /sys/kernel/debug/ion/bm_npu_heap_dump/summary
#on SE5/SE7 series use:
sudo cat /sys/kernel/debug/ion/cvi_vpp_heap_dump/summary
sudo cat /sys/kernel/debug/ion/cvi_npu_heap_dump/summary
#such command will print
Summary:
[1] vpp heap size:4294967296 bytes, used:144646144 bytes
usage rate:4%, memory usage peak 144646144 bytes #memory usage peak is the highest usage since startup
```
if a heap's `memory usage peak` is very closed to its `heap size`, you can use this tool to manually set each heap's size: [sophon memedit tool](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html).

### 7.4 The first time a function is executed slowly (such as decoding) just after booting, restart the process and run the program again, the time is normal

It may be that the file has not been cached in memory, which is slow at first.
You can do a verification, if you don't restart it can be reproduced, it means that the program is slow to run just now, and the file has not been cached, the steps are as follows:
    
1. Execute the program after powering on, the first execution is slow, and the second execution is normal.

2. Then go to the root user to clear the cache and run the command `echo 3 > /proc/sys/vm/drop_caches`.
    
3. If the program is slow to run again, it can be determined that it is caused by cache.