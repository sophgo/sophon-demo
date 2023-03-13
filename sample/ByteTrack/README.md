# ByteTrack

**Deploy YOLOX+ByteTrack target tracking using OpenCV and BMCV, including both C++ and Python versions of the program.**

- [ByteTrack](#bytetrack)
	- [1.Introduction](#1introduction)
	- [2.Abstract](#2abstract)
	- [3. Preparation](#3-preparation)
	- [4. Model Compilation](#4-model-compilation)
	- [5. Deployment and Testing](#5-deployment-and-testing)
		- [5.1 Automated Testing through Scripts](#51-automated-testing-through-scripts)
		- [5.2 C++ Routine Deployment and Testing](#52-c-routine-deployment-and-testing)
		- [5.3 Python Routine Deployment and Testing](#53-python-routine-deployment-and-testing)

## 1.Introduction
ByteTrack is a simple, fast and strong multi-object tracker.

**Paper** (https://arxiv.org/abs/2110.06864)

**Source Code** (https://github.com/ifzhang/ByteTrack)

## 2.Abstract
Multi-object tracking (MOT) aims at estimating bounding boxes and identities of objects in videos. Most methods obtain identities by associating detection boxes whose scores are higher than a threshold. The objects with low detection scores, e.g. occluded objects, are simply thrown away, which brings non-negligible true object missing and fragmented trajectories. To solve this problem, we present a simple, effective and generic association method, tracking by associating every detection box instead of only the high score ones. For the low score detection boxes, we utilize their similarities with tracklets to recover true objects and filter out the background detections.

## 3. Preparation
Pytorch's model need to goes through 'torch.jit.trace' before compiling, and the traced model can be used to compile BModel. The method and principle of trace can be found in [torch.jit.trace Guide](../../docs/torch.jit.trace_Guide.md)。

At the same time, you need to prepare a dataset for testing and, if quantizing the model, a dataset for quantization.

This routine provides the download script 'download.sh' of the relevant model and dataset in the 'scripts' directory, and automatically downloads the pt model, dataset and BModel after running, that is, you can skip Chapter 4 model compilation. You can also use the downloaded pt model and quantization dataset, or prepare the model and dataset yourself, and refer to [4. Model compilation](#4-model-compilation) for model conversion to generate BModel.

Make sure you don't have folder '/data', then
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install unzip
cd ./scripts
chmod +x download.sh
./download.sh
```

After execution, the model is saved to 'data/models', the test video is saved to 'data/video', and the test dataset file is saved to 'data/MOT5'
```
Downloaded models include:
/data/models/onnx/bytetrack_s.onnx: Onnx model
/data/models/BM1684/bytetrack_s_fp32_1b.bmodel: FP32 BModel, used for BM1684，batch_size=1
/data/models/BM1684X/bytetrack_s_st_fp32_1b.bmodel: FP32 BModel, used for BM1684X，batch_size=1

The downloaded datasets include:
/data/MOT15/dataset-name：Used to test accuracy metrics

Downloaded test videos include:
/data/video/sample.mp4：Test video
```

## 4. Model Compilation

If you directly use the BModel downloaded in the previous step, you can skip this section.

You can use the onnx model downloaded in the previous step at data/models/onnx/bytetrack_s.onnx, or you can use your own model and quantify the dataset.

TPU-NNTC (>=3.1.0) needs to be installed before model compilation, please refer to [tpu-nntc环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。

The pytorch model is compiled into FP32 BModel, and the specific method can be found in the TPU-NNTC Development Reference Manual.

This example provides a script to compile FP32 BModel in the 'scripts' directory.

```bash
cd ./scripts
chmod +x gen_fp32bmodel.sh
./gen_fp32bmodel.sh
```

Executing the above command will generate the bmodel files under BM1684 and BM1684X under 'data/models/', that is, the converted FP32 BModel.

## 5. Deployment and Testing

### 5.1 Automated Testing through Scripts

This automated test script needs to be performed on an x86 host or Sophi SoC device with a PCIe accelerator card.

Rely on the python package 'motmetrics'
```bash
pip3 install motmetrics
```

After preparing the BModel with test data:

```bash
cd scripts
chmod +x ./auto_test.sh
./auto_test
./auto_test.sh ${platform} ${target} ${tpu_id} ${sail_dir}
```

where 'platform' refers to the platform (x86 or soc), 'target' is the chip model (BM1684 or BM1684X), 'tpu_id' specifies the ID of the TPU (viewed using BM-SMI), 'sail_dir' is the installation path of SAIL. If the final output is 'Failed:', execution failed, otherwise it indicates success.

For example,

```bash
./auto_test.sh soc BM1684 0 /opt/sophon/sophon-sail
```

On x86, 'auto_test.sh' includes the compilation and operation of C++ programs in the cpp folder and the running of all Python programs in the Python folder, as well as the operation of MOT metrics scripts.

On soc, auto_test.sh includes the operation of C++ programs in the cpp folder and the operation of all Python programs in the Python folder, as well as the operation of MOT metrics scripts.


To execute this script on x86, refer to [x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)，then run this script, where ${sail_dir} builds the resulting sophon-sail installation path for the above environment, usually /opt/sophon/sophon-sail.

To execute this script on SoC, you first need to cross-compile the ARM program, refer to [交叉编译环境搭建](../../docs/Environment_Install_Guide.md#31-交叉编译环境搭建)，then move the generated executable to the cpp folder. After that, set the environment variables.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
```

Run this script again, where ${sail_dir} is the build_soc/sophon-sail folder created for the above environment.

### 5.2 C++ Routine Deployment and Testing

For detailed steps, refer to the [Readme.md](./cpp/README.md) under cpp folder

### 5.3 Python Routine Deployment and Testing

For detailed steps, refer to the [Readme.md](./python/README.md) under python folder
