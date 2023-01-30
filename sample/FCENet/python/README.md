# Python例程

# 目录
* [1 环境准备](#1-环境准备)
	* [1.1 x86 PCIe平台](#1.1-x86-PCIe平台)
	* [1.2 SoC平台](#1.2-SoC平台)
* [2 测试命令](#2-测试命令)


python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | fcenet_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |
| 2    | fcenet_bmcv.py   | 使用SAIL解码、BMCV前处理、SAIL推理     |

## 1 环境准备
### 1.1 x86 PCIe平台
如果您在x86平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt
```


## 2 测试命令
fcenet_opencv.py和fcenet_bmcv.py的命令参数相同，以fcenet_opencv.py的推理为例，参数说明如下：

```bash
usage:fcenetnet_opencv.py [--input_path IMG_PATH] [--bmodel BMODEL] [--tpu_id TPU]
--input_path:推理图片路径，可输入整个图片文件夹的路径；
--bmodel:用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--tpu_id:用于推理的tpu设备id。
```

测试实例如下：
```bash
# 测试整个文件夹
python3 fcenet_opencv.py --input_path ../datasets/ctw1500/imgs/test_opencv_read_write --bmodel ../models/BM1684/fcenet_fp32_b1.bmodel --tpu_id 0
```

执行完成后，会将预测结果保存在`results/fcenet_fp32_b1.bmodel_test_opencv_read_write_opencv_python_result.txt`下

可通过改变模型进行int8及batch_size=4的推理测试。