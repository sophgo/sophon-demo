## SSD Python例程

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | ssd_bmcv.py | 使用SAIL解码、BMCV前处理、SAIL推理、batchsize=1 |
| 2    | ssd_bmcv_4b.py   | 使用SAIL解码、BMCV前处理、SAIL推理、batchsize=4     |

## 1. x86 PCIe平台
## 1.1 环境准备
如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)、sophon-ffmpeg(>=0.2.4)和sophon-sail(>=3.1.0),具体请参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。
此外您可能还需要安装其他第三方库：
```bash
pip3 install opencv-python opencv-python-headless
```
## 1.2 测试命令
python例程不需要编译，可以直接运行。ssd_bmcv.py和ssd_bmcv_4b.py参数相同，说明如下:
```shell
# bmodel: bmodel path, can be fp32 or int8 model
# input:  input path, can be image/video file or rtsp stream
# loops:  frames count to be detected, default: 1
# compare: conpare file path  (default is false)     
```
测试实例如下：
```
python3 ssd_bmcv.py --bmodel ../data/models/BM1684/ssd300_fp32_1b.bmodel --input ../data/videos/test_car_person.mp4
```
执行完成后，会将预测结果保存在同目录的result-1.jpg中，打印日志信息如下：
```
[Frame 1 on tpu 0] Category: 6, Score: 0.947, Box: [238, 3, 639, 341]
[Frame 1 on tpu 0] Category: 15, Score: 0.933, Box: [101, 143, 154, 298]
Open /dev/bm-sophon0 successfully, device index = 0, jpu fd = 18, vpp fd = 18
[2022-09-16 15:01:23.845] [info] [tensor.cpp:102] Start delete_shaptr_bm_handle_t_allocated!
[2022-09-16 15:01:23.845] [info] [tensor.cpp:105] End delete_shaptr_bm_handle_t_allocated!
```
## 2. arm SoC平台
## 2.1 环境准备
如果您使用SoC平台测试本例程，您需要交叉编译安装sophon-sail(>=3.1.0)，具体可参考[交叉编译安装sophon-sail](../../docs/Environment_Install_Guide.md#32-交叉编译安装sophon-sail)。
此外您可能还需要安装其他第三方库：
```bash
pip3 install opencv-python opencv-python-headless
```
## 2.2 测试命令
SoC平台的测试方法与x86 PCIe平台相同，请参考[1.2 测试命令](#12-测试命令)。
