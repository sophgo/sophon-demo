# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 测试](#2-测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 测试视频：](#22-测试视频)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | [encode_sail.py](./encode_sail.py) | 使用SAIL将输入视频、图片、rtsp流保存为视频、图片，或转发为rtsp流 |

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

## 2. 测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明
```bash
usage: push_stream.py [--input_path INPUT_FILE_PATH] [--output_path OUTPUT_PATH] \
[--device_id DEVICE_ID] [--compressed_nv12 COMPRESSED_NV12] [--height HEIGHT] [--width WIDTH] \
[--enc_fmt ENC_FMT] [--bitrate BITRATE] [--pix_fmt PIX_FMT] [--gop GOP] [--gop_preset GOP_PRESET] \
[--framerate FRAMERATE]

--input_path: 输入视频文件的路径，例如 "input_video.mp4"；
--output_path: 推流输出的路径，默认是 "./output"；
--device_id: 使用的设备ID，默认为 0；
--compressed_nv12: 是否使用压缩的NV12格式，True 或 False，默认为 True；
--height: 视频帧的高度，默认为 1080；
--width: 视频帧的宽度，默认为 1920；
--enc_fmt: 编码格式，例如 "h264_bm"；
--bitrate: 码率（单位: Kbps），默认为 2000；
--pix_fmt: 像素格式，例如 "NV12"；
--gop: 关键帧间隔，默认为 32；
--gop_preset: GOP预设值，默认为 2；
--framerate: 帧率，默认为 25帧每秒。
```

### 2.2 测试视频：
视频测试实例1如下：
```bash
python3 python/encode_sail.py --input_path rtsp://127.0.0.1:8554/0 --output_path test.mp4
```
测试完成后，会将接受的rtsp流保存为test.mp4文件。

视频测试实例2如下：
```bash
python3 python/encode_sail.py --input_path rtsp://127.0.0.1:8554/0 --output_path rtsp://127.0.0.1:8554/1
```
测试开始后，会将接受的rtsp流转发到rtsp://127.0.0.1:8554/1。