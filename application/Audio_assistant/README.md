# Audio_assistant 例程

## 目录

- [Audio_assistant 例程](#Audio_assistant-例程)
  - [目录](#目录)
  - [简介](#简介)
  - [特性](#特性)
  - [1. 工程目录](#1-工程目录)
  - [2. 运行环境准备](#2-运行环境准备)
  - [3. 例程](#3-例程)
  - [4. FAQ](#4-FAQ)


## 简介
Audio_assistant 例程是一个基于Whisper、MiniCPM、Llama3、VITS模型的语音助手系统，支持输入为音频，输出为音频，其中包括对输入音频内容回答的相关信息，暂只支持中文。可以实现流畅的人机交互，能够应用到智能机器人、智能家具、自动驾驶等多样化的应用场景。

## 特性
* 支持BM1688(SoC)
* 支持BM1684X(PCIE、SOC)
* 支持本机麦克风、文件输入，本机喇叭、文件输出
* 支持网络麦克风，网络喇叭

## 1. 工程目录

```bash
Audio_assistant
├── models
│   ├── BM1688
│   │   ├── minicpm    # BM1688 MiniCPM LLM模型文件夹	
│   │   ├── vits       # BM1688 VITS语音生成模型文件夹	
│   │   └── whisper    # BM1688 Whisper语音识别模型文件夹
│   ├── BM1684X
│   │   ├── llama3     # BM1684X Llama3 LLM模型文件夹	
│   │   ├── vits       # BM1684X VITS语音生成模型文件夹	
│   │   └── whisper    # BM1684X Whisper语音识别模型文件夹
├── datasets       # 包含了音频测试文件
├── python
│   ├── socket_demo       # socket网络应用文件夹
│   │   ├── client        # 客户端文件夹，包括源码和依赖库信息
│   │   └── service       # 服务端文件夹，包括源码和依赖库信息
│   ├── Llama3     # Llama3源码和依赖库文件夹
│   │   ├── python_demo   # Llama3源码文件夹
│   │   └── support       # Llama3依赖库文件夹
│   ├── libfirmware_core.so # Bmodel运行时库
│   ├── MiniCPM    # MiniCPM源码和依赖库文件夹
│   │   ├── demo          # MiniCPM源码文件夹
│   │   └── support       # MiniCPM依赖库文件夹
│   ├── whisper-TPU_py    # Whisper源码文件夹
│   ├── whisper_minicpm_llama3_vits.py     # 全流程串通源代码
│   └── whisperWrapper.py                  # Whisper接口源代码
└── scripts
    ├── download_bm1684x_whisper_llama3_vits.sh       # BM1684X模型下载脚本，包括Whisper-small、Whisper-medium、Whisper-base、Llama3-8b、VITS-chinese
    ├── download_bm1688_whisper_minicpm_vits.sh       # BM1688模型下载脚本，包括Whisper-small、Whisper-base、MiniCPM-2b、VITS-chinese
    ├── download_datasets.sh                          # 音频测试文件下载脚本
    └── download.sh                                   # 完整的模型、数据下载脚本
```

## 2. 运行环境准备
在PCIe和BM1688 SOC上无需修改内存，对于1684X SOC模式系列设备（如SE7/SM7），都可以通过如下方式完成环境准备，使得满足运行条件，参考如下命令修改设备内存分布：
```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/09/11/13/DeviceMemoryModificationKit.tgz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_v2.9.tar.xz
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息
./memory_edit.sh -c -npu 6616 -vpu 512 -vpp 3072 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot
```
> **注意：**
> 1. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)。

## 3. 例程
- [Python例程](./python/README.md)
- [socket_demo例程](./python/socket_demo/README.md)

## 4. FAQ
- 若使用麦克风不能进行录音，则需要检查和修改参数`--microphone_devid`。排查流程如下：
  - 首先若使用的USB麦克风，不是则跳到下一步骤，需要添加当前用户到`audio`用户组，执行:
    ```bash
    sudo usermod -a -G audio $(whoami)
    ```

  - 然后需要使用如下命令查看所有音频输入设备，获得所有音频输入设备的`{card id}`和`{device id}`两个id
    ```bash
    # 安装必要的音频库
    sudo apt-get install alsa-utils
    arecord -l
    ```

  - 使用如下命令尝试录音，其中`-d 5`参数是指录5s音频，`-D hw:{card id},{device id}`参数用于指定设备，来自上一步骤命令输出，`{fs}`是指输入音频的采样率，部分麦克风仅仅支持`44100`或`16000`，需要测试获得
    ```bash
    arecord -D hw:{card id},{device id} -c 1 -f S16_LE -r {fs} -d 5 test.wav
    ```

  - 通过上述步骤即可获得能够录音的设备名称和采样率，接着执行如下命令即可将设备名称和程序里面音频输入设备id`microphone_devid`对应，即命令输出的输入设备里面的`index`字段
    ```bash
    python3 tools/get_audio_device.py
    ```

- 若使用喇叭不能进行播放，则需要检查和修改参数`--audio_devid`。排查流程如下：
  - 首先若使用的USB喇叭，不是则跳到下一步骤，需要添加当前用户到`audio`用户组，执行:
    ```bash
    sudo usermod -a -G audio $(whoami)
    ```

  - 然后需要使用如下命令查看所有音频输出设备，获得所有音频输出设备的`{card id}`和`{device id}`两个id
    ```bash
    # 安装必要的音频库
    sudo apt-get install alsa-utils
    aplay -l
    ```

  - 使用如下命令尝试播放，其中`-D plughw:{card id},{device id}`参数用于指定设备，来自上一步骤命令输出，`{fs}`是指输出音频的采样率，部分喇叭仅仅支持`44100`或`16000`，需要测试获得
    ```bash
    aplay -D plughw:{card id},{device id} -f S16_LE -r {fs} /path/to/audio/file
    ```

  - 通过上述步骤即可获得能够播放的设备名称和采样率，接着执行如下命令即可将设备名称和程序里面音频输出设备id`audio_devid`对应，即命令输出的输出设备里面的`index`字段
    ```bash
    python3 tools/get_audio_device.py
    ```