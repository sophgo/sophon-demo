# Grounded_SAM_WEB_UI 例程

## 目录

- [Grounded_SAM\_WEB\_UI 例程](#Grounded_SAM_web_ui-例程)
  - [目录](#目录)
  - [简介](#简介)
  - [1. 工程目录](#1-工程目录)
  - [2. 准备模型与数据](#2-准备模型与数据)
  - [3. 环境准备](#3-环境准备)
  - [4. 启动前后端程序](#4-启动前后端程序)


## 简介
Grounded_SAM_WEB_UI 例程是一个基于 Grounded_SAM 模型的图像检测和分割系统，支持输入为图像与文本，输出为检测和分割后的图片和文本输入相关信息。可以实现只输入图片，就可以无交互式完全自动化标注出图片的检测框和分割掩码，能够应对从自动驾驶、机器人视觉到医疗影像分析等多样化的应用场景。

此例程由三部分组成：

1. 业务中台：`../python/gdsam_server.py`， 使用了数据队列和线程，具有数据队列管理等功能，为网页后端提供接口服务；
2. 前端应用：`gd_server-front.py`， 为使用streamlit搭建的网页前端；运行在client客户端；
3. 后端应用：`gd_server-back.py`， 为后端接口服务，为前端提供接口请求调用；运行在server服务器端，如SE7 SE9微服务器；

## 1. 工程目录

```bash
Grounded-sam
├── assets
│   └── dog.jpg
├── models
│   ├── bert-base-uncased     # tokenizer 分词器文件夹	
│   └── BM1684X
│       ├── decode_bmodel
│       │   └── SAM-ViT-B_decoder_single_mask_fp16_1b.bmodel      
│       ├── embedding_bmodel
│       │   └── SAM-ViT-B_embedding_fp16_1b.bmodel
│       └── groundingdino
│           └── groundingdino_bm1684x_fp16.bmodel     # 用于BM1684X的FP16 BModel，batch_size=1
├── python
│   ├── custom_model.py   # 定制模块，用于初始化和创建groundingdino和sam实例
│   ├── gdsam_server.py   # 业务中台
│   ├── gdsam_util.py     # #辅助函数文件
│   ├── groundingdino     # groundingdino实现模块
│   │   ├── groundingdino_pil.py
│   │   ├── PostProcess.py
│   │   └── utils.py
│   └── sam               # sam实现模块
│       ├── predictor.py
│       ├── sam_amg.py
│       ├── sam_encoder.py
│       ├── sam_model.py
│       ├── sam_opencv.py
│       └── transforms.py
├── requirements.txt          #例程的依赖模块
├── scripts
│   └── download.sh           #下载脚本
└── web_server
    ├── gd_server-back.py     # 后端接口服务
    └── gd_server-front.py    # 前端网页
```

## 2. 准备模型与数据
​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，
```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

## 3. 环境准备

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt
```

## 4. 启动前后端程序

1. 您需要启动前后端程序，前端程序运行在您的客户端，后端程序运行在您的服务器端，如SC7服务器。

```bash
cd web_server
python3 gd_server-back.py --host 0.0.0.0 --port 8080 # 启动后端接口服务，在您的服务器端启动，如SE7 SE9微服务器，其中--host 0.0.0.0 --port 8080 用于指定后端服务器的地址和端口
streamlit run gd_server-front.py  # 启动前端网页，在您的客户端启动，会在终端显示前端网页的服务器地址和端口
```

推荐Prompt的使用语言为英文；

2. 您可以选择本地图片上传，并输入文本，点击“UPLOAD”按钮，您将看到预测结果；

  2.1 若您不选择本地图片，默认为本例程../assets/dog.jpg 的图片，直接在页面的Prompt处输入如下Prompt： 

    > dog and tree

  2.2. 点击“UPLOAD”按钮，您将看到预测结果

  2.3. 若您多次提交图片，您可以在“历史记录”中查看您的提交记录