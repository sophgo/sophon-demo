[简体中文](./README.md) | [English](./README_EN.md)

# WeChatQRCode

## 目录

- [WeChatQRCode](#wechatqrcode)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK、算法特性](#22-sdk算法特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 模型编译](#32-模型编译)
  - [4. 例程测试](#4-例程测试)
    - [4.1 C++ 例程测试](#41-c-例程测试)
    - [4.2 Python 例程测试](#42-python-例程测试)
  - [5. 精度测试](#5-精度测试)
    - [5.1 测试方法](#51-测试方法)
    - [5.2 测试结果](#52-测试结果)
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  - [7. FAQ](#7-faq)

## 1. 简介

WeChatQRCode 是 OpenCV 4.8.0 起内置的微信二维码**检测 + 识别**算法，其完整流程为：SDD-MobileNet 目标检测（detect，定位二维码位置）→ 按框裁剪并外扩填充 → 小图超分（sr，SRResNet 超分）+ 多尺度缩放 → zxing 解码（CPU）→ 角点坐标回投影与去重，最终输出每个二维码的文本与四个角点。本例程对 OpenCV 原生的 wechat_qrcode 模型与算法进行移植，使其能在 SOPHON BM1684X/BM1688/CV186X 上进行推理测试，其中 detect 与 sr 两个模型在 TPU 上推理，zxing 解码在 CPU 上完成。

本例程同时提供了 4 种实现以满足不同使用习惯：

| 实现 | 语言 | 图像加载 | 算法来源 | 说明 |
|------|------|----------|----------|------|
| `cpp/wechat_qrcode_bmcv` | C++ | sophon-opencv `cv::imread` | 源码移植（纯 bmrt + bmcv，不依赖 sail） | 完整移植 detect/crop/sr/zxing 管线，vendor 了 zxing |
| `cpp/wechat_qrcode_sail` | C++ | `sail::Decoder`（bench）/ sophon-opencv `cv::imread`（目录） | 调用 `sail::wechat_qrcode` 交付接口 | 约百行薄封装 |
| `python/wechat_qrcode_opencv.py` | Python | `cv2.imdecode` | 调用 `sail.wechat_qrcode` 交付接口 | 图像用 OpenCV 读入 ndarray |
| `python/wechat_qrcode_bmcv.py` | Python | `sail.Decoder`/`sail.BMImage` | 调用 `sail.wechat_qrcode` 交付接口 | 图像用 bmcv 硬件解码 |

四种实现共用同一套模型（detect `FP32` + sr `FP16`，均为 `fuse_preprocess` 的 uint8 灰度输入）与同一套测试口径（FPS 基准 + 文本/四角点精度）。同一平台上四种实现的精度在同一量级、但并不完全一致（图像加载路径不同 + `bmcv` 为纯移植实现），实测矩阵见 [5.2 精度测试结果](#52-测试结果)。

## 2. 特性

### 2.1 目录结构说明

```bash
├── cpp
│   ├── README.md  README_EN.md
│   ├── wechat_qrcode_bmcv            # 纯 bmrt+bmcv（不依赖 sail）的 C++ 例程，完整移植算法
│   │   ├── CMakeLists.txt
│   │   ├── main.cpp                  # 图像目录迭代 + bench 模式 + 结果 JSON 落盘
│   │   ├── wechat_qrcode.{hpp,cpp}   # detectAndDecode 编排（crop/padding/scale/dedup/remap）
│   │   ├── detector/ssd_detector.*   # detect 模型推理 + 19 shape 选网 + 输出解析
│   │   ├── scale/super_scale.*       # sr 超分推理 + 缩放回退
│   │   ├── binarizermgr.*  decodermgr.*  imgsource.*
│   │   ├── bm_util.hpp  bm_wrapper.hpp  infer.hpp  json.hpp  precomp.hpp
│   │   └── zxing/                    # vendored zxing（保留原 Apache-2.0 License）
│   └── wechat_qrcode_sail            # 调用 sail::wechat_qrcode 的薄封装例程
│       ├── CMakeLists.txt
│       ├── main.cpp
│       └── json.hpp
├── python
│   ├── README.md  README_EN.md
│   ├── requirements.txt
│   ├── wechat_qrcode_opencv.py       # OpenCV/cv2 读图 + sail.wechat_qrcode
│   └── wechat_qrcode_bmcv.py         # sail.Decoder/BMImage 读图 + sail.wechat_qrcode
├── docs
│   └── wechat_qrcode_process_guide.md  # 流程逐阶段解析
├── images                             # 测试图 qr_small.png / qr_big.png
├── scripts
│   ├── download.sh                    # 下载 3 芯片 bmodel 与 caffe 源模型
│   ├── download_datasets.sh           # 下载 BoofCV QR V4 评测集
│   ├── gen_bmodel.sh                  # caffe -> bmodel（detect 19 shape 合并 + sr）
│   └── auto_test.sh                   # 统一自动化测试（精度 + 性能回归）
└── tools
    ├── eval_qrcode.py                 # 解析 BoofCV 真值，输出检测 recall/precision + 解码准确率
    └── compare_statis.py              # 端到端 FPS 与基线回归对比
```

### 2.2 SDK、算法特性

* 支持 BM1684X(SoC/PCIe)、BM1688(SoC)、CV186X(SoC)
* 支持 FP32（detect）、FP16（sr）模型推理
* 支持 C++ 纯 bmrt+bmcv 移植（不依赖 sail）与基于 sail 交付接口的调用两种形态
* 支持 OpenCV 与 bmcv 两种图像加载路径（Python）
* 支持图片文件与目录（递归）批量推理，输出文本 + 四角点，并落盘结果 JSON
* detect 模型按 19 个静态 shape 子网合并（`detect_<W>_<H>`），推理时按宽高比自动选网，模拟 OpenCV 的比例缩放
* 同一测试口径：端到端/分段耗时（bench）+ BoofCV 检测/解码精度

## 3. 数据准备与模型编译

### 3.1 数据准备

本例程的评测集使用 [BoofCV QR Code V4](https://boofcv.org/notwiki/regression/fiducial/qrcodes_v4.zip)（约 250 MB），包含两段真值：

* `qrcodes/detection/<16 类>/imageNNN.{jpg,txt}`：检测子集，718 张图 / 1441 个二维码，真值为人工标注的四角点（`SETS` 格式），无文本与版本/ECC 信息。
* `qrcodes/decoding/<26 张 png>`：解码子集，干净合成码，`<name>.txt` 为期望文本。

> 来源与许可：dfss 上分发的是官方 `qrcodes_v4.zip` 的重打包副本（结构不变，解压得到 `BoofCV_qrcode_v4/`），原始下载地址 https://boofcv.org/notwiki/regression/fiducial/qrcodes_v4.zip 。BoofCV 主库为 Apache-2.0，数据仓库为 CC-BY-4.0（Peter Abeles）。

模型与数据集均可通过脚本下载：

```bash
chmod -R +x scripts/
./scripts/download.sh            # 下载 3 芯片 bmodel 与 caffe 源模型到 models/
./scripts/download_datasets.sh   # 下载评测集到 datasets/BoofCV_qrcode_v4/
```

下载的模型包括：

```bash
├── BM1684X
│   ├── detect_f32_fused.bmodel       # SSD-MobileNet 检测模型（FP32，19 shape 合并，uint8 灰度输入）
│   └── sr_f16_fused.bmodel           # 超分模型（FP16，224x224 -> 447x447，uint8 灰度输入）
├── BM1688
│   ├── detect_f32_fused.bmodel
│   └── sr_f16_fused.bmodel
└── CV186X
    ├── detect_f32_fused.bmodel
    └── sr_f16_fused.bmodel
```

此外 `download.sh` 还会下载 `models/opencv_3rdparty_wechat_qrcode/`（detect/sr 的 prototxt + caffemodel，供 `gen_bmodel.sh` 重新编译用，与 sophon-sail 打包一致）。

### 3.2 模型编译

detect/sr 的源模型（prototxt + caffemodel）来自 OpenCV 4.8.0 原生的 `wechat_qrcode`（`detect.prototxt`/`detect.caffemodel` 与 `sr.prototxt`/`sr.caffemodel`，上游仓库 [WeChatCV/opencv_3rdparty](https://github.com/WeChatCV/opencv_3rdparty)，Apache-2.0 许可）。这些源模型由 `download.sh` 一并下载到 `models/opencv_3rdparty_wechat_qrcode/`；在 tpu-mlir 环境（如 `lcx_mlir` 容器）中调用 `scripts/gen_bmodel.sh` 即可重新编译：

```bash
# 在 tpu-mlir 环境中（caffe 前端不支持动态 shape，因此 detect 用 19 静态 shape 合并模拟比例缩放）
./scripts/gen_bmodel.sh bm1684x    # 或 bm1688 / cv186x
```

脚本要点：

* detect：针对 19 个静态 shape（400x400 ... 1132x140 及其转置，面积 ≈160000、宽高比 1:1 ~ 8:1）逐个 `model_transform.py` + `model_deploy.py`，再用 `model_tool --combine` 合并为 `detect_f32_fused.bmodel`。
* sr：`224x224 -> 447x447` 的 SRResNet，F16 量化。
* 两个模型均用 `--fuse_preprocess`（mean=0、scale=0.0039216、pixel_format=gray），故推理输入为原始 uint8 单通道灰度图，归一化烘焙进 TPU 图内。

## 4. 例程测试

### 4.1 C++ 例程测试

C++ 例程可在 x86 PCIe 环境或交叉编译到 aarch64 SoC 后运行，交叉编译命令见 `cpp/README.md`（SoC 平台为各变体目录下 `cmake -DTARGET_ARCH=soc -DSDK=... [-DSAIL_PATH=...] .. && make`，产物为变体目录下的 `wechat_qrcode_{bmcv,sail}.soc`）。把产物与模型推送到盒子上后，运行方式（参数：`detect.bmodel sr.bmodel 输入路径 dev_id iters [core_id]`）：

```bash
# 单图/目录模式（iters=0）：打印每个二维码的 text 与四角点，落盘 results/wechat_qrcode_results.json
./cpp/wechat_qrcode_bmcv/wechat_qrcode_bmcv.soc models/BM1688/detect_f32_fused.bmodel models/BM1688/sr_f16_fused.bmodel images/qr_small.png 0 0
# bench 模式（iters>0）：预热 3 次后跑 N 次，打印 e2e / detect / sr / zxing 分段耗时
./cpp/wechat_qrcode_sail/wechat_qrcode_sail.soc models/BM1688/detect_f32_fused.bmodel models/BM1688/sr_f16_fused.bmodel images/qr_small.png 0 50
```

> `wechat_qrcode_sail` 链接完整的 `libsail.so`（内含 `sail::wechat_qrcode` 符号），运行前需保证 `LD_LIBRARY_PATH` 包含 libsail 所在目录；`wechat_qrcode_bmcv` 不依赖 sail，仅需 libsophon 的 bmlib/bmrt/bmcv。

### 4.2 Python 例程测试

Python 例程需要安装 pysail wheel（推盒时随部署脚本安装）、`numpy` 与 `opencv-python`（见 `python/requirements.txt`，opencv 变体需要）：

```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

运行方式：

```bash
# OpenCV 读图：cv2.imdecode 得到 ndarray 后交给 sail.wechat_qrcode
python3 python/wechat_qrcode_opencv.py --detect models/BM1688/detect_f32_fused.bmodel \
    --sr models/BM1688/sr_f16_fused.bmodel --input images/qr_small.png

# bmcv 读图：sail.Decoder -> sail.BMImage
python3 python/wechat_qrcode_bmcv.py --detect models/BM1688/detect_f32_fused.bmodel \
    --sr models/BM1688/sr_f16_fused.bmodel --input images/qr_small.png

# bench 模式
python3 python/wechat_qrcode_opencv.py --detect ... --sr ... --input images/qr_small.png --iters 50
```

任一实现单图/目录模式都会把结果 JSON 落盘到 `results/`（C++ 固定为 `results/wechat_qrcode_results.json`，Python 为 `results/{detect名}_{输入名}_{tag}_python_result.json`），供 `tools/eval_qrcode.py` 直接消费。

## 5. 精度测试

### 5.1 测试方法

在盒子上对评测集跑一遍目录模式（以 C++ bmcv 为例）：

```bash
./cpp/wechat_qrcode_bmcv/wechat_qrcode_bmcv.soc models/BM1688/detect_f32_fused.bmodel models/BM1688/sr_f16_fused.bmodel \
    datasets/BoofCV_qrcode_v4/qrcodes 0 0
```

然后用 `tools/eval_qrcode.py` 对比真值与结果：

```bash
python3 tools/eval_qrcode.py --gt_path datasets/BoofCV_qrcode_v4/qrcodes \
    --result_json results/wechat_qrcode_results.json
```

评测口径（详见 `tools/eval_qrcode.py`）：

* **detection 子集**：真值四角点与预测四角点按多边形 IoU 做贪心一对一匹配（`IoU >= 0.5` 判检出），统计 Recall = 匹配 / GT、Precision = 匹配 / 预测，并输出 16 类的逐类 recall。
* **decoding 子集**：解码文本与期望文本全等判对，输出解码准确率。

> 说明：wechat_qrcode 管线只上报 zxing 成功解码的二维码（检测 + 识别一体，无法解码时不上报候选框），因此 detection 子集上的“Recall”本质是“可识别的二维码召回”，对 damaged/blurred/perspective 等不可解码样本天然偏低，属正常现象。四种实现共用同一算法，同一平台精度在同一量级（实测见 [5.2 精度测试结果](#52-测试结果)）。
>
> 注（EXIF 朝向统一）：目录模式（精度读图）各程序都以不应用 EXIF 朝向的方式读图——C++ 两例程（`bmcv.soc`/`sail.soc`）用 sophon-opencv 的 `cv::imread(path, IMREAD_IGNORE_ORIENTATION)`（SoC 上即 JPU 硬解）、Python opencv 用 `cv2.imdecode(..., IMREAD_IGNORE_ORIENTATION)`、`bmcv.py` 用 `sail.Decoder`（本就不应用 EXIF）。原因是 OpenCV 4.1（BM1684X SDK）的 `cv::imread` 会自动应用 EXIF 旋转、而 4.8（BM1688/CV186X SDK）不会，两代 SDK 若不统一会导致同一批图检测出的四角点落在不同坐标系；统一后四角点始终落在与 BoofCV 真值一致的原始像素坐标系，跨平台结果可比。

### 5.2 测试结果

在 BoofCV QR Code V4 评测集（detection 718 图 / 1441 码，decoding 26 图）上，各测试平台、各测试程序的精度如下。测试模型统一为 `detect_f32_fused.bmodel`（FP32）+ `sr_f16_fused.bmodel`（FP16），并按平台分发对应芯片（BM1684X / BM1688 / CV186X）的 bmodel。

**detection 子集**（718 图 / 1441 个二维码，`IoU >= 0.5` 判检出）：

| 测试平台 | 测试程序 | 检测 Recall | 检测 Precision |
|---------|---------|------------|---------------|
| SE7-32 | wechat_qrcode_bmcv.soc    | 0.4754 (685/1441) | 1.0000 |
| SE7-32 | wechat_qrcode_sail.soc    | 0.4768 (687/1441) | 1.0000 |
| SE7-32 | wechat_qrcode_opencv.py   | 0.4802 (692/1441) | 1.0000 |
| SE7-32 | wechat_qrcode_bmcv.py     | 0.4774 (688/1441) | 0.9843 (688/699) |
| SE9-8  | wechat_qrcode_bmcv.soc    | 0.4740 (683/1441) | 1.0000 |
| SE9-8  | wechat_qrcode_sail.soc    | 0.4768 (687/1441) | 1.0000 |
| SE9-8  | wechat_qrcode_opencv.py   | 0.4837 (697/1441) | 1.0000 |
| SE9-8  | wechat_qrcode_bmcv.py     | 0.4879 (703/1441) | 0.9986 (703/704) |
| SE9-16 | wechat_qrcode_bmcv.soc    | 0.4740 (683/1441) | 1.0000 |
| SE9-16 | wechat_qrcode_sail.soc    | 0.4768 (687/1441) | 1.0000 |
| SE9-16 | wechat_qrcode_opencv.py   | 0.4837 (697/1441) | 1.0000 |
| SE9-16 | wechat_qrcode_bmcv.py     | 0.4879 (703/1441) | 0.9986 (703/704) |

> 参考基线（同一评测集、CPU 原生推理）：OpenCV 4.8.0 原生 `wechat_qrcode`（`opencv-contrib-python 4.8.0.76`，detect/sr caffe 与本示例同源）Recall 0.4559 / Precision 0.9880；OpenCV 5.0.0 原生（`opencv-contrib-python 5.0.0.93`，改用 ONNX 与全新管线，非同源模型）Recall 0.3491 / Precision 0.9843。本示例即把 4.8.0 的 detect/sr caffe 同源模型转 bmodel 上 TPU，故 TPU 各行 recall（0.4740–0.4879）与 4.8.0 CPU（0.4559）相近且略高，precision 基本为 1.0000。
>
> 说明：同一平台上四种实现的精度在同一量级、但并不完全一致，差异来自两点——（1）图像加载路径的 JPEG 解码器不同：`bmcv.soc` 与 `sail.soc` 用 sophon-opencv 的 `cv::imread`（SoC 上即 JPU 硬解，两者读图与主机算法路径完全相同），`opencv.py` 用 pip opencv 的 `cv2.imdecode`（libjpeg 软解），`bmcv.py` 用 `sail.Decoder` 的 ffmpeg 硬件解码，逐像素差异使 detect 在个别难样本上检出不同（`sail.soc` 与 `opencv.py` 的 5 码差即源于此）；（2）`bmcv.soc` 是纯 bmrt+bmcv 的算法重移植——它与 `sail.soc` 读图与主机算法路径一致，但宿主路径的个别实现细节（灰度转换、F16→uint8 取整等）存在 2 码以内的临界样本差异。此外 `bmcv.py` 的 ffmpeg 解码 + 设备 VPP（`LINEAR`，而非主机路径的 `INTER_CUBIC`）路径会在个别图片上产生 `IoU<0.5` 的误检框，precision 略低于 1.0000（SE7 上 688/699、SE9 上 703/704）。

逐类 recall（16 类）详见 `eval_qrcode.py` 运行输出，其中 `nominal`（0.94 / 0.92）、`pathological`（0.92）、`shadows`（0.92）等清晰样本 recall ≥ 0.92，`noncompliant`（0.86 / 0.88）等次之，`lots`（单图数十个码，0.02）与 `high_version`（0.23）等难例 recall 最低，拉低了整体值。

**decoding 子集**（26 张合成码）：

| 测试平台 | 测试程序 | 解码准确率 |
|---------|---------|-----------|
| SE7-32  | wechat_qrcode_bmcv.soc    | 26 / 26 = 1.0000 |
| SE7-32  | wechat_qrcode_sail.soc    | 26 / 26 = 1.0000 |
| SE7-32  | wechat_qrcode_opencv.py   | 26 / 26 = 1.0000 |
| SE7-32  | wechat_qrcode_bmcv.py     | 26 / 26 = 1.0000 |
| SE9-8   | wechat_qrcode_bmcv.soc    | 26 / 26 = 1.0000 |
| SE9-8   | wechat_qrcode_sail.soc    | 26 / 26 = 1.0000 |
| SE9-8   | wechat_qrcode_opencv.py   | 26 / 26 = 1.0000 |
| SE9-8   | wechat_qrcode_bmcv.py     | 26 / 26 = 1.0000 |
| SE9-16  | wechat_qrcode_bmcv.soc    | 26 / 26 = 1.0000 |
| SE9-16  | wechat_qrcode_sail.soc    | 26 / 26 = 1.0000 |
| SE9-16  | wechat_qrcode_opencv.py   | 26 / 26 = 1.0000 |
| SE9-16  | wechat_qrcode_bmcv.py     | 26 / 26 = 1.0000 |

> vCard/vEvent 的 CRLF 行尾已归一化后比较，见 `eval_qrcode.py`。

## 6. 性能测试

### 6.1 bmrt_test

用 `bmrt_test` 测得两个模型的 TPU 纯推理耗时（单 shape，不含前后处理）：

|   测试平台    |          测试模型              |      输入 shape      | calculate time(ms) |
|---------------|--------------------------------|----------------------|--------------------|
| BM1684X       | detect_f32_fused.bmodel        | `[1,1,400,400]`      | 1.64               |
| BM1684X       | sr_f16_fused.bmodel            | `[1,1,224,224]`      | 1.13               |
| BM1688        | detect_f32_fused.bmodel        | `[1,1,400,400]`      | 2.61               |
| BM1688        | sr_f16_fused.bmodel            | `[1,1,224,224]`      | 2.83               |
| CV186X        | detect_f32_fused.bmodel        | `[1,1,400,400]`      | 2.61               |
| CV186X        | sr_f16_fused.bmodel            | `[1,1,224,224]`      | 2.83               |

### 6.2 程序运行性能

端到端解码性能（`images/qr_small.png`，198x198，detect 一次 TPU 推理 + zxing 解码，iters=50 预热 3 次）：

| 平台  | C++ bmcv  | C++ sail  | Python opencv | Python bmcv |
|-------|-----------|-----------|---------------|-------------|
| SE7-32 | 204.9 fps | 205.3 fps | 171.2 fps     | 198.5 fps   |
| SE9-8  | 113.2 fps | 113.3 fps | 106.0 fps     | 111.1 fps   |
| SE9-16 | 110.8 fps | 114.2 fps | 107.3 fps     | 111.1 fps   |

> 平台代号（与 PP-OCR 等例程口径一致）：SE7-32 对应 BM1684X，SE9-16 对应 BM1688，SE9-8 对应 CV186X。基准值由 `tools/compare_statis.py` 维护（SE7-32≈205 / SE9-8≈113 / SE9-16≈114），四种实现共用同一 detect/sr/zxing 管线，端到端 FPS 应在同一量级。
>
> 注：测试图较小（198x198，未触发超分 sr），detect 单次 TPU 推理约占端到端 40%，zxing 解码（CPU）占比最高；图像越大 / 触发超分时端到端 FPS 越低。
>
> 注：`C++ bmcv` / `C++ sail` / `Python bmcv` 三者走设备直传路径，端到端速度接近——图像先经硬件解码到设备显存（`C++ bmcv` 用 sophon-opencv 的三参 `cv::imread(path, flags, dev_id)`；`C++ sail` 默认 `sail::Decoder`（`USE_OPENCV_DECODE=0`，编译时加 `-DUSE_OPENCV_DECODE=1` 可回退到三参 `cv::imread`）；`Python bmcv` 用 `sail.Decoder`），再经 `toBMI` / `sail::Bmcv::mat_to_bm_image` 零拷贝转为 `bm_image`，`detectAndDecode(bm_image)` 在设备上完成 VPP 缩放 + 推理，全流程无「主机→设备」整帧拷贝；而 `Python opencv` 用 `cv2.imdecode` 先读到主机 ndarray，每次迭代要把主机数据做一次 BGR→GRAY + H→D DMA 拷贝，detect 阶段约多 0.8ms（本例 SE7 上 detect 2.03ms vs 其余三者的约 1.2ms）。四种实现的图像解码均在计时循环外完成，口径一致。

整数据集端到端解码耗时（BoofCV QR Code V4 全量 744 张：detection 718 + decoding 26，共 1441 个二维码；目录模式一次性跑完，计时为进程启动至 `results/wechat_qrcode_results.json` 落盘的墙钟时间）：

| 平台    | 图片数 / 二维码数 | 端到端总耗时          | 平均单图耗时 |
|---------|------------------|----------------------|-------------|
| SE7-32  | 744 / 1441       | 1218 s（约 20 分 18 秒） | 1.64 s      |
| SE9-8   | 744 / 1441       | 1707 s（约 28 分 27 秒） | 2.29 s      |
| SE9-16  | 744 / 1441       | 1693 s（约 28 分 13 秒） | 2.28 s      |

> 注：整集耗时的主导项是 CPU 上的 zxing 解码——v4 集中存在大量高密度 / 高版本 / 反光 / 损坏二维码（`close` / `high_version` / `glare` / `damaged` 等类），单张可到 2–10 s，远大于 TPU 上毫秒级的 detect/sr。因此整集墙钟主要反映 zxing 解码速度，平台间差异主要来自 CPU 算力：SE7-32 明显更快，SE9-8 与 SE9-16 相当。

## 7. FAQ

* **为什么 C++ 有 bmcv 和 sail 两个例程？** `bmcv` 是纯 bmrt + bmcv 的完整算法移植（不依赖 sail，作为源码参考与最小依赖交付）；`sail` 是约百行的薄封装，直接调用 `sophon-sail` 里已交付的 `sail::wechat_qrcode` 接口。两者模型与性能一致，精度在同一平台上基本一致（实测见 [5.2 精度测试结果](#52-测试结果)），可按需选择。
* **为什么 detect 模型有 19 个网络？** tpu-mlir 的 caffe 前端不支持动态 shape，OpenCV 原算法按 `s=min(1, sqrt(160000/(w*h)))` 比例缩放输入。此处用 19 个覆盖常见宽高比的静态 shape 合并成一个 bmodel，推理时按输入宽高比自动选网，等价于 OpenCV 的比例缩放。
* **何时触发超分 sr？** 裁剪小图满足 `sqrt(w*h) < 160` 时才走 sr 超分，否则用 `INTER_CUBIC`/`INTER_AREA` 在 CPU 上缩放。小图不触发超分时可看到 bench 里 `sr : 0.00 ms`。
* **为什么检测子集 recall 偏低？** 见 [5. 精度测试](#5-精度测试) 的说明——管线只上报成功解码的二维码。
* **为什么需要注意 `LD_LIBRARY_PATH`？** `wechat_qrcode_sail` 依赖 `libsail.so`，需保证其所在目录在 `LD_LIBRARY_PATH` 中；测试环境里 sail 不一定预装。`wechat_qrcode_bmcv` 不依赖 sail，可避免此问题。
* **和同一设备纯 CPU 的 OpenCV wechat_qrcode 相比，性能如何？** 本例只把 detect/sr 两个神经网络移到 TPU，zxing 解码仍在 CPU 上与 OpenCV 原版一致。以同板卡（A53）上 `opencv-contrib-python 4.8.0` 的 `wechat_qrcode_WeChatQRCode`（cv2.dnn CPU 后端）为基线的同机实测，端到端加速约 **2.2–4.2×**：小图（198×198）约 2.2×，码小图大的画布（1280×720 / 2560×360）约 4.2×。分段看，detect 快约 **7–11×**、sr 快约 **51×**；但 zxing 解码两端同在 CPU（大图单张约 28 ms），是端到端主要瓶颈、TPU 无法加速，因此「码铺满整图」类的收益收敛到约 1.3–1.7×。