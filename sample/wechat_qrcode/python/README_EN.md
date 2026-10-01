[简体中文](./README.md) | [English](./README_EN.md)

# Python Example

## Catalogue

- [Python Example](#python-example)
  - [Catalogue](#catalogue)
  - [1. Environment Preparation](#1-environment-preparation)
    - [1.1 x86/arm/riscv PCIe](#11-x86armriscv-pcie)
    - [1.2 SoC](#12-soc)
  - [2. Inference Test](#2-inference-test)
    - [2.1 Arguments](#21-arguments)
    - [2.2 Test Images](#22-test-images)

Two Python examples are provided under `python`:

| # | Example                  | Description                                              | Image loading         |
| - | ------------------------ | -------------------------------------------------------- | --------------------- |
| 1 | wechat_qrcode_opencv.py  | OpenCV reading (`cv2.imdecode`) -> `sail.wechat_qrcode`  | CPU decode to ndarray |
| 2 | wechat_qrcode_bmcv.py    | bmcv reading (`sail.Decoder`/`sail.BMImage`) -> `sail.wechat_qrcode` | hardware decode to BMImage |

Both call the `sail.wechat_qrcode.WeChatQRCode` API shipped by `sophon-sail`, differing only in the image-loading path (`cv2.imdecode` vs `sail.Decoder`); they emit text + four corner points. Accuracy on the same platform differs measurably because of the two decoders (see the main [`README.md#5.2 Test Result`](../README.md#52-test-result)); on performance, `bmcv.py` offloads decoding and image transfer to TPU/VPU via `sail.Decoder`, while `opencv.py` decodes on the CPU and adds one host-to-device copy, so `opencv.py` is slightly slower (see main README §6.2 — both decode outside the timed loop, so the numbers are comparable).

## 1. Environment Preparation

### 1.1 x86/arm/riscv PCIe

If you installed a PCIe accelerator card (e.g. the SC series) on an x86/arm/riscv platform, install libsophon, sophon-opencv, sophon-ffmpeg and sophon-sail. See [x86-pcie environment setup](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建) or [arm-pcie environment setup](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建) or [riscv-pcie environment setup](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建).

Install the required Python packages:

```bash
pip3 install -r requirements.txt
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade  # if you need to download models/datasets
```

### 1.2 SoC

If you use an SoC platform (e.g. the SE/SM series edge devices), libsophon, sophon-opencv and sophon-ffmpeg are pre-installed under `/opt/sophon/` after flashing. You also need the pysail wheel (`sophon_arm-*.whl`, pushed with the deployment script):

```bash
pip3 install --user sophon_arm-*.whl
pip3 install -r requirements.txt
```

> **Note:** `opencv-python` in `requirements.txt` is only for `cv2.imdecode` in `wechat_qrcode_opencv.py`; `wechat_qrcode_bmcv.py` uses `sail.Decoder` and does not need it.

## 2. Inference Test

The Python examples need no compilation and can be run directly; the arguments and commands are identical on PCIe and SoC.

### 2.1 Arguments

Both examples share the same arguments (`--key value`):

```bash
usage: wechat_qrcode_opencv.py [-h] [--detect DETECT] [--sr SR] [--input INPUT]
                               [--dev_id DEV_ID] [--core_id CORE_ID] [--iters ITERS]

  --detect DETECT   detect model path (default ../models/BM1684X/detect_f32_fused.bmodel)
  --sr SR           sr model path (default ../models/BM1684X/sr_f16_fused.bmodel)
  --input INPUT     input image path or directory (default ../images/qr_small.png)
  --dev_id DEV_ID   TPU device id (default 0)
  --core_id CORE_ID BM1688 dual-core pin 0/1, -1 auto (default -1)
  --iters ITERS     >0 runs bench mode with that many iterations; <=0 single/dir (default 0)
```

`--iters <= 0` (single/directory mode): prints each QR's `text` and four corners per image, and dumps a result JSON to `results/{detect}_{input}_{tag}_python_result.json` (directly consumable by `../tools/eval_qrcode.py`).

`--iters > 0` (bench mode): 3 warmups then `iters` iterations, printing end-to-end and detect/sr/zxing timings.

### 2.2 Test Images

Examples (a directory is supported, scanned recursively):

```bash
# OpenCV reading
python3 wechat_qrcode_opencv.py --detect ../models/BM1688/detect_f32_fused.bmodel \
    --sr ../models/BM1688/sr_f16_fused.bmodel --input ../images/qr_small.png

# bmcv reading
python3 wechat_qrcode_bmcv.py --detect ../models/BM1688/detect_f32_fused.bmodel \
    --sr ../models/BM1688/sr_f16_fused.bmodel --input ../images/qr_small.png

# directory mode (evaluation set)
python3 wechat_qrcode_opencv.py --detect ../models/BM1688/detect_f32_fused.bmodel \
    --sr ../models/BM1688/sr_f16_fused.bmodel --input ../datasets/BoofCV_qrcode_v4/qrcodes

# bench mode
python3 wechat_qrcode_opencv.py --detect ../models/BM1688/detect_f32_fused.bmodel \
    --sr ../models/BM1688/sr_f16_fused.bmodel --input ../images/qr_small.png --iters 50
```

Expected output for `../images/qr_small.png`:

```text
INFO: [../images/qr_small.png] decoded 1 code(s)
INFO:   [0] text=sail wechat qrcode small corners=(24.0,24.0) (174.0,24.0) (174.0,174.0) (24.0,174.0)
INFO: result saved in results/detect_f32_fused_qr_small_opencv_python_result.json (1/1 images decoded)
```

The result JSON is saved under `results/`.