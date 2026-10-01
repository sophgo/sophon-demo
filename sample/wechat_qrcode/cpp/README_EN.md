[简体中文](./README.md) | [English](./README_EN.md)

# C++ Example

## Catalogue

- [C++ Example](#c-example)
  - [Catalogue](#catalogue)
  - [1. Environment Preparation](#1-environment-preparation)
    - [1.1 x86/arm/riscv PCIe](#11-x86armriscv-pcie)
    - [1.2 SoC](#12-soc)
  - [2. Compilation](#2-compilation)
    - [2.1 x86/arm/riscv PCIe](#21-x86armriscv-pcie)
    - [2.2 SoC](#22-soc)
  - [3. Inference Test](#3-inference-test)
    - [3.1 Arguments](#31-arguments)
    - [3.2 Test Images](#32-test-images)

Two C++ examples are provided under `cpp`:

| # | Example            | Description                                                       |
| - | ------------------ | ----------------------------------------------------------------- |
| 1 | wechat_qrcode_bmcv | Full pure bmrt + bmcv algorithm port (no sail), vendored zxing     |
| 2 | wechat_qrcode_sail | Thin wrapper calling `sail::wechat_qrcode` (depends on libsail.so) |

They share the same detect/sr models and algorithm, emit text + four corner points, and have identical performance; accuracy is essentially identical on the same platform (within a few codes, see the measured matrix in the main [`README.md#5.2 Test Result`](../README.md#52-test-result)).

## 1. Environment Preparation

### 1.1 x86/arm/riscv PCIe

If you installed a PCIe accelerator card (e.g. the SC series) on an x86/arm/riscv platform, you can use it directly as both the development and runtime environment. Install libsophon, sophon-opencv (and sophon-sail for `wechat_qrcode_sail`). See [x86-pcie environment setup](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建) or [arm-pcie environment setup](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建) or [riscv-pcie environment setup](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建).

### 1.2 SoC

If you use an SoC platform (e.g. the SE/SM series edge devices), libsophon and sophon-opencv runtime packages are pre-installed under `/opt/sophon/` after flashing. You usually still need an x86 host as the development environment for cross-compilation.

## 2. Compilation

The C++ programs must be compiled before running.

### 2.1 x86/arm/riscv PCIe

Compile directly on the PCIe platform:

```bash
# bmcv example (no sail)
cd cpp/wechat_qrcode_bmcv
mkdir build && cd build
cmake ..
make
cd ..

# sail example (needs /opt/sophon/sophon-sail)
cd cpp/wechat_qrcode_sail
mkdir build && cd build
cmake ..
make
cd ..
```

`wechat_qrcode_bmcv.pcie` / `wechat_qrcode_sail.pcie` are generated in the respective directories.

### 2.2 SoC

Usually cross-compiled on an x86 host (set up the cross environment first, see [cross-compilation setup](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)):

```bash
# bmcv example
cd cpp/wechat_qrcode_bmcv
mkdir build && cd build
# adjust -DSDK to an absolute path as needed
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make

# sail example (additionally -DSAIL_PATH pointing at libsail.so)
cd ../../wechat_qrcode_sail
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk -DSAIL_PATH=/path_to_sail ..
make
```

`wechat_qrcode_bmcv.soc` / `wechat_qrcode_sail.soc` are generated in the respective directories.

> **Note (sail decode path):** `wechat_qrcode_sail` defaults to `USE_OPENCV_DECODE=0`, i.e. the bench (device decode) path uses `sail::Decoder` (ffmpeg hardware decode). Add `-DUSE_OPENCV_DECODE=1` to the cmake command above to switch the bench read to sophon-opencv's 3-arg `cv::imread()` (VPU decode to device) instead. This only affects the bench read path; directory-mode (accuracy) image loading still uses host `cv::imread`, regardless of this flag.

## 3. Inference Test

On PCIe you can test directly; on SoC copy the executable, models and test data to the board first. The arguments and run commands are identical; the SoC mode is used below.

> **Note:** `wechat_qrcode_sail` depends on the full `libsail.so` (carrying `sail::wechat_qrcode`); make sure `LD_LIBRARY_PATH` includes its directory. `wechat_qrcode_bmcv` has no such dependency.

### 3.1 Arguments

Both examples use positional arguments (unlike Python, no `--key=value`):

```bash
./cpp/wechat_qrcode_bmcv/wechat_qrcode_bmcv.soc detect.bmodel sr.bmodel input_path dev_id iters
./cpp/wechat_qrcode_sail/wechat_qrcode_sail.soc detect.bmodel sr.bmodel input_path dev_id iters [core_id]
```

| Argument      | Description                                                          |
| ------------- | -------------------------------------------------------------------- |
| detect.bmodel | detect model path (default `../models/BM1684X/detect_f32_fused.bmodel`) |
| sr.bmodel     | sr model path (default `../models/BM1684X/sr_f16_fused.bmodel`)          |
| input_path    | input image path or directory (default `../images/qr_small.png`)          |
| dev_id        | TPU device id (default 0)                                             |
| iters         | >0 runs bench mode with that many iterations; <=0 single/dir mode (default 0) |
| core_id       | BM1688 dual-core NPU pin (0/1), -1 auto (default -1; sail example only) |

`iters <= 0` (single-file/directory mode): prints each QR's `text` and four corners per image, and dumps a result JSON to `results/wechat_qrcode_results.json` (directly consumable by `../tools/eval_qrcode.py`).

`iters > 0` (bench mode): 3 warmups then `iters` iterations, printing end-to-end and detect/sr/zxing timings (`ms/img` and `fps`).

### 3.2 Test Images

Examples (a whole directory is supported, scanned recursively):

```bash
# single image
./cpp/wechat_qrcode_bmcv/wechat_qrcode_bmcv.soc models/BM1688/detect_f32_fused.bmodel models/BM1688/sr_f16_fused.bmodel images/qr_small.png 0 0

# directory mode (recursively scans all jpg/png/bmp)
./cpp/wechat_qrcode_bmcv/wechat_qrcode_bmcv.soc models/BM1688/detect_f32_fused.bmodel models/BM1688/sr_f16_fused.bmodel datasets/BoofCV_qrcode_v4/qrcodes 0 0

# bench mode
./cpp/wechat_qrcode_sail/wechat_qrcode_sail.soc models/BM1688/detect_f32_fused.bmodel models/BM1688/sr_f16_fused.bmodel images/qr_small.png 0 50
```

Expected output for `images/qr_small.png`:

```text
[images/qr_small.png]
  decoded 1 code(s)
  [0] text=sail wechat qrcode small
     corners(4x2 float32): (24.0,24.0) (174.0,24.0) (174.0,174.0) (24.0,174.0)
```

The result JSON is saved under `results/`.