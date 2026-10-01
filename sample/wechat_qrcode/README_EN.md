[简体中文](./README.md) | [English](./README_EN.md)

# WeChatQRCode

## Catalogue

- [WeChatQRCode](#wechatqrcode)
  - [Catalogue](#catalogue)
  - [1. Introduction](#1-introduction)
  - [2. Characteristics](#2-characteristics)
    - [2.1 Directory Instructions](#21-directory-instructions)
    - [2.2 SDK and Algorithm Characteristics](#22-sdk-and-algorithm-characteristics)
  - [3. Data preparation and model compilation](#3-data-preparation-and-model-compilation)
    - [3.1 Data preparation](#31-data-preparation)
    - [3.2 Model Compilation](#32-model-compilation)
  - [4. Example Test](#4-example-test)
    - [4.1 C++ Example Test](#41-c-example-test)
    - [4.2 Python Example Test](#42-python-example-test)
  - [5. Precision Test](#5-precision-test)
    - [5.1 Testing Method](#51-testing-method)
    - [5.2 Test Result](#52-test-result)
  - [6. Performance Testing](#6-performance-testing)
    - [6.1 bmrt_test](#61-bmrt_test)
    - [6.2 Program Performance](#62-program-performance)
  - [7. FAQ](#7-faq)

## 1. Introduction

WeChatQRCode is the WeChat QR-code **detection + recognition** algorithm built into OpenCV since 4.8.0. Its full pipeline is: SSD-MobileNet object detection (detect, localising QR codes) -> crop around each box with padding -> super-resolution of the small crops (sr, SRResNet) plus multi-scale scaling -> zxing decoding (CPU) -> corner remap and dedup, finally emitting the text and the four corner points of every QR code. This example ports the native OpenCV wechat_qrcode model and algorithm so that it can run on SOPHON BM1684X/BM1688/CV186X, with the detect and sr models running on the TPU and the zxing decoding on the CPU.

Four implementations are provided to fit different usage habits:

| Implementation | Language | Image loading | Algorithm source | Notes |
|----------------|----------|---------------|------------------|-------|
| `cpp/wechat_qrcode_bmcv` | C++ | sophon-opencv `cv::imread` | Source port (pure bmrt + bmcv, no sail) | Full detect/crop/sr/zxing pipeline, vendored zxing |
| `cpp/wechat_qrcode_sail` | C++ | `sail::Decoder` (bench) / sophon-opencv `cv::imread` (folder) | Calls the `sail::wechat_qrcode` API | ~100-line thin wrapper |
| `python/wechat_qrcode_opencv.py` | Python | `cv2.imdecode` | Calls the `sail.wechat_qrcode` API | Image read into an ndarray with OpenCV |
| `python/wechat_qrcode_bmcv.py` | Python | `sail.Decoder`/`sail.BMImage` | Calls the `sail.wechat_qrcode` API | Image decoded by the bmcv hardware path |

The four implementations share the same models (detect `FP32` + sr `FP16`, both fused-preprocess uint8 gray input) and the same test methodology (FPS benchmark + text/4-corner accuracy). On the same platform their accuracy is in the same ballpark but not identical (different image-loading paths + `bmcv` being a port), see the measured matrix in [5.2 Precision Test](#52-test-result).

## 2. Characteristics

### 2.1 Directory Instructions

```bash
├── cpp
│   ├── README.md  README_EN.md
│   ├── wechat_qrcode_bmcv            # Pure bmrt+bmcv (no sail) C++ example, full algorithm port
│   │   ├── CMakeLists.txt
│   │   ├── main.cpp                  # Directory iteration + bench mode + result JSON dump
│   │   ├── wechat_qrcode.{hpp,cpp}   # detectAndDecode orchestration (crop/padding/scale/dedup/remap)
│   │   ├── detector/ssd_detector.*   # detect inference + 19-shape selection + output parsing
│   │   ├── scale/super_scale.*       # sr super-resolution inference + resize fallback
│   │   ├── binarizermgr.*  decodermgr.*  imgsource.*
│   │   ├── bm_util.hpp  bm_wrapper.hpp  infer.hpp  json.hpp  precomp.hpp
│   │   └── zxing/                    # vendored zxing (original Apache-2.0 License kept)
│   └── wechat_qrcode_sail            # Thin wrapper calling sail::wechat_qrcode
│       ├── CMakeLists.txt
│       ├── main.cpp
│       └── json.hpp
├── python
│   ├── README.md  README_EN.md
│   ├── requirements.txt
│   ├── wechat_qrcode_opencv.py       # OpenCV/cv2 image reading + sail.wechat_qrcode
│   └── wechat_qrcode_bmcv.py         # sail.Decoder/BMImage reading + sail.wechat_qrcode
├── docs
│   └── wechat_qrcode_process_guide.md  # stage-by-stage pipeline walkthrough
├── images                             # test images qr_small.png / qr_big.png
├── scripts
│   ├── download.sh                    # download bmodels for 3 chips + caffe source models
│   ├── download_datasets.sh           # download the BoofCV QR V4 evaluation set
│   ├── gen_bmodel.sh                  # caffe -> bmodel (detect 19-shape combine + sr)
│   └── auto_test.sh                   # unified auto test (accuracy + performance regression)
└── tools
    ├── eval_qrcode.py                 # parse BoofCV ground truth -> recall/precision + decode accuracy
    └── compare_statis.py              # end-to-end FPS regression against a baseline
```

### 2.2 SDK and Algorithm Characteristics

* Supports BM1684X (SoC/PCIe), BM1688 (SoC), CV186X (SoC)
* Supports FP32 (detect) and FP16 (sr) model inference
* Provides both a pure bmrt+bmcv source port (no sail) and a thin wrapper around the sail delivery API
* Supports both OpenCV and bmcv image loading paths (Python)
* Supports single-file and recursive-directory batch inference, emitting text + 4 corner points and a result JSON
* The detect model merges 19 static-shape sub-networks (`detect_<W>_<H>`); a network is chosen by aspect ratio at runtime, mimicking OpenCV's proportional rescale
* One test methodology: end-to-end / per-stage timing (bench) + BoofCV detection/decoding accuracy

## 3. Data preparation and model compilation

### 3.1 Data preparation

The evaluation set used here is [BoofCV QR Code V4](https://boofcv.org/notwiki/regression/fiducial/qrcodes_v4.zip) (~250 MB), with two kinds of ground truth:

* `qrcodes/detection/<16 categories>/imageNNN.{jpg,txt}`: detection subset, 718 images / 1441 QR codes, ground truth is hand-annotated corner sets (`SETS` format), no text or version/ECC info.
* `qrcodes/decoding/<26 png>`: decoding subset, clean synthetic codes; `<name>.txt` is the expected text.

> Source & license: the dfss tarball is a repackaged copy of the official `qrcodes_v4.zip` (unchanged layout, unpacked under `BoofCV_qrcode_v4/`); the original download URL is https://boofcv.org/notwiki/regression/fiducial/qrcodes_v4.zip . The BoofCV main library is Apache-2.0 and the data repository is CC-BY-4.0 (Peter Abeles).

Both models and dataset can be downloaded with the provided scripts:

```bash
chmod -R +x scripts/
./scripts/download.sh            # download the 3-chip bmodels + caffe source models into models/
./scripts/download_datasets.sh   # download the eval set into datasets/BoofCV_qrcode_v4/
```

The downloaded models are:

```bash
├── BM1684X
│   ├── detect_f32_fused.bmodel       # SSD-MobileNet detection model (FP32, 19-shape combine, uint8 gray input)
│   └── sr_f16_fused.bmodel           # super-resolution model (FP16, 224x224 -> 447x447, uint8 gray input)
├── BM1688
│   ├── detect_f32_fused.bmodel
│   └── sr_f16_fused.bmodel
└── CV186X
    ├── detect_f32_fused.bmodel
    └── sr_f16_fused.bmodel
```

`download.sh` also fetches `models/opencv_3rdparty_wechat_qrcode/` (the detect/sr prototxt + caffemodel, used by `gen_bmodel.sh` to regenerate the bmodels; the same tarball sophon-sail ships).

### 3.2 Model Compilation

The detect/sr source models (prototxt + caffemodel) come from OpenCV 4.8.0's native `wechat_qrcode` (`detect.prototxt`/`detect.caffemodel` and `sr.prototxt`/`sr.caffemodel`; upstream repo [WeChatCV/opencv_3rdparty](https://github.com/WeChatCV/opencv_3rdparty), Apache-2.0). `download.sh` fetches them into `models/opencv_3rdparty_wechat_qrcode/`; compile them inside a tpu-mlir environment (e.g. the `lcx_mlir` container) with `scripts/gen_bmodel.sh`:

```bash
# inside the tpu-mlir environment (the caffe frontend has no dynamic shape, so
# detect is a 19-shape combine that mimics proportional rescale)
./scripts/gen_bmodel.sh bm1684x    # or bm1688 / cv186x
```

Key points of the script:

* detect: for each of the 19 static shapes (400x400 ... 1132x140 and their transposes, area ~160000, aspect ratios 1:1 to 8:1) run `model_transform.py` + `model_deploy.py`, then `model_tool --combine` them into `detect_f32_fused.bmodel`.
* sr: SRResNet `224x224 -> 447x447`, F16 quantization.
* Both models use `--fuse_preprocess` (mean=0, scale=0.0039216, pixel_format=gray), so the inference input is a raw uint8 single-channel gray image with normalization baked into the TPU graph.

## 4. Example Test

### 4.1 C++ Example Test

The C++ examples can be built for x86 PCIe or cross-compiled for aarch64 SoC; see `cpp/README.md` for the cmake commands (SoC: `cmake -DTARGET_ARCH=soc -DSDK=... [-DSAIL_PATH=...] .. && make` in each variant dir, producing `wechat_qrcode_{bmcv,sail}.soc` there). After pushing the binaries and models to the board, run (args: `detect.bmodel sr.bmodel input dev_id iters [core_id]`):

```bash
# single-file / directory mode (iters=0): print text + 4 corners per QR, dump results/wechat_qrcode_results.json
./cpp/wechat_qrcode_bmcv/wechat_qrcode_bmcv.soc models/BM1688/detect_f32_fused.bmodel models/BM1688/sr_f16_fused.bmodel images/qr_small.png 0 0
# bench mode (iters>0): 3 warmups then N iterations, print e2e / detect / sr / zxing timings
./cpp/wechat_qrcode_sail/wechat_qrcode_sail.soc models/BM1688/detect_f32_fused.bmodel models/BM1688/sr_f16_fused.bmodel images/qr_small.png 0 50
```

> `wechat_qrcode_sail` links the full `libsail.so` (which carries `sail::wechat_qrcode`), so `LD_LIBRARY_PATH` must include its directory; `wechat_qrcode_bmcv` does not depend on sail and only needs libsophon's bmlib/bmrt/bmcv.

### 4.2 Python Example Test

The Python examples need the pysail wheel (installed alongside deployment), `numpy`, and `opencv-python` (the opencv variant only — see `python/requirements.txt`):

```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Run:

```bash
# OpenCV reading: cv2.imdecode -> ndarray -> sail.wechat_qrcode
python3 python/wechat_qrcode_opencv.py --detect models/BM1688/detect_f32_fused.bmodel \
    --sr models/BM1688/sr_f16_fused.bmodel --input images/qr_small.png

# bmcv reading: sail.Decoder -> sail.BMImage
python3 python/wechat_qrcode_bmcv.py --detect models/BM1688/detect_f32_fused.bmodel \
    --sr models/BM1688/sr_f16_fused.bmodel --input images/qr_small.png

# bench mode
python3 python/wechat_qrcode_opencv.py --detect ... --sr ... --input images/qr_small.png --iters 50
```

Every implementation, in single-image/directory mode, dumps a result JSON under `results/` (C++ writes `results/wechat_qrcode_results.json`; Python writes `results/{detect}_{input}_{tag}_python_result.json`) that `tools/eval_qrcode.py` consumes directly.

## 5. Precision Test

### 5.1 Testing Method

Run directory mode over the eval set on the board (C++ bmcv as an example):

```bash
./cpp/wechat_qrcode_bmcv/wechat_qrcode_bmcv.soc models/BM1688/detect_f32_fused.bmodel models/BM1688/sr_f16_fused.bmodel \
    datasets/BoofCV_qrcode_v4/qrcodes 0 0
```

Then compare against ground truth with `tools/eval_qrcode.py`:

```bash
python3 tools/eval_qrcode.py --gt_path datasets/BoofCV_qrcode_v4/qrcodes \
    --result_json results/wechat_qrcode_results.json
```

Metrics (see `tools/eval_qrcode.py`):

* **detection subset**: ground-truth vs predicted corners matched greedily one-to-one by polygon IoU (a code counts as detected when `IoU >= 0.5`); Recall = matched / GT, Precision = matched / predicted, plus per-category recall over the 16 categories.
* **decoding subset**: decoded text must equal the expected text (line endings normalized); reports decode accuracy.

> Note: the wechat_qrcode pipeline only reports QR codes that zxing successfully decodes (detection + recognition are fused; a box with no decode is not reported). Hence the "Recall" on the detection subset is effectively "recall of recognisable codes", naturally lower for un-decodable samples (damaged/blurred/perspective). All four implementations share the same algorithm; on the same platform their accuracy is in the same ballpark (see the measured matrix in [5.2 Precision Test](#52-test-result)).
>
> Note (EXIF orientation is pinned): directory-mode images (the accuracy path) are read in a way that ignores EXIF orientation — sophon-opencv's `cv::imread(path, IMREAD_IGNORE_ORIENTATION)` (JPU hardware decode on SoC) in both C++ examples (`bmcv.soc`/`sail.soc`), `cv2.imdecode(..., IMREAD_IGNORE_ORIENTATION)` in Python opencv, and `sail.Decoder` (which never applies EXIF) in `bmcv.py` — so EXIF orientation is ignored. OpenCV 4.1 (the BM1684X SDK) auto-applies EXIF rotation in `cv::imread`, while 4.8 (BM1688/CV186X SDK) does not; without the flag the same images would report corners in different coordinate frames across the two SDKs. With the flag the four corners always land in the stored (raw) pixel frame that matches the BoofCV ground truth, keeping results comparable across platforms. `sail::Decoder` never applies EXIF, so it is already consistent.

### 5.2 Test Result

On the BoofCV QR Code V4 set (detection 718 images / 1441 codes, decoding 26 images), per test platform and per test program:

**detection subset** (718 images / 1441 QR codes, a code counts as detected at `IoU >= 0.5`):

| Platform | Program | Recall | Precision |
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

> Reference baselines (the same set, CPU native inference): OpenCV 4.8.0 native `wechat_qrcode` (`opencv-contrib-python 4.8.0.76`, detect/sr caffe — the same source models as this example) Recall 0.4559 / Precision 0.9880; OpenCV 5.0.0 native (`opencv-contrib-python 5.0.0.93`, ONNX models and a new pipeline, not the same source models) Recall 0.3491 / Precision 0.9843. This example compiles 4.8.0's detect/sr caffe source models to bmodels on the TPU, so the TPU rows (recall 0.4740–0.4879) sit close to — and slightly above — 4.8.0 CPU (0.4559), with precision essentially 1.0000.
>
> Note: on the same platform the four programs are in the same ballpark but not identical. The gaps come from (1) different JPEG decoders on the image-loading path — `bmcv.soc` and `sail.soc` both use sophon-opencv's `cv::imread` (JPU hardware decode on SoC; their read and host-algorithm paths are identical), `opencv.py` uses the pip opencv `cv2.imdecode` (libjpeg soft decode), and `bmcv.py` uses `sail.Decoder`'s ffmpeg hardware decode, so per-pixel differences flip detections on a few hard samples; and (2) `bmcv.soc` is a full pure-bmrt+bmcv re-port of the algorithm — its read path matches `sail.soc`, so the ≤2-code gap between them comes from small host-path implementation details (gray conversion, F16→uint8 rounding) rather than image loading. The ffmpeg decode + device VPP (`LINEAR`, vs the host path's `INTER_CUBIC`) of `bmcv.py` also produces a few `IoU<0.5` false boxes on some images, dropping its precision slightly below 1.0000 (688/699 on SE7, 703/704 on SE9).

Per-category recall (16 categories) is printed by `eval_qrcode.py`; clean categories such as `nominal` (0.94 / 0.92) / `pathological` (0.92) / `shadows` (0.92) reach recall >= 0.92, `noncompliant` (0.86 / 0.88) is next, while hard ones such as `lots` (dozens of codes per image, 0.02) and `high_version` (0.23) are lowest and drag down the overall number.

**decoding subset** (26 synthetic codes):

| Platform | Program | Decode accuracy |
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

> vCard/vEvent CRLF line endings normalized before comparison, see `eval_qrcode.py`.

## 6. Performance Testing

### 6.1 bmrt_test

Pure TPU inference time of the two models measured with `bmrt_test` (a single shape, no pre/post-processing):

|   Platform    |            Model              |      input shape      | calculate time(ms) |
|---------------|-------------------------------|-----------------------|--------------------|
| BM1684X       | detect_f32_fused.bmodel       | `[1,1,400,400]`       | 1.64               |
| BM1684X       | sr_f16_fused.bmodel           | `[1,1,224,224]`       | 1.13               |
| BM1688        | detect_f32_fused.bmodel       | `[1,1,400,400]`       | 2.61               |
| BM1688        | sr_f16_fused.bmodel           | `[1,1,224,224]`       | 2.83               |
| CV186X        | detect_f32_fused.bmodel       | `[1,1,400,400]`       | 2.61               |
| CV186X        | sr_f16_fused.bmodel           | `[1,1,224,224]`       | 2.83               |

### 6.2 Program Performance

End-to-end decoding FPS (`images/qr_small.png`, 198x198, one detect TPU inference + zxing decode, iters=50 with 3 warmups):

| Platform | C++ bmcv  | C++ sail  | Python opencv | Python bmcv |
|----------|-----------|-----------|---------------|-------------|
| SE7-32   | 204.9 fps | 205.3 fps | 171.2 fps     | 198.5 fps   |
| SE9-8    | 113.2 fps | 113.3 fps | 106.0 fps     | 111.1 fps   |
| SE9-16   | 110.8 fps | 114.2 fps | 107.3 fps     | 111.1 fps   |

> Platform codes (same convention as PP-OCR and the rest of the demo repo): SE7-32 = BM1684X, SE9-16 = BM1688, SE9-8 = CV186X. The baseline is maintained by `tools/compare_statis.py` (SE7-32~205 / SE9-8~113 / SE9-16~114); all four implementations share the same detect/sr/zxing pipeline, so end-to-end FPS stays in the same order.
>
> Note: the test image is small (198x198) and does not trigger sr, so one detect TPU inference is ~40% of end-to-end time with zxing (CPU) taking the rest; larger images / sr-triggering inputs lower the end-to-end FPS.
>
> Note: `C++ bmcv`, `C++ sail` and `Python bmcv` all go through the device pass-through and land close to each other — the image is hardware-decoded straight into device memory (`C++ bmcv` uses sophon-opencv's 3-arg `cv::imread(path, flags, dev_id)`; `C++ sail` defaults to `sail::Decoder` (`USE_OPENCV_DECODE=0`, pass `-DUSE_OPENCV_DECODE=1` at build time to fall back to the 3-arg `cv::imread`); `Python bmcv` uses `sail.Decoder`), then zero-copy attached as a `bm_image` via `toBMI` / `sail::Bmcv::mat_to_bm_image`, and `detectAndDecode(bm_image)` does on-device VPP resize + inference with no host-to-device full-frame copy. `Python opencv`, by contrast, reads into a host `cv2` ndarray and pays a BGR→GRAY + H→D DMA copy every iteration (~0.8ms extra in detect; on SE7 detect is 2.03ms vs ~1.2ms for the other three). All four variants decode the image once outside the timed loop, so the measurement is like-for-like.

Full-dataset end-to-end decoding time (the whole BoofCV QR Code V4 set: 744 images — 718 detection + 26 decoding, 1441 ground-truth codes; run once through the whole directory, timed as wall-clock from process start to the `results/wechat_qrcode_results.json` flush):

| Platform | images / codes | end-to-end total        | avg per image |
|----------|----------------|-------------------------|---------------|
| SE7-32   | 744 / 1441     | 1218 s (≈20 min 18 s)   | 1.64 s        |
| SE9-8    | 744 / 1441     | 1707 s (≈28 min 27 s)   | 2.29 s        |
| SE9-16   | 744 / 1441     | 1693 s (≈28 min 13 s)   | 2.28 s        |

> Note: the full-set time is dominated by zxing decoding on the CPU — the v4 set contains many dense / high-version / glare / damaged codes (`close` / `high_version` / `glare` / `damaged`, up to 2–10 s each), far exceeding the millisecond-scale detect/sr on the TPU. The wall-clock therefore reflects zxing decode speed: the platform spread comes from CPU capability, with SE7-32 notably faster and SE9-8 / SE9-16 on par.

## 7. FAQ

* **Why two C++ examples (bmcv and sail)?** `bmcv` is a full pure bmrt+bmcv algorithm port (no sail dependency, serving as source reference and minimal-dependency delivery); `sail` is a ~100-line wrapper calling the `sail::wechat_qrcode` API already shipped in `sophon-sail`. They share the same models and performance; accuracy is essentially identical on the same platform (see [5.2 Precision Test](#52-test-result)); pick whichever fits.
* **Why does the detect model have 19 networks?** The tpu-mlir caffe frontend has no dynamic shape, and OpenCV's algorithm rescales input proportionally (`s=min(1, sqrt(160000/(w*h)))`). Nineteen static shapes covering common aspect ratios are combined into one bmodel; a network is chosen by input aspect ratio at runtime, equivalent to OpenCV's proportional rescale.
* **When is sr triggered?** Only crops with `sqrt(w*h) < 160` go through sr; otherwise the crop is scaled by `INTER_CUBIC`/`INTER_AREA` on the CPU. When sr is not triggered, bench shows `sr : 0.00 ms`.
* **Why is the detection-subset recall low?** See [5. Precision Test](#5-precision-test): the pipeline only reports successfully decoded codes.
* **Why is `LD_LIBRARY_PATH` important?** `wechat_qrcode_sail` depends on `libsail.so`, which may not be pre-installed on the test board; `wechat_qrcode_bmcv` avoids this by not depending on sail.
* **How does it compare with the upstream OpenCV wechat_qrcode running on the same board's CPU?** Only detect/sr are moved to the TPU; zxing decoding still runs on the CPU, as in upstream OpenCV. Against `opencv-contrib-python 4.8.0`'s `wechat_qrcode_WeChatQRCode` (cv2.dnn CPU backend) measured on the same A53 board, end-to-end speedup is about **2.2–4.2×**: ~2.2× on a small 198×198 code, ~4.2× on large canvases (1280×720 / 2560×360) carrying a small code. Per stage, detect is ~7–11× and sr ~51× faster, but zxing runs on the CPU on both sides (~28 ms per large image), leaving it the main end-to-end bottleneck the TPU cannot accelerate — so gains converge to ~1.3–1.7× for code-fills-image cases.