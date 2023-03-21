[简体中文](../python/README.md) | English
# ByteTrack Python Example

**This example can run in PCIe and SoC.**

- [ByteTrack Python Example](#bytetrack-python-example)
  - [1. For PCIe](#1-for-pcie)
    - [Environment Configuration](#environment-configuration)
    - [ByteTrack Bmcv](#bytetrack-bmcv)
    - [ByteTrack Opencv](#bytetrack-opencv)
  - [2. For SoC](#2-for-soc)
    - [Environment Configuration](#environment-configuration-1)
    - [ByteTrack Bmcv](#bytetrack-bmcv-1)
    - [ByteTrack Opencv](#bytetrack-opencv-1)
  - [3 Inference Test](#3-inference-test)
    - [3.1 Test MOT Dataset](#31-test-mot-dataset)
    - [4.2 Test Video](#42-test-video)
  - [3.3 Calculate MOT Metrics](#33-calculate-mot-metrics)

## 1. For PCIe

### Environment Configuration

libsophon sophon-ffmpeg sophon-opencv sophon-sail should be download and installed，for details see [x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建).

Also needs some third libs, run
```shell
pip3 install -r requirements.txt
```

### ByteTrack Bmcv
Decoder use sail.Decoder, perprocess use sail.bmcv, inference use sail.Engine.process(graph_name,input_tensors_dict, output_tensors_dict).

**Explanation:**

``` shell
    python3 bytetrack_bmcv.py \
      --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
      --file_name=.your-video-name-or-picture-folder \
      --bmodel=your-path-to-bmodel \
      --save_path=result-save-path    \          # ./results/
      --score_th=detect-threshold \           # default 0.1
      --nms_th=nms-threshold \                # default 0.7
      --device_id=use-tpu-id \                # default 0
      --track_thresh=track-thresh \           # default 0.5
      --track_buffer=track-buffer \           # default 30
      --match_thresh=match_thresh \           # default 0.8
      --min-box-area=min-box-area \           # default 10
```

**For example:**

```bash
    python3 ../python/bytetrack_bmcv.py \
      --is_video=0 \
      --file_name=../datasets/MOT15/ADL-Rundle-6/img1 \
      --bmodel=../models/BM1684/bytetrack_s_fp32_1b.bmodel \
      --save_path=../python/results/bytetrack_bmcv \
      --score_th=0.1 \
      --nms_th=0.7 \
      --device_id=0 \
      --track_thresh=0.5 \
      --track_buffer=30 \
      --match_thresh=0.8 \
      --min-box-area=10
```

**Result**

Results is in your save path.

For picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_py.txt.


### ByteTrack Opencv
Decoder use cv2, perprocess use cv2 and numpy, inference use sail.Engine.process(graph_name,input_numpys_dict).

**Explanation:**

``` shell
    python3 bytetrack_opencv.py  \
      --output_video=whether-output-video \   # 0 for not , 1 for is
      --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
      --file_name=.your-video-name-or-picture-folder \
      --bmodel=your-path-to-bmodel \
      --save_path=result-save-path  \         # ./results/
      --score_th=detect-threshold \           # default 0.1
      --nms_th=nms-threshold \                # default 0.7
      --device_id=use-tpu-id \                # default 0
      --track_thresh=track-thresh \           # default 0.5
      --track_buffer=track-buffer \           # default 30
      --match_thresh=match_thresh \           # default 0.8
      --min-box-area=min-box-area \           # default 10
```

**For example:**

```bash
    python3 ../python/bytetrack_opencv.py \
      --output_video=0 \
      --is_video=0 \
      --file_name=../datasets/MOT15/ADL-Rundle-6/img1 \
      --bmodel=../models/BM1684/bytetrack_s_fp32_1b.bmodel \
      --save_path=../python/results/bytetrack_opencv \
      --score_th=0.1 \
      --nms_th=0.7 \
      --device_id=0 \
      --track_thresh=0.5 \
      --track_buffer=30 \
      --match_thresh=0.8 \
      --min-box-area=10
```

**Result**

Result in your save path

For picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_py.txt.

For video, you can choose whether output video file, save video name is [video name].mp4, save txt name is [video name]_[bmodel name]_py.txt.


## 2. For SoC

### Environment Configuration

you need some third libs, run
```shell
pip3 install -r requirements.txt
```

**If not installed numpy, install numpy**

``` shell
sudo pip3 install numpy==1.17.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### ByteTrack Bmcv
Decoder use `sail.Decoder`, perprocess use `sail.bmcv`, inference use `sail.Engine.process(graph_name,input_tensors_dict, output_tensors_dict)`.

**Explanation:**

``` shell
    python3 bytetrack_bmcv.py \
      --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
      --file_name=.your-video-name-or-picture-folder \
      --bmodel=your-path-to-bmodel \
      --save_path=result-save-path    \        # ./results/
      --score_th=detect-threshold \           # default 0.1
      --nms_th=nms-threshold \                # default 0.7
      --device_id=use-tpu-id \                # default 0
      --track_thresh=track-thresh \           # default 0.5
      --track_buffer=track-buffer \           # default 30
      --match_thresh=match_thresh \           # default 0.8
      --min-box-area=min-box-area \           # default 10
```

**For example:**

```bash
    python3 ../python/bytetrack_bmcv.py \
      --is_video=0 \
      --file_name=../datasets/MOT15/ADL-Rundle-6/img1 \
      --bmodel=../models/BM1684/bytetrack_s_fp32_1b.bmodel \
      --save_path=../python/results/bytetrack_bmcv \
      --score_th=0.1 \
      --nms_th=0.7 \
      --device_id=0 \
      --track_thresh=0.5 \
      --track_buffer=30 \
      --match_thresh=0.8 \
      --min-box-area=10
```

**Result**

Results is in your save path.

For picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_py.txt.

For video, you can choose whether output video file, save video name is [video name].mp4, save txt name is [video name]_[bmodel name]_py.txt.

### ByteTrack Opencv
Decoder use `cv2`, perprocess use `cv2` and `numpy`, inference use `sail.Engine.process(graph_name,input_numpys_dict)`.

**Explanation:**

``` shell
    python3 bytetrack_opencv.py \
      --output_video=whether-output-video \   # 0 for not , 1 for is
      --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
      --file_name=.your-video-name-or-picture-folder \
      --bmodel=your-path-to-bmodel \
      --save_path=result-save-path  \          # ./results/
      --score_th=detect-threshold \           # default 0.1
      --nms_th=nms-threshold \                # default 0.7
      --device_id=use-tpu-id \                # default 0
      --track_thresh=track-thresh \           # default 0.5
      --track_buffer=track-buffer \           # default 30
      --match_thresh=match_thresh \           # default 0.8
      --min-box-area=min-box-area \           # default 10
```

**For example:**

```bash
    python3 ../python/bytetrack_opencv.py \
      --output_video=0 \
      --is_video=0 \
      --file_name=../datasets/MOT15/ADL-Rundle-6/img1 \
      --bmodel=../models/BM1684/bytetrack_s_fp32_1b.bmodel \
      --save_path=../python/results/bytetrack_opencv \
      --score_th=0.1 \
      --nms_th=0.7 \
      --device_id=0 \
      --track_thresh=0.5 \
      --track_buffer=30 \
      --match_thresh=0.8 \
      --min-box-area=10
```

**Result**

Results is in your save path.

For picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_py.txt.

For video, you can choose whether output video file, save video name is [video name].mp4, save txt name is [video name]_[bmodel name]_py.txt.


## 3 Inference Test
The python example does not need to be compiled and can be run directly. The test parameters and operation methods of the PCIe platform and SoC platform are the same.

### 3.1 Test MOT Dataset
```bash
    cd python
    python3 ../python/bytetrack_bmcv.py \
      --is_video=0 \
      --file_name=../datasets/MOT15/ADL-Rundle-6/img1 \
      --bmodel=../models/BM1684/bytetrack_s_fp32_1b.bmodel \
      --save_path=../python/results/bytetrack_bmcv \
      --score_th=0.1 \
      --nms_th=0.7 \
      --device_id=0 \
      --track_thresh=0.5 \
      --track_buffer=30 \
      --match_thresh=0.8 \
      --min-box-area=10
```

**bmcv**:
```bash
INFO:root:------------------------ByteTrack test-----------------------------
INFO:root:frame_num:525
INFO:root:overall_time(ms): 95.77
INFO:root:------------------Detector Predict Time Info ----------------------
INFO:root:preprocess_time(ms): 30.19
INFO:root:inference_time(ms): 41.50
INFO:root:postprocess_time(ms): 7.66
INFO:root:-------------------------------------------------------------------
INFO:root:------------------ByteTrack Tracker Time Info ----------------------
INFO:root:preprocess_time(ms): 9.39
INFO:root:-------------------------------------------------------------------
INFO:root:Save results to ../python/results/bytetrack_bmcv/img1_bytetrack_s_fp32_1b_py.txt
```

**opencv**:
```bash
INFO:root:------------------------ByteTrack test-----------------------------
INFO:root:frame_num:525
INFO:root:overall_time(ms): 359.31
INFO:root:------------------Detector Predict Time Info ----------------------
INFO:root:preprocess_time(ms): 214.70
INFO:root:inference_time(ms): 54.59
INFO:root:postprocess_time(ms): 7.87
INFO:root:-------------------------------------------------------------------
INFO:root:------------------ByteTrack Tracker Time Info ----------------------
INFO:root:preprocess_time(ms): 10.26
INFO:root:-------------------------------------------------------------------
INFO:root:Save results to ../python/results/bytetrack_opencv/img1_bytetrack_s_fp32_1b_py.txt
```


### 4.2 Test Video
The video test example is as follows, and it supports testing of video streams.
```bash
    cd python
    python3 ../python/bytetrack_bmcv.py \
      --is_video=1 \
      --file_name=../datasets/sample.mp4 \
      --bmodel=../models/BM1684/bytetrack_s_fp32_1b.bmodel \
      --save_path=../python/results/bytetrack_bmcv \
      --score_th=0.1 \
      --nms_th=0.7 \
      --device_id=0 \
      --track_thresh=0.5 \
      --track_buffer=30 \
      --match_thresh=0.8 \
      --min-box-area=10
```

```bash
INFO:root:------------------------ByteTrack test-----------------------------
INFO:root:frame_num:1624
INFO:root:overall_time(ms): 99.91
INFO:root:------------------Detector Predict Time Info ----------------------
INFO:root:preprocess_time(ms): 34.51
INFO:root:inference_time(ms): 41.53
INFO:root:postprocess_time(ms): 10.20
INFO:root:-------------------------------------------------------------------
INFO:root:------------------ByteTrack Tracker Time Info ----------------------
INFO:root:preprocess_time(ms): 12.88
INFO:root:-------------------------------------------------------------------
INFO:root:Save results to ../python/results/bytetrack_bmcv/sample_bytetrack_s_fp32_1b_py.txt
```



## 3.3 Calculate MOT Metrics

Run eval_mot.py to calculate MOT metrics, ground_truths is the lable file of the test dataset, normally datasets/MOT15/ADL-Rundle-6/gt/gt.txt. The --detections is the detect result file, under cpp/results and python/results.

``` shell
    pip3 install motmetrics
    python3 ../tools/eval_mot.py \
        --ground_truths=your-ground_truths-file \  # txt file
        --detections=your-detections-file   # txt file
```

**For example:**

``` bash
    pip3 install motmetrics
    python3 ../tools/eval_mot.py \
      --ground_truths=../datasets/MOT15/ADL-Rundle-6/gt/gt.txt \
      --detections=../python/results/bytetrack_bmcv/img1_bytetrack_s_fp32_1b_py.txt
```

结果：
```bash
MOTA = -0.3709323218207228
     num_frames      IDF1       IDP       IDR      Rcll      Prcn    GT  MT  PT  ML    FP    FN  IDsw   FM      MOTA      MOTP
acc         525  0.051613  0.068174  0.041525  0.142344  0.233694  5009   0   7  17  2338  4296   233  491 -0.370932  0.345532
```

**opencv**:
```bash
MOTA = -0.44140547015372333
     num_frames      IDF1       IDP       IDR     Rcll      Prcn    GT  MT  PT  ML    FP    FN  IDsw   FM      MOTA      MOTP
acc         525  0.054362  0.067037  0.045718  0.14454  0.211944  5009   0   6  18  2692  4285   243  492 -0.441405  0.345389
```