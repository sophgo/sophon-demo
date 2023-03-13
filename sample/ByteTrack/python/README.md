# Example of ByteTrack with Sophon Inference

**This example can run in PCIe and SoC.**

- [Example of ByteTrack with Sophon Inference](#example-of-bytetrack-with-sophon-inference)
  - [For PCIe](#for-pcie)
    - [Environment Configuration](#environment-configuration)
    - [ByteTrack Bmcv](#bytetrack-bmcv)
    - [ByteTrack Opencv](#bytetrack-opencv)
  - [For SoC](#for-soc)
    - [Environment Configuration](#environment-configuration-1)
    - [ByteTrack Bmcv](#bytetrack-bmcv-1)
    - [ByteTrack Opencv](#bytetrack-opencv-1)
  - [Calculate MOT Metrics](#calculate-mot-metrics)

## For PCIe

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
      --save_path=result-save-path            # ./results/
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
      --file_name=../data/MOT15/ADL-Rundle-6/img1 \
      --bmodel=../data/models/BM1684/bytetrack_s_fp32_1b.bmodel \
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
    python3 bytetrack_opencv.py \
      --output_video=whether-output-video \   # 0 for not , 1 for is
      --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
      --file_name=.your-video-name-or-picture-folder \
      --bmodel=your-path-to-bmodel \
      --save_path=result-save-path            # ./results/
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
      --file_name=../data/MOT15/ADL-Rundle-6/img1 \
      --bmodel=../data/models/BM1684/bytetrack_s_fp32_1b.bmodel \
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


## For SoC
### Environment Configuration

You need to use the SOPHON SDK on the x86 host to build a cross compilation environment, and package the header files and library files that the program depends on into the soc sdk directory. For details, see [交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建).

also needs some third libs, run
```shell
pip3 install -r requirements.txt
```

**If not installed numpy, install numpy**

``` shell
sudo pip3 install numpy==1.17.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### ByteTrack Bmcv
Decoder use sail.Decoder, perprocess use sail.bmcv, inference use sail.Engine.process(graph_name,input_tensors_dict, output_tensors_dict).

**Explanation:**

``` shell
    python3 bytetrack_bmcv.py \
      --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
      --file_name=.your-video-name-or-picture-folder \
      --bmodel=your-path-to-bmodel \
      --save_path=result-save-path            # ./results/
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
      --file_name=../data/MOT15/ADL-Rundle-6/img1 \
      --bmodel=../data/models/BM1684/bytetrack_s_fp32_1b.bmodel \
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
Decoder use cv2, perprocess use cv2 and numpy, inference use sail.Engine.process(graph_name,input_numpys_dict).

**Explanation:**

``` shell
    python3 bytetrack_opencv.py \
      --output_video=whether-output-video \   # 0 for not , 1 for is
      --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
      --file_name=.your-video-name-or-picture-folder \
      --bmodel=your-path-to-bmodel \
      --save_path=result-save-path            # ./results/
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
      --file_name=../data/MOT15/ADL-Rundle-6/img1 \
      --bmodel=../data/models/BM1684/bytetrack_s_fp32_1b.bmodel \
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

## Calculate MOT Metrics

Run mot_eval.py to calculate MOT metrics, ground_truths is the lable file of the test dataset, normally data/MOT15/dataset-name/gt/gt.txt. The detections is the detect result file, under cpp/results and python/results.

``` shell
    pip3 install motmetrics
    python3 ../tools/mot_eval.py \
        --ground_truths=your-ground_truths-file \  # txt file
        --detections=your-detections-file   # txt file
```

**For example:**

``` bash
    pip3 install motmetrics
    python3 ../tools/mot_eval.py \
      --ground_truths=../data/MOT15/ADL-Rundle-6/gt/gt.txt \
      --detections=../python/results/bytetrack_bmcv/img1_bytetrack_s_fp32_1b_py.txt
```

