简体中文 ｜ [English](../docs/README_PY_EN.md)
# ByteTrack Python 示例

此示例可在PCIe和SoC上运行。

- [ByteTrack Python 示例](#bytetrack-python-示例)
  - [1. PCIe](#1-pcie)
    - [环境配置](#环境配置)
    - [ByteTrack Bmcv](#bytetrack-bmcv)
    - [ByteTrack Opencv](#bytetrack-opencv)
  - [2. For SoC](#2-for-soc)
    - [环境配置](#环境配置-1)
    - [ByteTrack Bmcv](#bytetrack-bmcv-1)
    - [ByteTrack Opencv](#bytetrack-opencv-1)
  - [3 推理计算](#3-推理计算)
    - [3.1 测试MOT数据集](#31-测试mot数据集)
    - [3.2 测试视频](#32-测试视频)
  - [3.3 计算MOT指标](#33-计算mot指标)


## 1. PCIe

### 环境配置

需要下载和安装libsophon、sophon-ffmpeg、sophon-opencv和sophon-sail等软件包，具体细节请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建).

此外，还需要一些第三方库，请运行以下命令：
```shell
pip3 install -r requirements.txt
```

### ByteTrack Bmcv

解码器使用`sail.Decoder`，预处理使用`sail.bmcv`，推理使用`sail.Engine.process(graph_name,input_tensors_dict,output_tensors_dict)`。

**说明**：

``` shell
    python3 bytetrack_bmcv.py \
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

**例如**：

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

**结果**：

结果将保存在您指定的路径中。

对于图片，保存的图片名称与原始图片名称相同，保存的txt文件名称格式为 [ost picture name]_[bmodel name]_py.txt。

### ByteTrack Opencv

Decoder 使用的是 cv2，预处理使用的是 cv2 和 numpy，推理使用的是 `sail.Engine.process(graph_name,input_numpys_dict)`。

**翻译**：

``` shell
    python3 bytetrack_opencv.py  \
      --output_video=whether-output-video \   # 0 for not , 1 for is
      --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
      --file_name=.your-video-name-or-picture-folder \
      --bmodel=your-path-to-bmodel \
      --save_path=result-save-path   \         # ./results/
      --score_th=detect-threshold \           # default 0.1
      --nms_th=nms-threshold \                # default 0.7
      --device_id=use-tpu-id \                # default 0
      --track_thresh=track-thresh \           # default 0.5
      --track_buffer=track-buffer \           # default 30
      --match_thresh=match_thresh \           # default 0.8
      --min-box-area=min-box-area \           # default 10
```

例如：

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

**结果**：

结果保存在您的存储路径中。

对于图片，保存的名称与原始名称相同，保存的txt名称为[ost picture name]_[bmodel name]_py.txt。

对于视频，您可以选择是否输出视频文件，保存的视频名称为[video name].mp4，保存的txt名称为[video name]_[bmodel name]_py.txt。

## 2. For SoC

### 环境配置

需要安装第三方库，
```shell
pip3 install -r requirements.txt
```

**如果没有安装numpy，请安装numpy**

``` shell
sudo pip3 install numpy==1.17.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### ByteTrack Bmcv
解码器使用`sail.Decoder`，预处理使用`sail.bmcv`，推理使用`sail.Engine.process(graph_name,input_tensors_dict, output_tensors_dict)`。

**说明**：

``` shell
    python3 bytetrack_bmcv.py \
      --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
      --file_name=.your-video-name-or-picture-folder \
      --bmodel=your-path-to-bmodel \
      --save_path=result-save-path \           # ./results/
      --score_th=detect-threshold \           # default 0.1
      --nms_th=nms-threshold \                # default 0.7
      --device_id=use-tpu-id \                # default 0
      --track_thresh=track-thresh \           # default 0.5
      --track_buffer=track-buffer \           # default 30
      --match_thresh=match_thresh \           # default 0.8
      --min-box-area=min-box-area \           # default 10
```

例如：

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

**结果**：

结果保存在您的存储路径中。

对于图片，保存的名称与原始名称相同，保存的txt名称为[ost picture name]_[bmodel name]_py.txt。

对于视频，您可以选择是否输出视频文件，保存的视频名称为[video name].mp4，保存的txt名称为[video name]_[bmodel name]_py.txt。

### ByteTrack Opencv

Decoder 使用的是 `cv2`，预处理使用的是 `cv2` 和 `numpy`，推理使用的是 `sail.Engine.process(graph_name,input_numpys_dict)`。

**翻译**：

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

例如：

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

**结果**：

结果保存在您的存储路径中。

对于图片，保存的名称与原始名称相同，保存的txt名称为[ost picture name]_[bmodel name]_py.txt。

对于视频，您可以选择是否输出视频文件，保存的视频名称为[video name].mp4，保存的txt名称为[video name]_[bmodel name]_py.txt。


## 3 推理计算
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 3.1 测试MOT数据集
MOT数据集测试实例如下，支持对整个文件夹里的所有图片进行测试。以bmcv版本为例：

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
INFO:root:track_time(ms): 9.39
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
INFO:root:track_time(ms): 10.26
INFO:root:-------------------------------------------------------------------
INFO:root:Save results to ../python/results/bytetrack_opencv/img1_bytetrack_s_fp32_1b_py.txt
```

### 3.2 测试视频
视频测试实例如下，支持对视频流进行测试。
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
INFO:root:track_time(ms): 12.88
INFO:root:-------------------------------------------------------------------
INFO:root:Save results to ../python/results/bytetrack_bmcv/sample_bytetrack_s_fp32_1b_py.txt
```

## 3.3 计算MOT指标

运行 eval_mot.py 来计算 MOT 指标，其中 ground_truths 是测试数据集的标注文件，通常为 datasets/MOT15/ADL-Rundle-6/gt/gt.txt。--detections 是检测结果文件，位于 cpp/results 和 python/{bytetrack-version}/results 下。

``` shell
    pip3 install motmetrics
    python3 ../tools/eval_mot.py \
        --ground_truths=your-ground_truths-file \  # txt file
        --detections=your-detections-file   # txt file
```

例如：
**bmcv:**

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