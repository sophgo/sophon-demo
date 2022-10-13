# Example of YOLOX with Sophon Inference

  **this example can run in pcie with docker and soc**

## For pcie with docker

### Environment configuration 

libsophon sophon-ffmpeg sophon-opencv sophon-sail should be download and install

### yolox_bmcv.py
 decoder use sail.Decoder, perprocess use sail.bmcv, inference use sail.Engine.process(graph_name,input_tensors_dict, output_tensors_dict)

- Run example

``` shell
    python3 yolox_bmcv.py \
        --bmodel_path=your-path-to-bmodel \
        --is_video=test-file-is-video-or-not \  # 0 or 1
        --file_name=your-video-name-or-picture-folder \
        --loops=video-inference-count \         # only used for video
        --device_id=use-tpu-id \                # defaule 0
        --detect_threshold=detect-threshold \   # default 0.25
        --nms_threshold=nms-threshold \         # default 0.45
        --save_path=result-save-path            # ./results/
```
- Result

result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_py.txt

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_[device id].jpg, save txt name is [video name]_[bmodel name]_py.txt

### yolox_opencv.py
 decoder use cv2, perprocess use cv2 and numpy, inference use sail.Engine.process(graph_name,input_numpys_dict)

- Run example

``` shell
    python3 yolox_opencv.py \
        --bmodel_path=your-path-to-bmodel \
        --is_video=test-file-is-video-or-not \  # 0 or 1
        --file_name=your-video-name-or-picture-folder \
        --loops=video-inference-count \         # only used for video
        --device_id=use-tpu-id \                # defaule 0
        --detect_threshold=detect-threshold \   # default 0.25
        --nms_threshold=nms-threshold \         # default 0.45
        --save_path=result-save-path            # ./results/
```
- Result

result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_py.txt

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_[device id].jpg, save txt name is [video name]_[bmodel name]_py.txt


## For soc
### Environment configuration 

libsophon-soc sophon-ffmpeg-soc sophon-opencv-soc sophon-sail should be download and install 

### If not installed numpy, install numpy

``` shell
    sudo pip3 install numpy==1.17.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### yolox_bmcv.py
 decoder use sail.Decoder, perprocess use sail.bmcv, inference use sail.Engine.process(graph_name,input_tensors_dict, output_tensors_dict)

- Run example

``` shell
    python3 yolox_bmcv.py \
        --bmodel_path=your-path-to-bmodel \
        --is_video=test-file-is-video-or-not \  # 0 or 1
        --file_name=your-video-name-or-picture-folder \
        --loops=video-inference-count \         # only used for video
        --detect_threshold=detect-threshold \   # default 0.25
        --nms_threshold=nms-threshold \         # default 0.45
        --save_path=result-save-path            # default ./results/
```
- Result

result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_py.txt

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_[device id].jpg, save txt name is [video name]_[bmodel name]_py.txt

### yolox_opencv.py
 decoder use cv2, perprocess use cv2 and numpy, inference use sail.Engine.process(graph_name,input_numpys_dict)

- Run example

``` shell
    python3 yolox_opencv.py \
        --bmodel_path=your-path-to-bmodel \
        --is_video=test-file-is-video-or-not \  # 0 or 1
        --file_name=your-video-name-or-picture-folder \
        --loops=video-inference-count \         # only used for video
        --device_id=use-tpu-id \                # defaule 0
        --detect_threshold=detect-threshold \   # default 0.25
        --nms_threshold=nms-threshold \         # default 0.45
        --save_path=result-save-path            # default ./results/
```
- Result

result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_py.txt

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_0.jpg, save txt name is [video name]_[bmodel name]_py.txt


## calculate mAP
``` shell
    python3 ../scripts/calc_mAP.py \
        --ground_truths=your-ground_truths-file \ #json file
        --detections=your-detections-file \ #txt file
```
