# Example of YOLOX with Sophon Inference

  **this example can run in pcie and soc**

## For pcie 

### Environment configuration 

libsophon sophon-ffmpeg sophon-opencv sophon-sail should be download and installed，for details see [x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建).

also needs some third libs, run
```shell
pip3 install -r requirements.txt
```
### yolox_bmcv.py
 decoder use sail.Decoder, perprocess use sail.bmcv, inference use sail.Engine.process(graph_name,input_tensors_dict, output_tensors_dict)

- Run example

``` shell
    python3 yolox_bmcv.py \
        --bmodel_path=your-path-to-bmodel \
        --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
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
        --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
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

You need to use the SOPHON SDK on the x86 host to build a cross compilation environment, and package the header files and library files that the program depends on into the soc sdk directory. For details, see [交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#31-交叉编译环境搭建).

also needs some third libs, run
```shell
pip3 install -r requirements.txt
```

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
        --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
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
        --is_video=test-file-is-video-or-not \  # 0 for not , 1 for is
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

run calc_mAP.py to calculate mAP, ground_truths is the lable file of the test dataset, normally data/ground_truths/instances_val2017.json. The detections is the detect result file, under cpp/results and python/results

``` shell
    pip3 install pycocotools
    python3 ../tools/calc_mAP.py \
        --ground_truths=your-ground_truths-file \ #json file
        --detections=your-detections-file \ #txt file
```
