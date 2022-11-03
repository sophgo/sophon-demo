#!/bin/bash
pip3 install dfn

script_dir=$(dirname $(readlink -f "$0"))
root_dir=$script_dir/..
data_dir=$root_dir/data
if [ ! -d "$data_dir" ]; then
  echo "create data dir: $data_dir"
  mkdir -p $data_dir
fi
#下载测试图片
pushd $script_dir
python3 -m dfn --url http://219.142.246.77:65000/sharing/Pj5UVvVsO
python3 -m dfn --url  http://219.142.246.77:65000/sharing/OSPQaiNZV
python3 -m dfn --url  http://219.142.246.77:65000/sharing/FDEJ2DSGa
python3 -m dfn --url  http://219.142.246.77:65000/sharing/dFCjcxw5I
python3 -m dfn --url  http://219.142.246.77:65000/sharing/2CavuWvQx
#下载测试视频
python3 -m dfn --url  http://219.142.246.77:65000/sharing/v4mqladtx
#下载darknet_model原文件
python3 -m dfn --url http://219.142.246.77:65000/sharing/IFj5foNqc
python3 -m dfn --url http://219.142.246.77:65000/sharing/DNNA06C2W
#下载测试数据集
python3 -m dfn --url http://219.142.246.77:65000/sharing/yhT38P98U
#下载bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/FxRDnDSDR
python3 -m dfn --url http://219.142.246.77:65000/sharing/N5eGQBuy7
popd
pushd $data_dir

image_dir=$data_dir/images
if [ ! -d "$image_dir" ]; then
  echo "create image dir: $image_dir"
  mkdir -p $image_dir
fi
pushd $image_dir
mv $script_dir/bus.jpg ./
mv $script_dir/dog.jpg ./
mv $script_dir/horses.jpg ./
mv $script_dir/person.jpg ./
mv $script_dir/zidane.jpg ./
mv $script_dir/coco_val2017.zip ./

val_dir=$image_dir/val2017

if [ ! -d "$val_dir" ]; then
  echo "unzip coco_val2017.zip dir: $val_dir"
  unzip coco_val2017.zip
fi

coco200_dir=$image_dir/coco200
if [ ! -d "$coco200_dir" ]; then
    mkdir coco200 -p
    ls -l val2017 | sed -n '2,201p' | awk -F " " '{print $9}' | xargs -t -i cp ./val2017/{} ./coco200/
    echo "coco image 200 exists"
fi

popd

video_dir=$data_dir/videos
if [ ! -d "$video_dir" ]; then
  echo "create video dir: $video_dir"
  mkdir -p $video_dir
fi
pushd $video_dir
mv $script_dir/dance.mp4 ./
popd


models_dir=$data_dir/models
if [ ! -d "$models_dir" ]; then
  echo "create models dir: $models_dir"
  mkdir -p $models_dir
fi

pushd $models_dir
darknet_dir=$models_dir/darknet
if [ ! -d "$darknet_dir" ]; then
  echo "create models dir: $darknet_dir"
  mkdir -p $darknet_dir
fi

pushd $darknet_dir
mv $script_dir/yolov4.weights ./
mv $script_dir/yolov4.cfg ./
popd

bmodel_1684=$models_dir/BM1684
if [ ! -d "$bmodel_1684" ]; then
  echo "create models dir: $bmodel_1684"
  mkdir -p $bmodel_1684
fi

pushd $bmodel_1684
mv $script_dir/yolov4_416_coco_int8_1b.bmodel ./
mv $script_dir/yolov4_416_coco_fp32_1b.bmodel ./
popd

bmodel_1684X=$models_dir/BM1684X
if [ ! -d "$bmodel_1684X" ]; then
  echo "create models dir: $bmodel_1684X"
  mkdir -p $bmodel_1684X
fi

pushd $bmodel_1684X
python3 -m dfn --url http://219.142.246.77:65000/sharing/ThY1qIACq
python3 -m dfn --url http://219.142.246.77:65000/sharing/K7iu7DeYG
popd

popd
popd
