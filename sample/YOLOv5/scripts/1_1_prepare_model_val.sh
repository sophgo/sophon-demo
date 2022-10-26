#!/bin/bash

pip3 install dfn

root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
model_path=${root_dir}/data/models
images_path=${root_dir}/data/images

pushd ${root_dir}

# export PYTHONPATH=$root_dir/../yolov5
img_size=${1:-640}

function download_pt_model()
{
  # download pt model
  if [ "$img_size" == "1280" ]; then
    if [ ! -f "yolov5s6.pt" ]; then
      echo "download yolov5s6.pt from github"
      wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s6.pt
      if [ $? -ne 0 ];then
        echo "failed!"
      fi
    fi
  else
    if [ ! -f "yolov5s.pt" ]; then
      echo "download yolov5s.pt from github"
      wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
      if [ $? -ne 0 ];then
        echo "failed!"
      fi
    fi
  fi

}

function download_val_dataset()
{
  if [ ! -d ${images_path} ]; then
    mkdir ${images_path} -p
  fi
  pushd ${images_path}

  if [ -d "coco200" ]; then
    echo "coco image 200 exists"
    return 0
  fi

  # download val dataset
  if [ ! -f "coco2017val.zip" ]; then
    echo "download coco2017 val dataset from github"
    wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017val.zip
    if [ $? -ne 0 ];then
      echo "download coco2017 val dataset failed!"
      popd
      exit 1
    fi

    
  fi

  if [ ! -d "coco" ]; then
    echo "unzip coco2017val.zip"
    unzip coco2017val.zip
  fi

  # choose 200 images and copy to ./images/coco200
  mkdir coco200 -p
  ls -l coco/images/val2017 | sed -n '2,201p' | awk -F " " '{print $9}' | xargs -t -i cp ./coco/images/val2017/{} ./coco200/
  echo "200 jpg files has been choosen"

  popd
}

function download_model()
{
  # JIT模型
  if [ ! -d "${model_path}/torch" ]; then
    mkdir ${model_path}/torch -p
  fi

  pushd ${model_path}/torch
  if [ ! -f "yolov5s_640_coco_v6.1_3output.torchscript.pt" ]; then
    echo "start downloading bmodel yolov5s_640_coco_v6.1_3output.torchscript"
    python3 -m dfn --url  http://219.142.246.77:65000/sharing/B4osQWed8
    
    if [ $? -eq 0 ]; then
      mv ${model_path}/torch/yolov5s_640_coco_v6.1_3output.torchscript ${model_path}/torch/yolov5s_640_coco_v6.1_3output.torchscript.pt
      echo "yolov5s torchscript done!"
    else
        echo "Something is wrong, pleae have a check!"
        popd
        exit -1
    fi
  fi
  popd

  # fp32bmodel
  if [ ! -d "${model_path}/BM1684X" ]; then
    mkdir ${model_path}/BM1684X -p
  fi

  pushd ${model_path}/BM1684X
  if [ ! -f "yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel" ]; then
    echo "start downloading BM1684X fp32 bmodel"
    python3 -m dfn --url  http://219.142.246.77:65000/sharing/x0jnYnwU4
    popd
    if [ $? -eq 0 ]; then
      echo "BM1684X fp32 bmodel done!"        
    else
        echo "Something is wrong, pleae have a check!"
        exit -1
    fi
  fi
 
  # int8bmodel
  if [ ! -d "${model_path}/BM1684X" ]; then
    mkdir ${model_path}/BM1684X -p
  fi

  pushd ${model_path}/BM1684X
  if [ ! -f "yolov5s_640_coco_v6.1_3output_int8_1b.bmodel" ]; then
    echo "start downloading BM1684X int8 bmodel"
    python3 -m dfn --url  http://219.142.246.77:65000/sharing/ZSRqgFlYD
    popd
    if [ $? -eq 0 ]; then
      echo "BM1684X int8 bmodel done!"
    else
        echo "Something is wrong, pleae have a check!"
        exit -1
    fi
  fi

  # fp32bmodel
  if [ ! -d "${model_path}/BM1684" ]; then
    mkdir ${model_path}/BM1684 -p
  fi

  pushd ${model_path}/BM1684
  if [ ! -f "yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel" ]; then
    echo "start downloading BM1684 fp32 bmodel"
    python3 -m dfn --url  http://219.142.246.77:65000/sharing/OSjGIEF6x
    popd
    if [ $? -eq 0 ]; then
      echo "BM1684 fp32 bmodel done!"        
    else
        echo "Something is wrong, pleae have a check!"
        exit -1
    fi
  fi
 
  # int8bmodel
  if [ ! -d "${model_path}/BM1684" ]; then
    mkdir ${model_path}/BM1684 -p
  fi

  pushd ${model_path}/BM1684
  if [ ! -f "yolov5s_640_coco_v6.1_3output_int8_1b.bmodel" ]; then
    echo "start downloading BM1684 int8 bmodel"
    python3 -m dfn --url  http://219.142.246.77:65000/sharing/DJVDbPsOO
    popd
    if [ $? -eq 0 ]; then
      echo "BM1684 int8 bmodel done!"
    else
        echo "Something is wrong, pleae have a check!"
        exit -1
    fi
  fi
}

download_val_dataset
download_model

popd
