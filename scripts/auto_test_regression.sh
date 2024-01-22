#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir
#default config
TARGET="BM1684X"
MODE="pcie_test"
TPUID=0
RENEW=0
sail_list=("YOLOv5" "CenterNet" "BERT" "ppYOLOv3" "YOLOv34" "YOLOX" "segformer" "ppYoloe" "YOLOv5_opt") #c++ sail demo
PYTEST="auto_test"

usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [-a SAIL] [ -d TPUID] [-r Renew] [-p PYTEST auto_test|pytest]" 1>&2 
}

while getopts ":m:t:s:a:d:r:p:" opt
do
  case $opt in 
    m)
      MODE=${OPTARG}
      echo "mode is $MODE";;
    t)
      TARGET=${OPTARG}
      echo "target is $TARGET";;
    s)
      SOCSDK=${OPTARG}
      SOCSDK=$(readlink -f "$SOCSDK")
      echo "soc-sdk is $SOCSDK";;
    a)
      SAIL_PATH=${OPTARG}
      SAIL_PATH=$(readlink -f "$SAIL_PATH")
      echo "sail_path is $SAIL_PATH";;
    d)
      TPUID=${OPTARG}
      echo "using tpu $TPUID";;
    r)
      RENEW=${OPTARG}
      echo "Renew datasets and models";;
    p)
      PYTEST=${OPTARG}
      echo "run in $PYTEST";;
    ?)
      usage
      exit 1;;
  esac
done

# $1: sample name 
function test_sample(){
    echo "=============="
    echo "NOW: $1"
    echo "=============="
    chmod +x ./sample/$1/scripts/auto_test.sh
    if [ $RENEW != 0 ]; then
      echo "Renew all datasets and models."
      if [ -d ./sample/$1/datasets ];then
          rm -r ./sample/$1/datasets
      fi
      if [ -d ./sample/$1/models ];then
          rm -r ./sample/$1/models
      fi
    else
      echo "Use current datasets and models."
    fi
    if [ ! -d ./log_auto_test_regression ];then
        mkdir log_auto_test_regression
    fi
    current_time=$(date +%Y-%m-%d_%H:%M:%S)
    if [[ " ${sail_list[*]} " == *" $1 "* ]]; then
      ./sample/$1/scripts/auto_test.sh -t $TARGET -m $MODE -s $SOCSDK -a $SAIL_PATH -d $TPUID > ./log_auto_test_regression/$1_$current_time.log 2>&1
    else
      ./sample/$1/scripts/auto_test.sh -t $TARGET -m $MODE -s $SOCSDK -d $TPUID > ./log_auto_test_regression/$1_$current_time.log 2>&1
    fi
    tail -n 4 ./log_auto_test_regression/$1_$current_time.log | head -n 3
    if test $MODE = "soc_test"
    then
        res=$(find -name results)
        if [ "$res" != '' ]; then
            rm -r `find -name results`
        fi
        rm -r ./sample/$1/datasets
        rm -r ./sample/$1/models
    fi
    echo "=============="
    echo "EXIT: $1"
    echo "=============="
}

function test_dpkg(){
  package_name=$1
  if dpkg -s "$package_name" >/dev/null 2>&1; then
      echo "$package_name is already installed."
  else
      echo "$package_name is not installed. Installing now..."
      if [ $MODE = "compile_nntc" ] || [ $MODE = "compile_mlir" ];then
        apt-get install "$package_name"
      else
        sudo apt-get install "$package_name"
      fi
  fi
}

# Assume sophon-driver/sophon-libsophon/sophon-mw were previously installed.

# dependencies
test_dpkg unzip
test_dpkg p7zip #LPRNet
test_dpkg p7zip-full
if [ $MODE = "pcie_test" ] || [ $MODE = "soc_build" ]
then
  test_dpkg libeigen3-dev #Trackers
fi

if [ $MODE = "pcie_test" ] || [ $MODE = "soc_test" ]
then
  pip3 install pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple 
  pip3 install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple 
  res=$(pip3 list|grep sophon)
  if [ $? != 0 ];
  then
      echo "Please install sail on your system!"
      exit
  fi
fi

if [ $PYTEST != "pytest" ]
then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
  test_sample ByteTrack
  test_sample YOLOv5
  test_sample YOLOv8
  test_sample C3D
  test_sample CenterNet
  test_sample DeepSORT
  test_sample LPRNet
  test_sample OpenPose
  test_sample ResNet
  test_sample YOLOv5
  test_sample YOLOv5_opt
  test_sample YOLOv8
fi

popd
