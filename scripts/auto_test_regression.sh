#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir
#default config
TARGET="BM1684X"
MODE="pcie_test"
TPUID=0
sail_list=("YOLOv5" "CenterNet") #c++ sail demo

usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [-a SAIL] [ -d TPUID]" 1>&2 
}

while getopts ":m:t:s:a:d:" opt
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
      echo "soc-sdk is $SOCSDK";;
    a)
      SAIL_PATH=${OPTARG}
      echo "sail_path is $SAIL_PATH";;
    d)
      TPUID=${OPTARG}
      echo "using tpu $TPUID";;
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
    if [ -d ./sample/$1/datasets ];then
        rm -r ./sample/$1/datasets
    fi
    if [ -d ./sample/$1/models ];then
        rm -r ./sample/$1/models
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
        res=$(find -name results| grep results)
        if [ $? != 0 ]; then
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
      sudo apt-get install "$package_name"
  fi
}

# Confirm sophon-driver/sophon-libsophon/sophon-mw were previously installed.

# dependencies
test_dpkg unzip
test_dpkg p7zip #LPRNet
test_dpkg p7zip-full
if [ $MODE = "pcie_test" ] || [ $MODE = "soc_build" ]
then
  test_dpkg libeigen3-dev #Trackers
fi

pip3 install dfn==1.0.2
if [ $MODE = "pcie_test" ] || [ $MODE = "soc_test" ]
then
  pip3 install pycocotools
  res=$(pip3 list|grep sophon)
  if [ $? != 0 ];
  then
      echo "Please install sail on your system!"
      exit
  fi
fi

test_sample ByteTrack
test_sample YOLOv5
test_sample YOLOv8
test_sample C3D
test_sample ResNet
test_sample LPRNet
test_sample DeepSORT
test_sample OpenPose

popd
