#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir
#default config
TARGET="BM1684X"
MODE="pcie_test"
TPUID=0

usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [ -d TPUID]" 1>&2 
}

while getopts ":m:t:s:d:" opt
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
    ./sample/$1/scripts/auto_test.sh -t $TARGET -m $MODE -s $SOCSDK -d $TPUID > ./log_auto_test_regression/$1_$current_time.log 2>&1
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

test_sample YOLOv5
test_sample C3D
test_sample ResNet
test_sample YOLOv8
test_sample LPRNet
test_sample DeepSORT
test_sample OpenPose


popd