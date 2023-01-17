#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684X"
MODE="pcie_test"
SOCSDK="/home/lihengfang/work/sophon-demo/sample/C3D/soc-sdk"
TPUID=0
ALL_PASS=1

usage() 
{
  echo "Usage: $0 [ -m MODE compile|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [ -d TPUID]" 1>&2 
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

function download(){
    #download dataset and models.
    if [ ! -d './data/UCF_test_01' -o ! -d './data/models' ]; then
        echo "preparing datasets and models......"
        chmod +x ./scripts/download.sh
        ./scripts/download.sh
    else
        echo "data already exists!"
    fi
}

function build_pcie(){
    pushd cpp/c3d_$1
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. && make
    popd
}

function build_soc()
{
    pushd cpp/c3d_$1
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
    popd
}

function compare_res(){
    ret=`awk -v x=$1 -v y=$2 'BEGIN{print(x-y<0.001 && y-x<0.001)?1:0}'`
    if [ $ret -eq 0 ]
    then
        ALL_PASS=0
        echo "compare wrong!"
    else
        echo "compare right!"
    fi
}
#e.g.: test_cpp opencv pcie c3d_int8_1b.bmodel 0.715
function test_cpp(){
    pushd cpp/c3d_$1/build
    echo "------------------"
    echo "testing cpp $1 $3:"
    echo "------------------"
    chmod +x ./c3d_$1.$2
    res=$(./c3d_$1.$2 ../../../data/UCF_test_01 ../../../data/models/$TARGET/$3 $TPUID 2>&1 | tee $1_$3.log)
    acc=(${res##*now:})
    gt=$4
    compare_res $acc $gt
    popd
}

#e.g.: test_python opencv c3d_int8_1b.bmodel 0.715
function test_python(){
    pushd python
    echo "---------------------"
    echo "testing python $1 $2:"
    echo "---------------------"
    res=$(python3 c3d_$1.py --input_path ../data/UCF_test_01 --bmodel ../data/models/$TARGET/$2 --tpu_id $TPUID 2>&1 | tee $1_$2.log)
    acc=(${res#*ACC:})
    gt=$3
    compare_res $acc $gt
    popd
}

#test pipeline:
if test $MODE = "compile"
then
    download
    chmod +x ./scripts/gen_fp32bmodel.sh
    ./scripts/gen_fp32bmodel.sh $TARGET
    chmod +x ./scripts/gen_int8bmodel.sh
    ./scripts/gen_int8bmodel.sh $TARGET
elif test $MODE = "pcie_test"
then
    download
    test_python opencv c3d_fp32_1b.bmodel 0.7154
    test_python opencv c3d_fp32_4b.bmodel 0.7154
    [ $TARGET = "BM1684" ] && gt=0.6910 || gt=0.7154
    test_python opencv c3d_int8_1b.bmodel $gt
    test_python opencv c3d_int8_4b.bmodel $gt
    build_pcie opencv
    test_cpp opencv pcie c3d_fp32_1b.bmodel 0.7154
    test_cpp opencv pcie c3d_fp32_4b.bmodel 0.7154
    [ $TARGET = "BM1684" ] && gt=0.6910 || gt=0.7154
    test_cpp opencv pcie c3d_int8_1b.bmodel $gt
    test_cpp opencv pcie c3d_int8_4b.bmodel $gt
elif test $MODE = "soc_build"
then
    build_soc opencv
elif test $MODE = "soc_test"
then
    download
    test_python opencv c3d_fp32_1b.bmodel 0.7154
    test_python opencv c3d_fp32_4b.bmodel 0.7154
    [ $TARGET = "BM1684" ] && gt=0.6948 || gt=0.7154
    test_python opencv c3d_int8_1b.bmodel $gt
    test_python opencv c3d_int8_4b.bmodel $gt   
    test_cpp opencv soc c3d_fp32_1b.bmodel 0.7154
    test_cpp opencv soc c3d_fp32_4b.bmodel 0.7154
    [ $TARGET = "BM1684" ] && gt=0.6948 || gt=0.7154
    test_cpp opencv soc c3d_int8_1b.bmodel $gt
    test_cpp opencv soc c3d_int8_4b.bmodel $gt
fi

if [ $ALL_PASS -eq 0 ]
then
    echo "====================================================================="
    echo "Some process produced unexpected results, please look out their logs!"
    echo "====================================================================="
else
    echo "===================="
    echo "Test cases all pass!"
    echo "===================="
fi

popd
