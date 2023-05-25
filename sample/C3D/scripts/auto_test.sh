#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684X"
MODE="pcie_test"
SOCSDK="/home/lihengfang/work/sophonsdk/soc-sdk"
TPUID=0
ALL_PASS=1
PYTEST="auto_test"
ECHO_LINES=20

usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2 
}

while getopts ":m:t:s:d:p:" opt
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
    p)
      PYTEST=${OPTARG}
      echo "generate logs for $PYTEST";;
    ?)
      usage
      exit 1;;
  esac
done

if test $PYTEST = "pytest"
then
  >${top_dir}auto_test_result.txt
fi

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
    if test $PYTEST = "pytest"
    then
      echo "Passed: $2" >> ${top_dir}auto_test_result.txt
      echo "#######Debug Info Start#######" >> ${top_dir}auto_test_result.txt
    fi
  else
    echo "Failed: $2"
    ALL_PASS=0
    if test $PYTEST = "pytest"
    then
      echo "Failed: $2" >> ${top_dir}auto_test_result.txt
      echo "#######Debug Info Start#######" >> ${top_dir}auto_test_result.txt
    fi
  fi

  if test $PYTEST = "pytest"
  then
    if [[ $3 != 0 ]];then
      tail -n ${ECHO_LINES} $3 >> ${top_dir}auto_test_result.txt
    fi
    echo "########Debug Info End########" >> ${top_dir}auto_test_result.txt
  fi

  sleep 3
}

function download(){
    #download dataset and models.
    if [ ! -d './datasets/UCF_test_01' -o ! -d './models' ]; then
        echo "preparing datasets and models......"
        chmod +x ./scripts/download.sh
        ./scripts/download.sh
        judge_ret $? "download" 0
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
    judge_ret $? "build c3d_$1" 0
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
    judge_ret $? "build soc c3d_$1" 0
    popd
}

function compare_res(){
    ret=`awk -v x=$1 -v y=$2 'BEGIN{print(x-y<0.001 && y-x<0.001)?1:0}'`
    if [ $ret -eq 0 ]
    then
        ALL_PASS=0
        echo "***************************************"
        echo "Ground truth is $2, your result is: $1"
        echo -e "\e[41m compare wrong! \e[0m" #red
        echo "***************************************"
        return 1
    else
        echo "***************************************"
        echo -e "\e[42m compare right! \e[0m" #green
        echo "***************************************"
        return 0
    fi
}
#e.g.: test_cpp opencv pcie c3d_int8_1b.bmodel 0.715
function test_cpp(){
    echo -e "\n########################\nCase Start: eval cpp\n########################"
    pushd cpp/c3d_$1
    if [ ! -d log ];then
        mkdir log
    fi
    echo "testing cpp $1 $3:"
    chmod +x ./c3d_$1.$2
    ./c3d_$1.$2 --input=../../datasets/UCF_test_01 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$3.log 2>&1
    judge_ret $? "./c3d_$1.$2 --input=../../datasets/UCF_test_01 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$3.log 2>&1" log/$1_$3.log
    tail -n 15 log/$1_$3.log
    echo "Evaluating..."
    res=$(python3 ../../tools/eval_ucf.py --result_json results/$3_$1_cpp.json --gt_path ../../datasets/ground_truth.json 2>&1 | tee log/$1_$2_$3_eval.log)
    echo -e "$res"
    array=(${res//=/ })
    acc=${array[1]}
    compare_res $acc $4
    judge_ret $? "$3_$1_cpp: Precision compare!" log/$1_$2_$3_eval.log
    popd
    echo -e "########################\nCase End: eval cpp\n########################\n"
}

#e.g.: test_python opencv c3d_int8_1b.bmodel 0.715
function test_python(){
    echo -e "\n########################\nCase Start: eval python\n########################"
    pushd python
    if [ ! -d log ];then
        mkdir log
    fi
    echo "testing python $1 $2:"
    python3 c3d_$1.py --input ../datasets/UCF_test_01 --bmodel ../models/$TARGET/$2 --dev_id $TPUID > log/$1_$2.log 2>&1
    judge_ret $? "python3 c3d_$1.py --input ../datasets/UCF_test_01 --bmodel ../models/$TARGET/$2 --dev_id $TPUID > log/$1_$2.log 2>&1" log/$1_$2.log
    tail -n 15 log/$1_$2.log 
    echo "Evaluating..."
    res=$(python3 ../tools/eval_ucf.py --result_json results/$2_$1_python.json --gt_path ../datasets/ground_truth.json 2>&1 | tee log/$1_$2_eval.log)
    echo -e "$res"
    array=(${res//=/ })
    acc=${array[1]}
    compare_res $acc $3
    judge_ret $? "$2_$1_python: Precision compare!" log/$1_$2_eval.log
    popd
    echo -e "########################\nCase End: eval python\n########################\n"
}

#test pipeline:
if test $MODE = "compile_nntc"
then
    download
    chmod +x ./scripts/gen_fp32bmodel_nntc.sh
    ./scripts/gen_fp32bmodel_nntc.sh 
    chmod +x ./scripts/gen_int8bmodel_nntc.sh
    ./scripts/gen_int8bmodel_nntc.sh 
elif test $MODE = "compile_mlir"
then
    download
    chmod +x ./scripts/gen_fp32bmodel_mlir.sh
    ./scripts/gen_fp32bmodel_mlir.sh
    chmod +x ./scripts/gen_fp16bmodel_mlir.sh
    ./scripts/gen_fp16bmodel_mlir.sh 
    chmod +x ./scripts/gen_int8bmodel_mlir.sh
    ./scripts/gen_int8bmodel_mlir.sh  
elif test $MODE = "pcie_test"
then
    download
    test_python opencv c3d_fp32_1b.bmodel 0.715356
    test_python opencv c3d_fp32_4b.bmodel 0.715356
    [ $TARGET = "BM1684" ] && gt=0.6910 || gt=0.7097378
    test_python opencv c3d_int8_1b.bmodel $gt
    test_python opencv c3d_int8_4b.bmodel $gt
    build_pcie opencv
    test_cpp opencv pcie c3d_fp32_1b.bmodel 0.715356
    test_cpp opencv pcie c3d_fp32_4b.bmodel 0.715356
    [ $TARGET = "BM1684" ] && gt=0.6910 || gt=0.7097378
    test_cpp opencv pcie c3d_int8_1b.bmodel $gt
    test_cpp opencv pcie c3d_int8_4b.bmodel $gt
    if test $TARGET = "BM1684X"
    then
        test_cpp opencv pcie c3d_fp16_1b.bmodel 0.715356
        test_cpp opencv pcie c3d_fp16_4b.bmodel 0.715356
        test_python opencv c3d_fp16_1b.bmodel 0.715356
        test_python opencv c3d_fp16_4b.bmodel 0.715356
    fi
    #############################################
    build_pcie bmcv
    test_cpp bmcv pcie c3d_fp32_1b.bmodel 0.715356
    test_cpp bmcv pcie c3d_fp32_4b.bmodel 0.715356
    [ $TARGET = "BM1684" ] && gt=0.692884 || gt=0.713483
    test_cpp bmcv pcie c3d_int8_1b.bmodel $gt
    test_cpp bmcv pcie c3d_int8_4b.bmodel $gt
    if test $TARGET = "BM1684X"
    then
        test_cpp bmcv pcie c3d_fp16_1b.bmodel 0.715356
        test_cpp bmcv pcie c3d_fp16_4b.bmodel 0.715356
    fi
elif test $MODE = "soc_build"
then
    build_soc opencv
    build_soc bmcv
elif test $MODE = "soc_test"
then
    download
    test_python opencv c3d_fp32_1b.bmodel 0.715356
    test_python opencv c3d_fp32_4b.bmodel 0.715356
    [ $TARGET = "BM1684" ] && gt=0.6910 || gt=0.711610
    test_python opencv c3d_int8_1b.bmodel $gt
    test_python opencv c3d_int8_4b.bmodel $gt
    test_cpp opencv soc c3d_fp32_1b.bmodel 0.715356
    test_cpp opencv soc c3d_fp32_4b.bmodel 0.715356
    [ $TARGET = "BM1684" ] && gt=0.6910 || gt=0.711610
    test_cpp opencv soc c3d_int8_1b.bmodel $gt
    test_cpp opencv soc c3d_int8_4b.bmodel $gt
    if test $TARGET = "BM1684X"
    then
        test_cpp opencv soc c3d_fp16_1b.bmodel 0.715356
        test_cpp opencv soc c3d_fp16_4b.bmodel 0.715356
        test_python opencv c3d_fp16_1b.bmodel 0.715356
        test_python opencv c3d_fp16_4b.bmodel 0.715356
    fi
    #############################################
    test_cpp bmcv soc c3d_fp32_1b.bmodel 0.715356
    test_cpp bmcv soc c3d_fp32_4b.bmodel 0.715356
    [ $TARGET = "BM1684" ] && gt=0.692884 || gt=0.711610
    test_cpp bmcv soc c3d_int8_1b.bmodel $gt
    test_cpp bmcv soc c3d_int8_4b.bmodel $gt
    if test $TARGET = "BM1684X"
    then
        test_cpp bmcv soc c3d_fp16_1b.bmodel 0.715356
        test_cpp bmcv soc c3d_fp16_4b.bmodel 0.715356
    fi
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
