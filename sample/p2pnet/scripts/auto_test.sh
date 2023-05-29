#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684X"
MODE="pcie_test"
TPUID=0
ALL_PASS=1

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

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
  else
    echo "Failed: $2"
    ALL_PASS=0
  fi
  sleep 3
}

function download()
{
  chmod -R +x scripts/
  ./scripts/download.sh
  judge_ret $? "download"
}

function compile_nntc()
{
  ./scripts/gen_fp32bmodel_nntc.sh BM1684
  judge_ret $? "generate BM1684 fp32bmodel"
  ./scripts/gen_int8bmodel_nntc.sh BM1684
  judge_ret $? "generate BM1684 int8bmodel"
}

function build_pcie()
{
  pushd cpp/yolact_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build yolact_$1"
  popd
}

function build_soc()
{
  pushd cpp/yolact_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc yolact_$1"
  popd
}

function compare_res()
{
    ret=`awk -v x=$1 -v y=$2 'BEGIN{print(y-x<0.01)?1:0}'`
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

function test_cpp()
{
  pushd cpp/p2pnet_$2
  ./p2pnet_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID
  judge_ret $? "./p2pnet_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID"
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/p2pnet_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./p2pnet_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID
  judge_ret $? "./p2pnet_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID"
  echo "Evaluating..."
  res=$(python3 ../../tools/eval_SHTech.py --gt_path ../../datasets/ShanghaiTech/part_A/test_data/ground-truth --result_json results/result_$2 2>&1 | tee log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  echo -e "########################\nCase End: eval cpp\n########################\n"
  popd
}

function test_python()
{
  python3 python/p2pnet_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID
  judge_ret $? "python3 python/p2pnet_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID"
}

function eval_python()
{  
  echo -e "\n########################\nCase Start: eval python\n########################"
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  python3 python/p2pnet_$1.py --input $4 --model models/$TARGET/$2 --dev_id $TPUID > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/p2pnet_$1.py --input $4 --bmodel models/$TARGET/$2 --dev_id $TPUID > python/log/$1_$2_debug.log 2>&1"
  echo "Evaluating..."
  res=$(python3 tools/eval_SHTech.py --gt_path ../../datasets/ShanghaiTech/part_A/test_data/ground-truth --result_json results/result_$2 2>&1 | tee log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
  echo -e "########################\nCase End: eval python\n########################\n"
}

if test $MODE = "compile_nntc"
then
  download
  compile_nntc
elif test $MODE = "pcie_test"
then
  build_pcie bmcv
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  if test $TARGET = "BM1684"
  then
    test_python bmcv p2pnet_bm1684_fp32_1b.bmodel datasets/ShanghaiTech/part_A/test_data/images
    test_python bmcv p2pnet_bm1684_int8_1b.bmodel datasets/ShanghaiTech/part_A/test_data/images
    test_cpp soc bmcv p2pnet_bm1684_fp32_1b.bmodel datasets/ShanghaiTech/part_A/test_data/images
    test_cpp soc bmcv p2pnet_bm1684_int8_1b.bmodel datasets/ShanghaiTech/part_A/test_data/images
   
    eval_python opencv p2pnet_bm1684_fp32_1b.bmodel 84
    eval_python opencv p2pnet_bm1684_int8_1b.bmodel 84
    eval_python bmcv p2pnet_bm1684_fp32_1b.bmodel 96
    eval_python bmcv p2pnet_bm1684_int8_1b.bmodel 94
    eval_cpp soc bmcv p2pnet_bm1684_fp32_1b.bmodel 81
    eval_cpp soc bmcv p2pnet_bm1684_int8_1b.bmodel 113
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
