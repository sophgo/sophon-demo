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

function compile_mlir()
{
  ./scripts/gen_fp32bmodel_mlir.sh bm1684x
  judge_ret $? "generate BM1684X fp32bmodel"
  ./scripts/gen_fp16bmodel_mlir.sh bm1684x
  judge_ret $? "generate BM1684X fp16bmodel"
  ./scripts/gen_int8bmodel_mlir.sh bm1684x
  judge_ret $? "generate BM1684X int8bmodel"
}

function build_pcie()
{
  pushd cpp/lprnet_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build lprnet_$1"
  popd
}

function build_soc()
{
  pushd cpp/lprnet_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc lprnet_$1"
  popd
}

function compare_res(){
    ret=`awk -v x=$1 -v y=$2 'BEGIN{print(x-y<0.0001 && y-x<0.0001)?1:0}'`
    if [ $ret -eq 0 ]
    then
        ALL_PASS=0
        echo "***************************************"
        echo "Ground truth is $2, your result is: $1"
        echo -e "\e[41m compare wrong! \e[0m" #red
        echo "***************************************"
    else
        echo "***************************************"
        echo -e "\e[42m compare right! \e[0m" #green
        echo "***************************************"
    fi
}
function test_cpp()
{
  pushd cpp/lprnet_$2
  ./lprnet_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID
  judge_ret $? "./lprnet_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID"
  popd
}

function eval_cpp()
{
  pushd cpp/lprnet_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./lprnet_$2.$1 --input=../../datasets/test --bmodel=../../models/$TARGET/$3 --dev_id $TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./lprnet_$2.$1 --input=../../datasets/test --bmodel=../../models/$TARGET/$3 --dev_id $TPUID > log/$1_$2_$3_debug.log 2>&1"
  echo "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
  echo "python3 ../../tools/eval_ccpd.py --gt_path ../../datasets/test_label.json --result_json results/$3_test_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log"
  echo "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
  res=$(python3 ../../tools/eval_ccpd.py --gt_path ../../datasets/test_label.json --result_json results/$3_test_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log)
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  popd
}

function test_python()
{
  python3 python/lprnet_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID
  judge_ret $? "python3 python/lprnet_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID"
}

function eval_python()
{  
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  python3 python/lprnet_$1.py --input datasets/test --bmodel models/$TARGET/$2 --dev_id $TPUID > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/lprnet_$1.py --input datasets/test --bmodel models/$TARGET/$2 --dev_id $TPUID > python/log/$1_$2_debug.log 2>&1"
  echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
  echo "python3 tools/eval_ccpd.py --gt_path datasets/test_label.json --result_json results/$2_test_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log"
  echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
  res=$(python3 tools/eval_ccpd.py --gt_path datasets/test_label.json --result_json results/$2_test_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3

}

if test $MODE = "compile_nntc"
then
  download
  compile_nntc
elif test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_test"
then
  build_pcie bmcv
  build_pcie opencv
  download
  if test $TARGET = "BM1684"
  then
    test_python opencv lprnet_fp32_1b.bmodel datasets/test
    test_python opencv lprnet_int8_4b.bmodel datasets/test
    test_python bmcv lprnet_fp32_1b.bmodel datasets/test
    test_python bmcv lprnet_int8_4b.bmodel datasets/test
    test_cpp pcie opencv lprnet_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie opencv lprnet_int8_4b.bmodel ../../datasets/test
    test_cpp pcie bmcv lprnet_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv lprnet_int8_4b.bmodel ../../datasets/test


    eval_python opencv lprnet_fp32_1b.bmodel 0.894
    eval_python opencv lprnet_int8_1b.bmodel 0.887
    eval_python opencv lprnet_int8_4b.bmodel 0.898
    eval_python bmcv lprnet_fp32_1b.bmodel 0.879
    eval_python bmcv lprnet_int8_1b.bmodel 0.876
    eval_python bmcv lprnet_int8_4b.bmodel 0.878
    eval_cpp pcie opencv lprnet_fp32_1b.bmodel 0.882
    eval_cpp pcie opencv lprnet_int8_1b.bmodel 0.874
    eval_cpp pcie opencv lprnet_int8_4b.bmodel 0.879
    eval_cpp pcie bmcv lprnet_fp32_1b.bmodel 0.882
    eval_cpp pcie bmcv lprnet_int8_1b.bmodel 0.874
    eval_cpp pcie bmcv lprnet_int8_4b.bmodel 0.879

  elif test $TARGET = "BM1684X"
  then
    test_python opencv lprnet_fp32_1b.bmodel datasets/test
    test_python opencv lprnet_int8_4b.bmodel datasets/test
    test_python bmcv lprnet_fp32_1b.bmodel datasets/test
    test_python bmcv lprnet_int8_4b.bmodel datasets/test
    test_cpp pcie opencv lprnet_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie opencv lprnet_int8_4b.bmodel ../../datasets/test
    test_cpp pcie bmcv lprnet_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv lprnet_int8_4b.bmodel ../../datasets/test


    eval_python opencv lprnet_fp32_1b.bmodel 0.894
    eval_python opencv lprnet_fp16_1b.bmodel 0.894
    eval_python opencv lprnet_int8_1b.bmodel 0.867
    eval_python opencv lprnet_int8_4b.bmodel 0.88
    eval_python bmcv lprnet_fp32_1b.bmodel 0.879
    eval_python bmcv lprnet_fp16_1b.bmodel 0.879
    eval_python bmcv lprnet_int8_1b.bmodel 0.866
    eval_python bmcv lprnet_int8_4b.bmodel 0.862
    eval_cpp pcie opencv lprnet_fp32_1b.bmodel 0.882
    eval_cpp pcie opencv lprnet_fp16_1b.bmodel 0.882
    eval_cpp pcie opencv lprnet_int8_1b.bmodel 0.861
    eval_cpp pcie opencv lprnet_int8_4b.bmodel 0.872
    eval_cpp pcie bmcv lprnet_fp32_1b.bmodel 0.882
    eval_cpp pcie bmcv lprnet_fp16_1b.bmodel 0.882
    eval_cpp pcie bmcv lprnet_int8_1b.bmodel 0.861
    eval_cpp pcie bmcv lprnet_int8_4b.bmodel 0.872
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
  build_soc opencv
elif test $MODE = "soc_test"
then
  download
  if test $TARGET = "BM1684"
  then
    test_python opencv lprnet_fp32_1b.bmodel datasets/test
    test_python opencv lprnet_int8_4b.bmodel datasets/test
    test_python bmcv lprnet_fp32_1b.bmodel datasets/test
    test_python bmcv lprnet_int8_4b.bmodel datasets/test
    test_cpp soc opencv lprnet_fp32_1b.bmodel ../../datasets/test
    test_cpp soc opencv lprnet_int8_4b.bmodel ../../datasets/test
    test_cpp soc bmcv lprnet_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv lprnet_int8_4b.bmodel ../../datasets/test

    eval_python opencv lprnet_fp32_1b.bmodel 0.894
    eval_python opencv lprnet_int8_1b.bmodel 0.887
    eval_python opencv lprnet_int8_4b.bmodel 0.898
    eval_python bmcv lprnet_fp32_1b.bmodel 0.882
    eval_python bmcv lprnet_int8_1b.bmodel 0.871
    eval_python bmcv lprnet_int8_4b.bmodel 0.878
    eval_cpp soc opencv lprnet_fp32_1b.bmodel 0.88 
    eval_cpp soc opencv lprnet_int8_1b.bmodel 0.873
    eval_cpp soc opencv lprnet_int8_4b.bmodel 0.884
    eval_cpp soc bmcv lprnet_fp32_1b.bmodel 0.88 
    eval_cpp soc bmcv lprnet_int8_1b.bmodel 0.873
    eval_cpp soc bmcv lprnet_int8_4b.bmodel 0.884
  elif test $TARGET = "BM1684X"
  then
    test_python opencv lprnet_fp32_1b.bmodel datasets/test
    test_python opencv lprnet_int8_4b.bmodel datasets/test
    test_python bmcv lprnet_fp32_1b.bmodel datasets/test
    test_python bmcv lprnet_int8_4b.bmodel datasets/test
    test_cpp soc opencv lprnet_fp32_1b.bmodel ../../datasets/test
    test_cpp soc opencv lprnet_int8_4b.bmodel ../../datasets/test
    test_cpp soc bmcv lprnet_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv lprnet_int8_4b.bmodel ../../datasets/test


    eval_python opencv lprnet_fp32_1b.bmodel 0.882
    eval_python opencv lprnet_fp16_1b.bmodel 0.882
    eval_python opencv lprnet_int8_1b.bmodel 0.861
    eval_python opencv lprnet_int8_4b.bmodel 0.88
    eval_python bmcv lprnet_fp32_1b.bmodel 0.879
    eval_python bmcv lprnet_fp16_1b.bmodel 0.879
    eval_python bmcv lprnet_int8_1b.bmodel 0.866
    eval_python bmcv lprnet_int8_4b.bmodel 0.862 
    eval_cpp soc opencv lprnet_fp32_1b.bmodel 0.882
    eval_cpp soc opencv lprnet_fp16_1b.bmodel 0.882
    eval_cpp soc opencv lprnet_int8_1b.bmodel 0.861
    eval_cpp soc opencv lprnet_int8_4b.bmodel 0.872
    eval_cpp soc bmcv lprnet_fp32_1b.bmodel 0.882
    eval_cpp soc bmcv lprnet_fp16_1b.bmodel 0.882
    eval_cpp soc bmcv lprnet_int8_1b.bmodel 0.861
    eval_cpp soc bmcv lprnet_int8_4b.bmodel 0.872
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