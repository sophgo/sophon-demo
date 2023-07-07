#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684X"
MODE="pcie_test"
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

function download()
{
  chmod -R +x scripts/
  ./scripts/download.sh
  judge_ret $? "download" 0
}

function compile_nntc()
{
  ./scripts/gen_fp32bmodel_nntc.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  ./scripts/gen_int8bmodel_nntc.sh $TARGET
  judge_ret $? "generate $TARGET int8bmodel" 0
}

function compile_mlir()
{
  ./scripts/gen_fp32bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  ./scripts/gen_fp16bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp16bmodel" 0
  ./scripts/gen_int8bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET int8bmodel" 0
}

function build_pcie()
{
  pushd cpp/lprnet_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build lprnet_$1" 0
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
  judge_ret $? "build soc lprnet_$1" 0
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
function test_cpp()
{
  pushd cpp/lprnet_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./lprnet_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_cpp_test.log
  judge_ret $? "./lprnet_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID" log/$1_$2_$3_cpp_test.log
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/lprnet_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./lprnet_$2.$1 --input=../../datasets/test --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./lprnet_$2.$1 --input=../../datasets/test --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1" log/$1_$2_$3_debug.log
  tail -n 15 log/$1_$2_$3_debug.log
  
  echo "Evaluating..."
  res=$(python3 ../../tools/eval_ccpd.py --gt_path ../../datasets/test_label.json --result_json results/$3_test_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"

  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  judge_ret $? "$3_test_$2_cpp_result: Precision compare!" log/$1_$2_$3_eval.log
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function test_python()
{
  if [ ! -d log ];then
    mkdir log
  fi
  python3 python/lprnet_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_python_test.log
  judge_ret $? "python3 python/lprnet_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID" log/$1_$2_python_test.log
}

function eval_python()
{  
  echo -e "\n########################\nCase Start: eval python\n########################"
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  python3 python/lprnet_$1.py --input datasets/test --bmodel models/$TARGET/$2 --dev_id $TPUID > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/lprnet_$1.py --input datasets/test --bmodel models/$TARGET/$2 --dev_id $TPUID > python/log/$1_$2_debug.log 2>&1"  python/log/$1_$2_debug.log
  tail -n 15 python/log/$1_$2_debug.log
  
  echo "Evaluating..."
  res=$(python3 tools/eval_ccpd.py --gt_path datasets/test_label.json --result_json results/$2_test_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
  judge_ret $? "$2_test_$1_python_result: Precision compare!" python/log/$1_$2_eval.log
  echo -e "########################\nCase End: eval python\n########################\n"
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
    eval_python opencv lprnet_int8_1b.bmodel 0.858
    eval_python opencv lprnet_int8_4b.bmodel 0.881
    eval_python bmcv lprnet_fp32_1b.bmodel 0.88
    eval_python bmcv lprnet_int8_1b.bmodel 0.857
    eval_python bmcv lprnet_int8_4b.bmodel 0.865
    eval_cpp pcie opencv lprnet_fp32_1b.bmodel 0.88 
    eval_cpp pcie opencv lprnet_int8_1b.bmodel 0.857
    eval_cpp pcie opencv lprnet_int8_4b.bmodel 0.869
    eval_cpp pcie bmcv lprnet_fp32_1b.bmodel 0.88 
    eval_cpp pcie bmcv lprnet_int8_1b.bmodel 0.857
    eval_cpp pcie bmcv lprnet_int8_4b.bmodel 0.869

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
    eval_python bmcv lprnet_fp32_1b.bmodel 0.882
    eval_python bmcv lprnet_fp16_1b.bmodel 0.882
    eval_python bmcv lprnet_int8_1b.bmodel 0.861
    eval_python bmcv lprnet_int8_4b.bmodel 0.88
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
    eval_python opencv lprnet_int8_1b.bmodel 0.858
    eval_python opencv lprnet_int8_4b.bmodel 0.881
    eval_python bmcv lprnet_fp32_1b.bmodel 0.88
    eval_python bmcv lprnet_int8_1b.bmodel 0.857
    eval_python bmcv lprnet_int8_4b.bmodel 0.865
    eval_cpp soc opencv lprnet_fp32_1b.bmodel 0.88 
    eval_cpp soc opencv lprnet_int8_1b.bmodel 0.857
    eval_cpp soc opencv lprnet_int8_4b.bmodel 0.869
    eval_cpp soc bmcv lprnet_fp32_1b.bmodel 0.88 
    eval_cpp soc bmcv lprnet_int8_1b.bmodel 0.857
    eval_cpp soc bmcv lprnet_int8_4b.bmodel 0.869
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


    eval_python opencv lprnet_fp32_1b.bmodel 0.894
    eval_python opencv lprnet_fp16_1b.bmodel 0.894
    eval_python opencv lprnet_int8_1b.bmodel 0.867
    eval_python opencv lprnet_int8_4b.bmodel 0.88
    eval_python bmcv lprnet_fp32_1b.bmodel 0.882
    eval_python bmcv lprnet_fp16_1b.bmodel 0.882
    eval_python bmcv lprnet_int8_1b.bmodel 0.861
    eval_python bmcv lprnet_int8_4b.bmodel 0.88 
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