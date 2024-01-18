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


function judge_ret()
{
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
  pushd cpp/bert_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build bert_$1"
  popd
  ls
}

function build_soc()
{
  pushd cpp/bert_$1
  if [ -d build ]; then
      rm -rf build
  fi
  if test $1 = "sail"; then
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK -DSAIL_PATH=$SAIL_PATH && make
    judge_ret $? "build soc bert_$1"
  else
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
    judge_ret $? "build soc bert_$1"
  fi
  popd
}

function compare_res(){
    ret=`awk -v x=$1 -v y=$2 'BEGIN{print(x-y<0.1 && y-x<0.1)?1:0}'`
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
  pushd cpp/bert_$2
  ./bert_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID
  judge_ret $? "./bert_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID"
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"

  pushd cpp/bert_$2
  if [ ! -d log ];then
    mkdir log
  fi
  if [ ! -d results ];then
    mkdir results
  fi
  ./bert_$2.$1 --input=../../datasets/china-people-daily-ner-corpus/example.test --bmodel=../../models/$TARGET/$3 --dev_id $TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./bert_$2.$1 --input=../../datasets/china-people-daily-ner-corpus/example.test --bmodel=../../models/$TARGET/$3  --dev_id $TPUID > log/$1_$2_$3_debug.log 2>&1"
  tail -n 15 log/$1_$2_$3_debug.log

  echo "Evaluating..."  
  res=$(python3 ../../tools/eval_people.py  --test_path ../datasets/china-people-daily-ner-corpus/example.test --input_path ../cpp/bert_sail/results/$3_test_$2_cpp_result.txt 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[7]}
  compare_res $acc $4
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"

}

function test_python()
{
  pushd python

  python3 bert_$1.py --input $3 --bmodel ../models/$TARGET/$2 --dev_id $TPUID
  judge_ret $? "python3 bert_$1.py --input $3 --bmodel ../models/$TARGET/$2 --dev_id $TPUID"
  popd
}

function eval_python()
{  
  echo -e "\n########################\nCase Start: eval python\n########################"

  if [ ! -d python/log ];then
    mkdir python/log
  fi
  if [ ! -d python/results ];then
    mkdir python/results
  fi
  pushd python
  python3 bert_$1.py --input=../datasets/china-people-daily-ner-corpus/example.test --bmodel ../models/$TARGET/$2 --dev_id $TPUID  > log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 bert_$1.py --input=../datasets/china-people-daily-ner-corpus/example.test --bmodel ../models/$TARGET/$2 --dev_id $TPUID  > log/$1_$2_debug.log 2>&1"
  tail -n 20 python/log/$1_$2_debug.log
  
  echo "Evaluating..."
  res=$(python3 ../tools/eval_people.py --test_path ../datasets/china-people-daily-ner-corpus/example.test --input_path ../python/results/$2_$1_python_result.txt 2>&1 | tee log/$1_$2_eval.log)
  echo -e "$res"

  array=(${res//=/ })
  acc=${array[7]}
  compare_res $acc $3
  popd
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
  build_pcie sail
  download
  if test $TARGET = "BM1684"
  then

    test_python sail bert4torch_output_fp32_1b.bmodel ../datasets/china-people-daily-ner-corpus/test.txt
    test_cpp pcie sail bert4torch_output_fp32_1b.bmodel ../../datasets/china-people-daily-ner-corpus/test.txt
    eval_python sail bert4torch_output_fp32_1b.bmodel 0.9183410613086039
    eval_python sail bert4torch_output_fp32_8b.bmodel 0.9201187249967738
    eval_cpp pcie sail bert4torch_output_fp32_1b.bmodel 0.912984583628975
    eval_cpp pcie sail bert4torch_output_fp32_8b.bmodel 0.912984583628975
   

  elif test $TARGET = "BM1684X"
  then
    test_python sail bert4torch_output_fp32_1b.bmodel ../datasets/china-people-daily-ner-corpus/test.txt
    test_cpp pcie sail bert4torch_output_fp32_1b.bmodel ../../datasets/china-people-daily-ner-corpus/test.txt
    test_python sail bert4torch_output_fp16_1b.bmodel ../datasets/china-people-daily-ner-corpus/test.txt
    test_cpp pcie sail bert4torch_output_fp16_1b.bmodel ../../datasets/china-people-daily-ner-corpus/test.txt
    eval_python sail bert4torch_output_fp32_1b.bmodel 0.9183410613086039
    eval_python sail bert4torch_output_fp32_8b.bmodel 0.9201187249967738
    eval_python sail bert4torch_output_fp16_1b.bmodel 0.9183410613086039
    eval_python sail bert4torch_output_fp16_8b.bmodel 0.9201187249967738
    eval_cpp pcie sail bert4torch_output_fp32_1b.bmodel 0.912984583628975
    eval_cpp pcie sail bert4torch_output_fp32_8b.bmodel 0.912984583628975
    eval_cpp pcie sail bert4torch_output_fp16_1b.bmodel 0.9183410613086039
    eval_cpp pcie sail bert4torch_output_fp16_8b.bmodel 0.9201187249967738
  fi
elif test $MODE = "soc_build"
then
  build_soc sail
elif test $MODE = "soc_test"
then
  download
  if test $TARGET = "BM1684"
  then
    test_python sail bert4torch_output_fp32_1b.bmodel ../datasets/china-people-daily-ner-corpus/test.txt
    test_cpp soc sail bert4torch_output_fp32_1b.bmodel ../../datasets/china-people-daily-ner-corpus/test.txt
    eval_python sail bert4torch_output_fp32_1b.bmodel 0.9183410613086039
    eval_python sail bert4torch_output_fp32_8b.bmodel 0.9201187249967738
    eval_cpp soc sail bert4torch_output_fp32_1b.bmodel 0.912984583628975
    eval_cpp soc sail bert4torch_output_fp32_8b.bmodel 0.912984583628975
  elif test $TARGET = "BM1684X"
  then
    test_python sail bert4torch_output_fp32_1b.bmodel ../datasets/china-people-daily-ner-corpus/test.txt
    test_cpp soc sail bert4torch_output_fp32_1b.bmodel ../../datasets/china-people-daily-ner-corpus/test.txt
    test_python sail bert4torch_output_fp16_1b.bmodel ../datasets/china-people-daily-ner-corpus/test.txt
    test_cpp soc sail bert4torch_output_fp16_1b.bmodel ../../datasets/china-people-daily-ner-corpus/test.txt
    eval_python sail bert4torch_output_fp32_1b.bmodel 0.9183410613086039
    eval_python sail bert4torch_output_fp32_8b.bmodel 0.9201187249967738
    eval_python sail bert4torch_output_fp16_1b.bmodel 0.9183410613086039
    eval_python sail bert4torch_output_fp16_8b.bmodel 0.9201187249967738
    eval_cpp soc sail bert4torch_output_fp32_1b.bmodel 0.912984583628975
    eval_cpp soc sail bert4torch_output_fp32_8b.bmodel 0.912984583628975
    eval_cpp soc sail bert4torch_output_fp16_1b.bmodel 0.9183410613086039
    eval_cpp soc sail bert4torch_output_fp16_8b.bmodel 0.9201187249967738
  fi
  elif test $TARGET = "BM1688"
  then
    test_python sail bert4torch_output_fp32_1b.bmodel ../datasets/china-people-daily-ner-corpus/test.txt
    test_cpp soc sail bert4torch_output_fp32_1b.bmodel ../../datasets/china-people-daily-ner-corpus/test.txt
    test_python sail bert4torch_output_fp16_1b.bmodel ../datasets/china-people-daily-ner-corpus/test.txt
    test_cpp soc sail bert4torch_output_fp16_1b.bmodel ../../datasets/china-people-daily-ner-corpus/test.txt
    eval_python sail bert4torch_output_fp32_1b.bmodel 0.9183410613086039
    eval_python sail bert4torch_output_fp32_8b.bmodel 0.9201187249967738
    eval_python sail bert4torch_output_fp16_1b.bmodel 0.9183410613086039
    eval_python sail bert4torch_output_fp16_8b.bmodel 0.9201187249967738
    eval_cpp soc sail bert4torch_output_fp32_1b.bmodel 0.912984583628975
    eval_cpp soc sail bert4torch_output_fp32_8b.bmodel 0.912984583628975
    eval_cpp soc sail bert4torch_output_fp16_1b.bmodel 0.8931268398822476
    eval_cpp soc sail bert4torch_output_fp16_8b.bmodel 0.9201187249967738
    eval_python sail bert4torch_output_fp32_1b_2core.bmodel 0.9183410613086039
    eval_python sail bert4torch_output_fp32_8b_2core.bmodel 0.9201187249967738
    eval_python sail bert4torch_output_fp16_1b_2core.bmodel 0.9183410613086039
    eval_python sail bert4torch_output_fp16_8b_2core.bmodel 0.9201187249967738
    eval_cpp soc sail bert4torch_output_fp32_1b_2core.bmodel 0.912984583628975
    eval_cpp soc sail bert4torch_output_fp32_8b_2core.bmodel 0.912984583628975
    # eval_cpp soc sail bert4torch_output_fp16_1b_2core.bmodel 0.8220128904313336
    eval_cpp soc sail bert4torch_output_fp16_8b_2core.bmodel 0.9201187249967738
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