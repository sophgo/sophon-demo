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
export LD_LIBRARY_PATH=/opt/sophon/sophon-sail/lib:$LD_LIBRARY_PATH

usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [-a SAIL] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2 
}

while getopts ":m:t:s:a:d:p:" opt
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
    p)
      PYTEST=${OPTARG}
      echo "generate logs for $PYTEST";;
    ?)
      usage
      exit 1;;
  esac
done
if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi
if [ -f "scripts/acc.txt" ]; then
  rm scripts/acc.txt
fi
echo "|   测试平台    |      测试程序     |    测试模型        | ACC | F1 |" >> scripts/acc.txt
PLATFORM=$TARGET
if test $MODE = "soc_test"; then
  if test $TARGET = "BM1684X"; then
    PLATFORM="SE7-32"
  elif test $TARGET = "BM1684"; then
    PLATFORM="SE5-16"
  elif test $TARGET = "BM1688"; then
    PLATFORM="SE9-16"
  elif test $TARGET = "CV186X"; then
    PLATFORM="SE9-8"
  else
    echo "Unknown TARGET type: $TARGET"
  fi
fi
if test $PYTEST = "pytest"
then
  >${top_dir}auto_test_result.txt
fi

function bmrt_test_case(){
   calculate_time_log=$(bmrt_test --bmodel $1 --devid $TPUID | grep "calculate" 2>&1)
   is_8b=$(echo $1 |grep "8b")

   if [ "$is_8b" != "" ]; then
    readarray -t calculate_times < <(echo "$calculate_time_log" | grep -oP 'calculate  time\(s\): \K\d+\.\d+' | awk '{printf "%.2f \n", $1 * 125}')
   else
    readarray -t calculate_times < <(echo "$calculate_time_log" | grep -oP 'calculate  time\(s\): \K\d+\.\d+' | awk '{printf "%.2f \n", $1 * 1000}')
   fi
   for time in "${calculate_times[@]}"
   do
     printf "| %-35s| % 15s |\n" "$1" "$time"
   done
}
function bmrt_test_benchmark(){
    pushd models
    printf "| %-35s| % 15s |\n" "测试模型" "calculate time(ms)"
    printf "| %-35s| % 15s |\n" "-------------------" "--------------"
   
    if test $TARGET = "BM1684"; then
      bmrt_test_case BM1684/bert4torch_output_fp32_1b.bmodel
      bmrt_test_case BM1684/bert4torch_output_fp32_8b.bmodel
    elif test $TARGET = "BM1684X"; then
     bmrt_test_case BM1684/bert4torch_output_fp16_1b.bmodel
      bmrt_test_case BM1684X/bert4torch_output_fp32_1b.bmodel
      bmrt_test_case BM1684X/bert4torch_output_fp16_8b.bmodel
      bmrt_test_case BM1684X/bert4torch_output_fp32_8b.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/bert4torch_output_fp16_1b.bmodel
      bmrt_test_case BM1688/bert4torch_output_fp32_1b.bmodel
      bmrt_test_case BM1688/bert4torch_output_fp16_8b.bmodel
      bmrt_test_case BM1688/bert4torch_output_fp32_8b.bmodel
      bmrt_test_case BM1688/bert4torch_output_fp16_1b_2core.bmodel
      bmrt_test_case BM1688/bert4torch_output_fp32_1b_2core.bmodel
      bmrt_test_case BM1688/bert4torch_output_fp16_8b_2core.bmodel
      bmrt_test_case BM1688/bert4torch_output_fp32_8b_2core.bmodel
    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/bert4torch_output_fp16_1b.bmodel
      bmrt_test_case CV186X/bert4torch_output_fp32_1b.bmodel
      bmrt_test_case CV186X/bert4torch_output_fp16_8b.bmodel
      bmrt_test_case CV186X/bert4torch_output_fp32_8b.bmodel
   
    fi
    popd
}

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
   if [ ! -d log ];then
    mkdir log
  fi
  ./bert_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID --dict_path=../../models/pre_train/chinese-bert-wwm/vocab.txt> log/$1_$2_$3_cpp_test.log 2>&1
  judge_ret $? "./bert_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID --dict_path=../../models/pre_train/chinese-bert-wwm/vocab.txt" log/$1_$2_$3_cpp_test.log
  
  tail -n 15 log/$1_$2_$3_cpp_test.log
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
  ./bert_$2.$1 --input=../../datasets/china-people-daily-ner-corpus/example.test --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID --dict_path=../../models/pre_train/chinese-bert-wwm/vocab.txt > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./bert_$2.$1 --input=../../datasets/china-people-daily-ner-corpus/example.test --bmodel=../../models/$TARGET/$3  --dev_id=$TPUID --dict_path=../../models/pre_train/chinese-bert-wwm/vocab.txt > log/$1_$2_$3_debug.log 2>&1"
  tail -n 15 log/$1_$2_$3_debug.log

  echo "Evaluating..."  
  res=$(python3 ../../tools/eval_people.py  --test_path ../datasets/china-people-daily-ner-corpus/example.test --input_path ../cpp/bert_sail/results/$3_test_$2_cpp_result.txt 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  f1=${array[7]}
  compare_res $f1 $3
  printf "| %-12s | %-14s | %-22s | %8.4f | %8.4f |\n" "$PLATFORM" "bert_$2.$1" "$3" "$(printf "%.4f" $f1)" "$(printf "%.4f" $acc)" >> ../../scripts/acc.txt

  echo -e "########################\nCase End: eval cpp\n########################\n"

    echo "==================="
    echo "Comparing statis..."
    python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=bert_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3
    judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=bert_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3"
    echo "==================="

  popd

}

function test_python()
{
  if [ ! -d log ];then
    mkdir log
  fi
  pushd python

  python3 bert_$1.py --input $3 --bmodel ../models/$TARGET/$2 --dev_id $TPUID --dict_path ../models/pre_train/chinese-bert-wwm/vocab.txt > log/$1_$2_python_test.log 2>&1
  judge_ret $? "python3 bert_$1.py --input $3 --bmodel ../models/$TARGET/$2 --dev_id $TPUID --dict_path ../models/pre_train/chinese-bert-wwm/vocab.txt > log/$1_$2_debug.log 2>&1"

  tail -n 20 log/$1_$2_python_test.log
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
  python3 bert_$1.py --input=../datasets/china-people-daily-ner-corpus/example.test --bmodel ../models/$TARGET/$2 --dev_id $TPUID --dict_path ../models/pre_train/chinese-bert-wwm/vocab.txt > log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 bert_$1.py --input=../datasets/china-people-daily-ner-corpus/example.test --bmodel ../models/$TARGET/$2 --dev_id $TPUID --dict_path ../models/pre_train/chinese-bert-wwm/vocab.txt > log/$1_$2_debug.log 2>&1"
  tail -n 20 log/$1_$2_debug.log
  
  echo "Evaluating..."
  res=$(python3 ../tools/eval_people.py --test_path ../datasets/china-people-daily-ner-corpus/example.test --input_path ../python/results/$2_$1_python_result.txt 2>&1 | tee log/$1_$2_eval.log)

  echo -e "$res"

  array=(${res//=/ })
  acc=${array[1]}
  f1=${array[7]}
  compare_res $f1 $3
  printf "| %-12s | %-14s | %-22s | %8.4f | %8.4f |\n" "$PLATFORM" "bert_$1.py" "$2" "$(printf "%.4f" $f1)" "$(printf "%.4f" $acc)" >> ../scripts/acc.txt
  popd
  echo -e "########################\nCase End: eval python\n########################\n"

    echo "==================="
    echo "Comparing statis..."
    python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=bert_$1.py --language=python --input=python/log/$1_$2_debug.log --bmodel=$2
    judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=bert_$1.py --language=python --input=python/log/$1_$2_debug.log --bmodel=$2"
    echo "==================="
  

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
    eval_cpp soc sail bert4torch_output_fp16_1b_2core.bmodel 0.8220128904313336
    eval_cpp soc sail bert4torch_output_fp16_8b_2core.bmodel 0.9201187249967738
  elif test $TARGET = "CV186X"
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
  fi
fi
if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------BERT ACC----------"
  cat scripts/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------BERT performance-----------"
  cat tools/benchmark.txt
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