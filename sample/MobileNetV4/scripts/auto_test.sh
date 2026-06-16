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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
CASE_MODE="fully"

usage()
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_build|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X|BM1688|CV186X] [ -s SOCSDK] [ -d TPUID] [ -p PYTEST auto_test|pytest] [ -c CASE_MODE fully|partly]" 1>&2
}

while getopts ":m:t:s:a:d:p:c:" opt
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
    c)
      CASE_MODE=${OPTARG}
      echo "case mode is $CASE_MODE";;
    ?)
      usage
      exit 1;;
  esac
done

if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi
if [ -f "tools/acc.txt" ]; then
  rm tools/acc.txt
fi

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

function bmrt_test_case(){
   calculate_time_log=$(bmrt_test --bmodel $1 --devid $TPUID | grep "calculate" 2>&1)
   is_4b=$(echo $1 |grep "4b")

   if [ "$is_4b" != "" ]; then
    readarray -t calculate_times < <(echo "$calculate_time_log" | grep -oP 'calculate  time\(s\): \K\d+\.\d+' | awk '{printf "%.2f \n", $1 * 250}')
   else
    readarray -t calculate_times < <(echo "$calculate_time_log" | grep -oP 'calculate  time\(s\): \K\d+\.\d+' | awk '{printf "%.2f \n", $1 * 1000}')
   fi
   for time in "${calculate_times[@]}"
   do
     printf "| %-15s | %-35s| % 15s |\n" "$PLATFORM" "$1" "$time"
   done
}

function bmrt_test_benchmark(){
    pushd models
    printf "| %-15s | %-35s| % 15s |\n" "测试平台" "测试模型" "calculate time(ms)"
    printf "| %-15s | %-35s| % 15s |\n" "-------" "-------------------" "--------------"

    if test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/mobilenetv4_conv_medium_fp32_1b.bmodel
      bmrt_test_case BM1684X/mobilenetv4_conv_medium_fp16_1b.bmodel
      bmrt_test_case BM1684X/mobilenetv4_conv_medium_int8_1b.bmodel
      bmrt_test_case BM1684X/mobilenetv4_conv_medium_int8_4b.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/mobilenetv4_conv_medium_fp32_1b.bmodel
      bmrt_test_case BM1688/mobilenetv4_conv_medium_fp16_1b.bmodel
      bmrt_test_case BM1688/mobilenetv4_conv_medium_int8_1b.bmodel
      bmrt_test_case BM1688/mobilenetv4_conv_medium_int8_4b.bmodel
      bmrt_test_case BM1688/mobilenetv4_conv_medium_fp32_1b_2core.bmodel
      bmrt_test_case BM1688/mobilenetv4_conv_medium_fp16_1b_2core.bmodel
      bmrt_test_case BM1688/mobilenetv4_conv_medium_int8_1b_2core.bmodel
      bmrt_test_case BM1688/mobilenetv4_conv_medium_int8_4b_2core.bmodel
    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/mobilenetv4_conv_medium_fp32_1b.bmodel
      bmrt_test_case CV186X/mobilenetv4_conv_medium_fp16_1b.bmodel
      bmrt_test_case CV186X/mobilenetv4_conv_medium_int8_1b.bmodel
      bmrt_test_case CV186X/mobilenetv4_conv_medium_int8_4b.bmodel
    fi

    popd
}

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
    if [[ $3 != 0 ]] && [[ $3 != "" ]];then
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
  pushd cpp/mobilenetv4_bmcv
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build mobilenetv4_bmcv" 0
  popd
}

function build_soc()
{
  pushd cpp/mobilenetv4_bmcv
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc mobilenetv4_bmcv" 0
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/mobilenetv4_bmcv
  if [ ! -d log ];then
    mkdir log
  fi
  ./mobilenetv4_bmcv.$1 --input=../../datasets/imagenet_val_1k/img --bmodel=../../models/$TARGET/$2 --dev_id=$TPUID > log/$1_bmcv_$2_debug.log 2>&1
  judge_ret $? "./mobilenetv4_bmcv.$1 --input=../../datasets/imagenet_val_1k/img --bmodel=../../models/$TARGET/$2 --dev_id=$TPUID > log/$1_bmcv_$2_debug.log 2>&1" log/$1_bmcv_$2_debug.log
  tail -n 15 log/$1_bmcv_$2_debug.log

  echo "==================="
  echo "Comparing statis..."
  python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=mobilenetv4_bmcv.$1 --language=cpp --input=log/$1_bmcv_$2_debug.log --bmodel=$2
  judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=mobilenetv4_bmcv.$1 --language=cpp --input=log/$1_bmcv_$2_debug.log --bmodel=$2"
  echo "==================="

  echo "Evaluating..."
  python3 ../../tools/eval_imagenet.py --gt_path ../../datasets/imagenet_val_1k/label.txt --result_json results/$2_img_bmcv_cpp_result.json 2>&1 | tee log/$1_bmcv_$2_eval.log
  echo "==================="
  echo "Comparing acc..."
  python3 ../../tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=mobilenetv4_bmcv.$1 --language=cpp --input=log/$1_bmcv_$2_eval.log --bmodel=$2 2>&1
  judge_ret $? "python3 ../../tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=mobilenetv4_bmcv.$1 --language=cpp --input=log/$1_bmcv_$2_eval.log --bmodel=$2"
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function eval_python()
{
  echo -e "\n########################\nCase Start: eval python\n########################"
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  python3 python/mobilenetv4_$1.py --input datasets/imagenet_val_1k/img --bmodel models/$TARGET/$2 --dev_id $TPUID > python/log/$1_$2_debug.log 2>&1
  judge_ret $? " python3 python/mobilenetv4_$1.py --input datasets/imagenet_val_1k/img --bmodel models/$TARGET/$2 --dev_id $TPUID > python/log/$1_$2_debug.log 2>&1" python/log/$1_$2_debug.log
  tail -n 15 python/log/$1_$2_debug.log

  echo "==================="
  echo "Comparing statis..."
  python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=mobilenetv4_$1.py --language=python --input=python/log/$1_$2_debug.log --bmodel=$2
  judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=mobilenetv4_$1.py --language=python --input=python/log/$1_$2_debug.log --bmodel=$2"
  echo "==================="

  echo "Evaluating..."
  python3 tools/eval_imagenet.py --gt_path datasets/imagenet_val_1k/label.txt --result_json python/results/$2_img_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log
  echo "==================="
  echo "Comparing acc..."
  python3 tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=mobilenetv4_$1.py --language=python --input=python/log/$1_$2_eval.log --bmodel=$2 2>&1
  judge_ret $? "python3 tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=mobilenetv4_$1.py --language=python --input=python/log/$1_$2_eval.log --bmodel=$2"
  echo -e "########################\nCase End: eval python\n########################\n"
}

if test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_build"
then
  build_pcie
elif test $MODE = "pcie_test"
then
  download
  if test $TARGET = "BM1684X"
  then
    if test $CASE_MODE = "fully"
    then
      eval_python opencv   mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b.bmodel
      eval_cpp   pcie      mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_cpp   pcie      mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_cpp   pcie      mobilenetv4_conv_medium_int8_1b.bmodel
      eval_cpp   pcie      mobilenetv4_conv_medium_int8_4b.bmodel
    elif test $CASE_MODE = "partly"
    then
      eval_python opencv   mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b.bmodel
      eval_cpp   pcie      mobilenetv4_conv_medium_int8_4b.bmodel
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  elif test $TARGET = "BM1688"
  then
    if test $CASE_MODE = "fully"
    then
      eval_python opencv   mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_fp32_1b_2core.bmodel
      eval_python opencv   mobilenetv4_conv_medium_fp16_1b_2core.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_1b_2core.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_4b_2core.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp32_1b_2core.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp16_1b_2core.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_1b_2core.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b_2core.bmodel
    elif test $CASE_MODE = "partly"
    then
      eval_python opencv   mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_4b_2core.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b_2core.bmodel
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  elif test $TARGET = "CV186X"
  then
    if test $CASE_MODE = "fully"
    then
      eval_python opencv   mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b.bmodel
    elif test $CASE_MODE = "partly"
    then
      eval_python opencv   mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b.bmodel
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  fi
elif test $MODE = "soc_build"
then
  build_soc
elif test $MODE = "soc_test"
then
  download
  pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684X"
  then
    if test $CASE_MODE = "fully"
    then
      eval_python opencv   mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b.bmodel
      eval_cpp   pcie      mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_cpp   pcie      mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_cpp   pcie      mobilenetv4_conv_medium_int8_1b.bmodel
      eval_cpp   pcie      mobilenetv4_conv_medium_int8_4b.bmodel
    elif test $CASE_MODE = "partly"
    then
      eval_python opencv   mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b.bmodel
      eval_cpp   pcie      mobilenetv4_conv_medium_int8_4b.bmodel
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  elif test $TARGET = "BM1688"
  then
    if test $CASE_MODE = "fully"
    then
      eval_python opencv   mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_fp32_1b_2core.bmodel
      eval_python opencv   mobilenetv4_conv_medium_fp16_1b_2core.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_1b_2core.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_4b_2core.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp32_1b_2core.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp16_1b_2core.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_1b_2core.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b_2core.bmodel
    elif test $CASE_MODE = "partly"
    then
      eval_python opencv   mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_4b_2core.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b_2core.bmodel
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  elif test $TARGET = "CV186X"
  then
    if test $CASE_MODE = "fully"
    then
      eval_python opencv   mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_1b.bmodel
      eval_python opencv   mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp32_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_fp16_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_1b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b.bmodel
    elif test $CASE_MODE = "partly"
    then
      eval_python opencv   mobilenetv4_conv_medium_int8_4b.bmodel
      eval_python bmcv     mobilenetv4_conv_medium_int8_4b.bmodel
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------mobilenetv4 acc----------"
  cat tools/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------mobilenetv4 performance-----------"
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
