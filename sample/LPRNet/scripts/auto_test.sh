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


if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi

if [ -f "scripts/acc.txt" ]; then
  rm scripts/acc.txt
fi
echo "|   测试平台    |      测试程序      |        测试模型        |    acc    |" >> scripts/acc.txt

PLATFORM=$TARGET
if test $MODE = "soc_test"; then
  if test $TARGET = "BM1684X"; then
    PLATFORM="SE7-32"
  elif test $TARGET = "BM1684"; then
    PLATFORM="SE5-16"
  elif test $TARGET = "BM1688"; then
    PLATFORM="SE9-16"
    cpu_core_num=$(nproc)
    if [ "$cpu_core_num" -eq 6 ]; then
      PLATFORM="SE9-8"
    fi
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
     printf "| %-35s| % 15s |\n" "$1" "$time"
   done
}
function bmrt_test_benchmark(){
    pushd models
    printf "| %-35s| % 15s |\n" "测试模型" "calculate time(ms)"
    printf "| %-35s| % 15s |\n" "-------------------" "--------------"
   
    if test $TARGET = "BM1684"; then
      bmrt_test_case BM1684/lprnet_fp32_1b.bmodel
      bmrt_test_case BM1684/lprnet_int8_1b.bmodel
      bmrt_test_case BM1684/lprnet_int8_4b.bmodel
    elif test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/lprnet_fp32_1b.bmodel
      bmrt_test_case BM1684X/lprnet_fp16_1b.bmodel
      bmrt_test_case BM1684X/lprnet_int8_1b.bmodel
      bmrt_test_case BM1684X/lprnet_int8_4b.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/lprnet_fp32_1b.bmodel
      bmrt_test_case BM1688/lprnet_fp16_1b.bmodel
      bmrt_test_case BM1688/lprnet_int8_1b.bmodel
      bmrt_test_case BM1688/lprnet_int8_4b.bmodel
      if test "$PLATFORM" = "SE9-16"; then 
        bmrt_test_case BM1688/lprnet_fp32_1b_2core.bmodel
        bmrt_test_case BM1688/lprnet_fp16_1b_2core.bmodel
        bmrt_test_case BM1688/lprnet_int8_1b_2core.bmodel
        bmrt_test_case BM1688/lprnet_int8_4b_2core.bmodel
      fi
    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/lprnet_fp32_1b.bmodel
      bmrt_test_case CV186X/lprnet_fp16_1b.bmodel
      bmrt_test_case CV186X/lprnet_int8_1b.bmodel
      bmrt_test_case CV186X/lprnet_int8_4b.bmodel
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
    ret=`awk -v x=$1 -v y=$2 'BEGIN{print(x-y<0.01 && y-x<0.01)?1:0}'`
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

  echo "==================="
  echo "Comparing statis..."
  python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=lprnet_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3
  judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=lprnet_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../../tools/eval_ccpd.py --gt_path ../../datasets/test_label.json --result_json results/$3_test_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"

  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  judge_ret $? "$3_test_$2_cpp_result: Precision compare!" log/$1_$2_$3_eval.log
  popd
  
  printf "| %-12s | %-18s | %-40s | %8.3f |\n" "$PLATFORM" "lprnet_$2.$1" "$3" "$(printf "%.3f" $acc)">> scripts/acc.txt

  echo -e "########################\nCase End: eval cpp\n########################\n"
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
  
  echo "==================="
  echo "Comparing statis..."
  python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=lprnet_$1.py --language=python --input=python/log/$1_$2_debug.log --bmodel=$2
  judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=lprnet_$1.py --language=python --input=python/log/$1_$2_debug.log --bmodel=$2"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 tools/eval_ccpd.py --gt_path datasets/test_label.json --result_json results/$2_test_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
  judge_ret $? "$2_test_$1_python_result: Precision compare!" python/log/$1_$2_eval.log

  printf "| %-12s | %-18s | %-40s | %8.3f |\n" "$PLATFORM" "lprnet_$1.py" "$2" "$(printf "%.3f" $acc)">> scripts/acc.txt

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
  elif [ "$TARGET" = "BM1688" ] || [ "$TARGET" = "CV186X" ]
  then
    eval_python opencv lprnet_fp32_1b.bmodel  0.894
    eval_python opencv lprnet_fp16_1b.bmodel  0.894
    eval_python opencv lprnet_int8_1b.bmodel  0.886
    eval_python opencv lprnet_int8_4b.bmodel  0.909
    eval_python bmcv lprnet_fp32_1b.bmodel    0.882
    eval_python bmcv lprnet_fp16_1b.bmodel    0.882
    eval_python bmcv lprnet_int8_1b.bmodel    0.882
    eval_python bmcv lprnet_int8_4b.bmodel    0.889 
    eval_cpp soc opencv lprnet_fp32_1b.bmodel 0.881
    eval_cpp soc opencv lprnet_fp16_1b.bmodel 0.881
    eval_cpp soc opencv lprnet_int8_1b.bmodel 0.883
    eval_cpp soc opencv lprnet_int8_4b.bmodel 0.880
    eval_cpp soc bmcv lprnet_fp32_1b.bmodel   0.882
    eval_cpp soc bmcv lprnet_fp16_1b.bmodel   0.882
    eval_cpp soc bmcv lprnet_int8_1b.bmodel   0.882
    eval_cpp soc bmcv lprnet_int8_4b.bmodel   0.879

    if test "$PLATFORM" = "SE9-16"; then 
      eval_python opencv lprnet_fp32_1b_2core.bmodel  0.894
      eval_python opencv lprnet_fp16_1b_2core.bmodel  0.894
      eval_python opencv lprnet_int8_1b_2core.bmodel  0.886
      eval_python opencv lprnet_int8_4b_2core.bmodel  0.909
      eval_python bmcv lprnet_fp32_1b_2core.bmodel    0.882
      eval_python bmcv lprnet_fp16_1b_2core.bmodel    0.882
      eval_python bmcv lprnet_int8_1b_2core.bmodel    0.882
      eval_python bmcv lprnet_int8_4b_2core.bmodel    0.889 
      eval_cpp soc opencv lprnet_fp32_1b_2core.bmodel 0.881
      eval_cpp soc opencv lprnet_fp16_1b_2core.bmodel 0.881
      eval_cpp soc opencv lprnet_int8_1b_2core.bmodel 0.883
      eval_cpp soc opencv lprnet_int8_4b_2core.bmodel 0.880
      eval_cpp soc bmcv lprnet_fp32_1b_2core.bmodel   0.882
      eval_cpp soc bmcv lprnet_fp16_1b_2core.bmodel   0.882
      eval_cpp soc bmcv lprnet_int8_1b_2core.bmodel   0.882
      eval_cpp soc bmcv lprnet_int8_4b_2core.bmodel   0.879
    fi
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------lprnet acc----------"
  cat scripts/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------lprnet performance-----------"
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