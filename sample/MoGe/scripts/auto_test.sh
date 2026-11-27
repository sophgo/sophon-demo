#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1688"
MODE="soc_test"
TPUID=0
ALL_PASS=1
PYTEST="auto_test"
ECHO_LINES=20
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
CASE_MODE="fully"
usage()
{
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_build|pcie_test|soc_build|soc_test] [ -t TARGET BM1684X|BM1688|CV186X] [ -s SOCSDK] [ -d TPUID] [ -p PYTEST auto_test|pytest] [ -c fully|partly]" 1>&2
}

while getopts ":m:t:s:d:p:c:" opt
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

# bmrt_test 的 `calculate  time(s): X` 为纯推理耗时(秒)，此处转为 ms
function bmrt_test_case(){
   calculate_time_log=$(bmrt_test --bmodel $1 | grep "calculate" 2>&1)
   readarray -t calculate_times < <(echo "$calculate_time_log" | grep -oP 'calculate\s+time\(s\):\s*\K\d+\.\d+' | awk '{printf "%.2f \n", $1*1000}')
   for time in "${calculate_times[@]}"
   do
     printf "| %-15s | %-42s| % 15s |\n" "$PLATFORM" "$1" "$time"
   done
}
function bmrt_test_benchmark(){
    pushd models
    printf "| %-15s | %-42s| % 15s |\n" "测试平台" "测试模型" "calculate time(ms)"
    printf "| %-15s | %-42s| % 15s |\n" "-------" "-------------------" "--------------"
    for pre in fp32_1b fp16_1b; do
      bmrt_test_case ${TARGET}/moge-2-vitl-normal_${pre}.bmodel
    done
    if test $TARGET = "BM1688"; then
      for pre in fp16_1b; do
        bmrt_test_case ${TARGET}/moge-2-vitl-normal_${pre}_2core.bmodel
      done
    fi
    popd
}

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
  ./scripts/download.sh --$1
  judge_ret $? "download $1" 0
}

function build_soc()
{
  pushd cpp/moge_bmcv
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc moge_bmcv" 0
  popd
}

function build_pcie()
{
  pushd cpp/moge_bmcv
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build pcie moge_bmcv" 0
  popd
}

# 例程性能：python 与 cpp 在 datasets/test 上跑一遍，输出 decode/preprocess/inference/postprocess 平均耗时
function test_python()
{
  if [ ! -d log ];then
    mkdir log
  fi
  python3 python/moge_bmcv.py --input datasets/test --bmodel models/$TARGET/$1 --dev_id $TPUID > log/moge_bmcv_$1_python_test.log 2>&1
  judge_ret $? "python3 python/moge_bmcv.py --input datasets/test --bmodel models/$TARGET/$1 --dev_id $TPUID" log/moge_bmcv_$1_python_test.log
  tail -n 10 log/moge_bmcv_$1_python_test.log
  echo "==================="
  echo "Comparing statis..."
  python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=moge_bmcv.py --language=python --input=log/moge_bmcv_$1_python_test.log --bmodel=$1 2>&1
  judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=moge_bmcv.py --language=python --input=log/moge_bmcv_$1_python_test.log --bmodel=$1"
  echo "==================="
}

function test_cpp()
{
  pushd cpp/moge_bmcv
  if [ ! -d log ];then
    mkdir log
  fi
  ./moge_bmcv.$1 --input=../../datasets/test --bmodel=../../models/$TARGET/$2 --dev_id=$TPUID > log/moge_bmcv_$2_${1}_test.log 2>&1
  judge_ret $? "./moge_bmcv.$1 --input=../../datasets/test --bmodel=../../models/$TARGET/$2 --dev_id=$TPUID" log/moge_bmcv_$2_${1}_test.log
  tail -n 10 log/moge_bmcv_$2_${1}_test.log
  echo "==================="
  echo "Comparing statis..."
  python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=moge_bmcv.$1 --language=cpp --input=log/moge_bmcv_$2_${1}_test.log --bmodel=$2 2>&1
  judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=moge_bmcv.$1 --language=cpp --input=log/moge_bmcv_$2_${1}_test.log --bmodel=$2"
  echo "==================="
  popd
}

# 精度：eval_moge.py 在官方 NYUv2 数据集上计算 depth_metric 的 rel/delta1，再与基线比较
function eval_acc()
{
  echo -e "\n########################\nCase Start: eval $1 $2\n########################"
  if [ ! -d log ];then
    mkdir log
  fi
  python3 tools/eval_moge.py --backend bmodel --dataset datasets/NYUv2 --bmodel models/$TARGET/$2 --dev_id $TPUID 2>&1 | tee log/moge_bmcv_$2_eval.log
  judge_ret $? "python3 tools/eval_moge.py --backend bmodel --dataset datasets/NYUv2 --bmodel models/$TARGET/$2 --dev_id $TPUID" log/moge_bmcv_$2_eval.log
  echo "==================="
  echo "Comparing acc..."
  python3 tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=$1 --language=$3 --input=log/moge_bmcv_$2_eval.log --bmodel=$2 2>&1
  judge_ret $? "python3 tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=$1 --language=$3 --input=log/moge_bmcv_$2_eval.log --bmodel=$2"
  echo "==================="
  echo -e "########################\nCase End: eval $1 $2\n########################\n"
}

function compile_mlir()
{
  ./scripts/gen_fp32bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  ./scripts/gen_fp16bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp16bmodel" 0
}

if test $MODE = "compile_mlir"
then
  download onnx
  compile_mlir
elif test $MODE = "pcie_build"
then
  build_pcie
elif test $MODE = "pcie_test"
then
  download $TARGET
  if test $CASE_MODE = "fully"
  then
    for pre in fp32_1b fp16_1b; do
      test_python moge-2-vitl-normal_${pre}.bmodel
      test_cpp pcie moge-2-vitl-normal_${pre}.bmodel
    done
  elif test $CASE_MODE = "partly"
  then
    test_python moge-2-vitl-normal_fp16_1b.bmodel
    test_cpp pcie moge-2-vitl-normal_fp16_1b.bmodel
  fi
elif test $MODE = "soc_build"
then
  build_soc
elif test $MODE = "soc_test"
then
  download $TARGET
  if test $CASE_MODE = "fully"
  then
    for pre in fp32_1b fp16_1b; do
      test_python moge-2-vitl-normal_${pre}.bmodel
      test_cpp soc moge-2-vitl-normal_${pre}.bmodel
    done
  elif test $CASE_MODE = "partly"
  then
    test_python moge-2-vitl-normal_fp16_1b.bmodel
    test_cpp soc moge-2-vitl-normal_fp16_1b.bmodel
  fi

  # 精度评测（需要官方数据集：./scripts/download.sh --eval）
  if [ -d datasets/NYUv2 ]; then
    if test $CASE_MODE = "fully"
    then
      for pre in fp32_1b fp16_1b; do
        eval_acc moge_bmcv.py moge-2-vitl-normal_${pre}.bmodel python
      done
    elif test $CASE_MODE = "partly"
    then
      eval_acc moge_bmcv.py moge-2-vitl-normal_fp16_1b.bmodel python
    fi
  else
    echo "datasets/NYUv2 not found, skip accuracy evaluation (run: ./scripts/download.sh --eval)"
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  cat tools/acc.txt
  echo "-----------------------------"
  cat tools/benchmark.txt
  echo "-----------------------------"
  bmrt_test_benchmark
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