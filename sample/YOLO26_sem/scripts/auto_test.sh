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

if [ -f "tools/acc.txt" ]; then
  rm tools/acc.txt
fi
if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi
PLATFORM=$TARGET
if test $MODE = "soc_test"; then
  if test $TARGET = "BM1684X"; then
    PLATFORM="SE7-32"
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
   calculate_time_log=$(bmrt_test --bmodel $1 | grep "calculate" 2>&1)
   readarray -t calculate_times < <(echo "$calculate_time_log" | grep -oP 'calculate  time\(s\): \K\d+\.\d+' | awk '{printf "%.2f \n", $1 * 1000}')
   for time in "${calculate_times[@]}"
   do
     printf "| %-15s | %-35s| % 15s |\n" "$PLATFORM" "$1" "$time"
   done
}
function bmrt_test_benchmark(){
    pushd models
    printf "| %-15s | %-35s| % 15s |\n" "测试平台" "测试模型" "calculate time(ms)"
    printf "| %-15s | %-35s| % 15s |\n" "-------" "-------------------" "--------------"

    for model in yolo26s; do
      for pre in fp32_1b fp16_1b int8_1b; do
        bmrt_test_case ${TARGET}/${model}_${pre}.bmodel
      done
      if test $TARGET = "BM1688"; then
        bmrt_test_case ${TARGET}/${model}_int8_1b_2core.bmodel
      fi
    done

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
  judge_ret $? "download" 0
}

function build_pcie()
{
  pushd cpp/yolo26_sem_bmcv
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build yolo26_sem_bmcv" 0
  popd
}

function build_soc()
{
  pushd cpp/yolo26_sem_bmcv
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc yolo26_sem_bmcv" 0
  popd
}

function test_python()
{
  if [ ! -d log ];then
    mkdir log
  fi
  python3 python/yolo26_sem_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_python_test.log 2>&1
  judge_ret $? "python3 python/yolo26_sem_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID" log/$1_$2_python_test.log
  tail -n 20 log/$1_$2_python_test.log
}

function eval_python()
{
  echo -e "\n########################\nCase Start: eval python $1_$2\n########################"
  if [ ! -d log ];then
    mkdir log
  fi
  python3 python/yolo26_sem_$1.py --input datasets/cityscapes/leftImg8bit/val --bmodel models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_eval.log 2>&1
  judge_ret $? "python3 python/yolo26_sem_$1.py --input datasets/cityscapes/leftImg8bit/val --bmodel models/$TARGET/$2 --dev_id $TPUID" log/$1_$2_eval.log

  echo "Evaluating..."
  python3 tools/eval_cityscapes.py --pred_dir results/segmaps --gt_dir datasets/cityscapes/gtFine/val --img_suffix _leftImg8bit.png --gt_suffix _gtFine_labelIds.png > log/$1_$2_eval_res.log 2>&1
  judge_ret $? "python3 tools/eval_cityscapes.py (pred_dir=results/segmaps, $1_$2)" log/$1_$2_eval_res.log
  tail -n 25 log/$1_$2_eval_res.log

  echo "==================="
  echo "Comparing acc..."
  python3 tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=yolo26_sem_$1.py --language=python --input=log/$1_$2_eval_res.log --bmodel=$2 2>&1
  judge_ret $? "python3 tools/compare_acc.py --program=yolo26_sem_$1.py --input=log/$1_$2_eval_res.log --bmodel=$2"

  echo "Comparing statis..."
  python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolo26_sem_$1.py --language=python --input=log/$1_$2_eval.log --bmodel=$2 2>&1
  judge_ret $? "python3 tools/compare_statis.py --program=yolo26_sem_$1.py --input=log/$1_$2_eval.log --bmodel=$2"
  echo "==================="
  echo -e "########################\nCase End: eval python $1_$2\n########################\n"
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp $1_$2\n########################"
  pushd cpp/yolo26_sem_bmcv
  if [ ! -d log ];then
    mkdir log
  fi
  ./yolo26_sem_bmcv.$1 --input=../../datasets/cityscapes/leftImg8bit/val --bmodel=../../models/$TARGET/$2 --dev_id=$TPUID > log/$1_yolo26_sem_bmcv_$2_eval.log 2>&1
  judge_ret $? "./yolo26_sem_bmcv.$1 --input=../../datasets/cityscapes/leftImg8bit/val --bmodel=../../models/$TARGET/$2 --dev_id=$TPUID" log/$1_yolo26_sem_bmcv_$2_eval.log
  tail -n 15 log/$1_yolo26_sem_bmcv_$2_eval.log

  echo "Evaluating..."
  python3 ../../tools/eval_cityscapes.py --pred_dir results/segmaps --gt_dir ../../datasets/cityscapes/gtFine/val --img_suffix _leftImg8bit.png --gt_suffix _gtFine_labelIds.png > log/$1_yolo26_sem_bmcv_$2_eval_res.log 2>&1
  judge_ret $? "python3 ../../tools/eval_cityscapes.py (cpp $1_$2)" log/$1_yolo26_sem_bmcv_$2_eval_res.log
  tail -n 25 log/$1_yolo26_sem_bmcv_$2_eval_res.log

  echo "==================="
  echo "Comparing acc..."
  python3 ../../tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=yolo26_sem_bmcv.$1 --language=cpp --input=log/$1_yolo26_sem_bmcv_$2_eval_res.log --bmodel=$2 2>&1
  judge_ret $? "python3 ../../tools/compare_acc.py (cpp $1_$2)"

  echo "Comparing statis..."
  python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolo26_sem_bmcv.$1 --language=cpp --input=log/$1_yolo26_sem_bmcv_$2_eval.log --bmodel=$2 2>&1
  judge_ret $? "python3 ../../tools/compare_statis.py (cpp $1_$2)"
  echo "==================="
  popd
  echo -e "########################\nCase End: eval cpp $1_$2\n########################\n"
}

function test_cpp()
{
  pushd cpp/yolo26_sem_bmcv
  if [ ! -d log ];then
    mkdir log
  fi
  ./yolo26_sem_bmcv.$1 --input=$3 --bmodel=../../models/$TARGET/$2 --dev_id=$TPUID > log/$1_yolo26_sem_bmcv_$2_cpp_test.log 2>&1
  judge_ret $? "./yolo26_sem_bmcv.$1 --input=$3 --bmodel=../../models/$TARGET/$2 --dev_id=$TPUID" log/$1_yolo26_sem_bmcv_$2_cpp_test.log
  tail -n 15 log/$1_yolo26_sem_bmcv_$2_cpp_test.log
  popd
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

if test $MODE = "compile_mlir"
then
  download onnx
  compile_mlir
elif test $MODE = "pcie_build"
then
  build_pcie
elif test $MODE = "pcie_test"
then
  pip3 install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  download $TARGET
  if test $CASE_MODE = "fully"
  then
    for model in yolo26s; do
      for pre in int8_1b; do
        test_python opencv ${model}_${pre}.bmodel datasets/cityscapes_video.avi
        test_python bmcv ${model}_${pre}.bmodel datasets/cityscapes_video.avi
        test_cpp pcie ${model}_${pre}.bmodel ../../datasets/cityscapes_video.avi
      done
      for pre in fp32_1b fp16_1b int8_1b; do
        test_python opencv ${model}_${pre}.bmodel datasets/test
        test_python bmcv ${model}_${pre}.bmodel datasets/test
        test_cpp pcie ${model}_${pre}.bmodel ../../datasets/test
        eval_python opencv ${model}_${pre}.bmodel
        eval_python bmcv ${model}_${pre}.bmodel
        eval_cpp pcie ${model}_${pre}.bmodel
      done
    done
  elif test $CASE_MODE = "partly"
  then
    test_python opencv yolo26s_int8_1b.bmodel datasets/test
    test_python bmcv yolo26s_int8_1b.bmodel datasets/test
    test_cpp pcie yolo26s_int8_1b.bmodel ../../datasets/test
    eval_python opencv yolo26s_int8_1b.bmodel
    eval_python bmcv yolo26s_int8_1b.bmodel
    eval_cpp pcie yolo26s_int8_1b.bmodel
  fi
elif test $MODE = "soc_build"
then
  build_soc
elif test $MODE = "soc_test"
then
  pip3 install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  download $TARGET
  if test $CASE_MODE = "fully"
  then
    for model in yolo26s; do
      for pre in int8_1b; do
        test_python opencv ${model}_${pre}.bmodel datasets/cityscapes_video.avi
        test_python bmcv ${model}_${pre}.bmodel datasets/cityscapes_video.avi
        test_cpp soc ${model}_${pre}.bmodel ../../datasets/cityscapes_video.avi
      done
      for pre in fp32_1b fp16_1b int8_1b; do
        test_python opencv ${model}_${pre}.bmodel datasets/test
        test_python bmcv ${model}_${pre}.bmodel datasets/test
        test_cpp soc ${model}_${pre}.bmodel ../../datasets/test
        eval_python opencv ${model}_${pre}.bmodel
        eval_python bmcv ${model}_${pre}.bmodel
        eval_cpp soc ${model}_${pre}.bmodel
      done
      if test "$PLATFORM" = "SE9-16"; then
        test_python opencv ${model}_int8_1b_2core.bmodel datasets/test
        test_python bmcv ${model}_int8_1b_2core.bmodel datasets/test
        test_cpp soc ${model}_int8_1b_2core.bmodel ../../datasets/test
        eval_python opencv ${model}_int8_1b_2core.bmodel
        eval_python bmcv ${model}_int8_1b_2core.bmodel
        eval_cpp soc ${model}_int8_1b_2core.bmodel
      fi
    done
  elif test $CASE_MODE = "partly"
  then
    test_python opencv yolo26s_int8_1b.bmodel datasets/test
    test_python bmcv yolo26s_int8_1b.bmodel datasets/test
    test_cpp soc yolo26s_int8_1b.bmodel ../../datasets/test
    eval_python opencv yolo26s_int8_1b.bmodel
    eval_python bmcv yolo26s_int8_1b.bmodel
    eval_cpp soc yolo26s_int8_1b.bmodel
  fi
fi
if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  cat tools/acc.txt
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