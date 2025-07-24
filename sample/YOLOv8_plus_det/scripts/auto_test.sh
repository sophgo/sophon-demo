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
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_build|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X|BM1688|CV186X] [ -s SOCSDK] [-a SAIL] [ -d TPUID] [ -p PYTEST auto_test|pytest] [ -c fully|partly]" 1>&2 
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

    for model in yolov8s yolov9s yolov11s yolov12s; do
      for pre in fp32_1b fp16_1b int8_1b int8_4b; do
        bmrt_test_case ${TARGET}/${model}_${pre}.bmodel
      done
      if test $TARGET = "BM1688"; then
        bmrt_test_case ${TARGET}/${model}_int8_4b_2core.bmodel
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
  pushd cpp/yolov8_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build yolov8_$1" 0
  popd
}

function build_soc()
{
  pushd cpp/yolov8_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc yolov8_$1" 0
  popd
}

function test_python()
{
  if [ ! -d log ];then
    mkdir log
  fi
  python3 python/yolov8_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_python_test.log 2>&1
  judge_ret $? "python3 python/yolov8_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID" log/$1_$2_python_test.log
  tail -n 20 log/$1_$2_python_test.log
  if test $3 = "datasets/coco/val2017_1000"; then
    echo "==================="
    echo "Comparing statis..."
    python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$1.py --language=python --input=log/$1_$2_python_test.log --bmodel=$2 2>&1
    judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$1.py --language=python --input=log/$1_$2_python_test.log --bmodel=$2"
    echo "==================="
  fi
}

function eval_python()
{ 
  echo -e "\n########################\nCase Start: eval python\n########################"
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  python3 python/yolov8_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.001 --nms_thresh 0.7 > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/yolov8_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.001 --nms_thresh 0.7 > python/log/$1_$2_debug.log 2>&1" python/log/$1_$2_debug.log
  tail -n 20 python/log/$1_$2_debug.log

  echo "Evaluating..."
  python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/$2_val2017_1000_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log

  echo "==================="
  echo "Comparing acc..."
  python3 tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$1.py --language=python --input=python/log/$1_$2_eval.log --bmodel=$2 2>&1
  judge_ret $? "python3 tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$1.py --language=python --input=python/log/$1_$2_eval.log --bmodel=$2"
  echo "==================="
  echo -e "########################\nCase End: eval python\n########################\n"
}

function test_cpp()
{
  pushd cpp/yolov8_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./yolov8_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_cpp_test.log 2>&1
  judge_ret $? "./yolov8_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID" log/$1_$2_$3_cpp_test.log
  tail -n 15 log/$1_$2_$3_cpp_test.log
  if test $4 = "../../datasets/coco/val2017_1000"; then
    echo "==================="
    echo "Comparing statis..."
    python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$2.$1 --language=cpp --input=log/$1_$2_$3_cpp_test.log --bmodel=$3 2>&1
    judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$2.$1 --language=cpp --input=log/$1_$2_$3_cpp_test.log --bmodel=$3"
    echo "==================="
  fi
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/yolov8_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./yolov8_$2.$1 --input=../../datasets/coco/val2017_1000 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.001 --nms_thresh=0.7  --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./yolov8_$2.$1 --input=../../datasets/coco/val2017_1000 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.001 --nms_thresh=0.7  --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1" log/$1_$2_$3_debug.log
  tail -n 15 log/$1_$2_$3_debug.log
  
  echo "Evaluating..."
  python3 ../../tools/eval_coco.py --gt_path ../../datasets/coco/instances_val2017_1000.json --result_json results/$3_val2017_1000_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log
  echo "==================="
  echo "Comparing acc..."
  python3 ../../tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$2.$1 --language=cpp --input=log/$1_$2_$3_eval.log --bmodel=$3 2>&1
  judge_ret $? "python3 ../../tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$2.$1 --language=cpp --input=log/$1_$2_$3_eval.log --bmodel=$3"
  echo "==================="
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
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

if test $MODE = "compile_mlir"
then
  download onnx
  compile_mlir
elif test $MODE = "pcie_build"
then
  build_pcie bmcv
elif test $MODE = "pcie_test"
then
  pip3 install pycocotools opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  download $TARGET
  if test $CASE_MODE = "fully"
  then
    for model in yolov8s yolov9s yolov11s yolov12s; do
      for pre in int8_1b int8_4b; do
        test_python bmcv ${model}_${pre}.bmodel datasets/test_car_person_1080P.mp4
        test_cpp pcie bmcv ${model}_${pre}.bmodel ../../datasets/test_car_person_1080P.mp4
      done
      for pre in fp32_1b fp16_1b int8_1b int8_4b; do
        test_python bmcv ${model}_${pre}.bmodel datasets/coco/val2017_1000
        test_cpp pcie bmcv ${model}_${pre}.bmodel ../../datasets/coco/val2017_1000
        eval_python bmcv ${model}_${pre}.bmodel
        eval_cpp pcie bmcv ${model}_${pre}.bmodel
      done
      if test "$TARGET" = "BM1688"; then 
        test_python bmcv ${model}_int8_4b_2core.bmodel datasets/coco/val2017_1000
        test_cpp pcie bmcv ${model}_int8_4b_2core.bmodel ../../datasets/coco/val2017_1000
        eval_python bmcv ${model}_int8_4b_2core.bmodel
        eval_cpp pcie bmcv ${model}_int8_4b_2core.bmodel
      fi
    done
  elif test $CASE_MODE = "partly"
  then
    test_python bmcv yolov8s_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov8s_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    eval_python bmcv yolov8s_int8_4b.bmodel
    eval_cpp pcie bmcv yolov8s_int8_4b.bmodel
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  pip3 install pycocotools opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  download $TARGET
  if test $CASE_MODE = "fully"
  then
    for model in yolov8s yolov9s yolov11s yolov12s; do
      for pre in int8_1b int8_4b; do
        test_python bmcv ${model}_${pre}.bmodel datasets/test_car_person_1080P.mp4
        test_cpp soc bmcv ${model}_${pre}.bmodel ../../datasets/test_car_person_1080P.mp4
      done
      for pre in fp32_1b fp16_1b int8_1b int8_4b; do
        test_python bmcv ${model}_${pre}.bmodel datasets/coco/val2017_1000
        test_cpp soc bmcv ${model}_${pre}.bmodel ../../datasets/coco/val2017_1000
        eval_python bmcv ${model}_${pre}.bmodel
        eval_cpp soc bmcv ${model}_${pre}.bmodel
      done
      if test "$PLATFORM" = "SE9-16"; then 
        test_python bmcv ${model}_int8_4b_2core.bmodel datasets/coco/val2017_1000
        test_cpp soc bmcv ${model}_int8_4b_2core.bmodel ../../datasets/coco/val2017_1000
      fi
    done
  elif test $CASE_MODE = "partly"
  then
    test_python bmcv yolov8s_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov8s_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    test_python bmcv yolov8s_int8_4b.bmodel datasets/coco/val2017_1000
    test_cpp soc bmcv yolov8s_int8_4b.bmodel ../../datasets/coco/val2017_1000

    eval_python bmcv yolov8s_int8_4b.bmodel
    eval_cpp soc bmcv yolov8s_int8_4b.bmodel
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
