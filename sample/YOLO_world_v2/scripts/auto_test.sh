#!/bin/bash
# ==============================================================================
# YOLO-World v2 自动化测试 (参考 sample/YOLO_world/auto_test.sh)
# 用法: ./scripts/auto_test.sh -m <compile_mlir|pcie_test|soc_test> -t <BM1684X> -d <TPUID> -c <fully|partly>
# 仅 Python (opencv/bmcv), FP32/FP16, BM1684X; 无 INT8/C++
# ==============================================================================
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
CASE_MODE="partly"
if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi

usage()
{
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_test|soc_test] [ -t TARGET BM1684X] [-a SAIL] [ -d TPUID] [ -p PYTEST auto_test|pytest] [ -c fully|partly]" 1>&2
}

while getopts ":m:t:s:a:d:p:c:" opt
do
  case $opt in
    m) MODE=${OPTARG}; echo "mode is $MODE";;
    t) TARGET=${OPTARG}; echo "target is $TARGET";;
    s) SOCSDK=${OPTARG}; echo "soc-sdk is $SOCSDK";;
    a) SAIL_PATH=${OPTARG}; echo "sail_path is $SAIL_PATH";;
    d) TPUID=${OPTARG}; echo "using tpu $TPUID";;
    p) PYTEST=${OPTARG}; echo "generate logs for $PYTEST";;
    c) CASE_MODE=${OPTARG}; echo "case mode is $CASE_MODE";;
    ?) usage; exit 1;;
  esac
done

if [ -f "scripts/acc.txt" ]; then
  rm scripts/acc.txt
fi
echo "|   测试平台    |      测试程序     |    测试模型        | AP@IoU=0.5:0.95 | AP@IoU=0.5 |" >> scripts/acc.txt

PLATFORM=$TARGET
if test $MODE = "soc_test"; then
  if test $TARGET = "BM1684X"; then
    PLATFORM="SE7-32"
  else
    echo "Unknown TARGET type: $TARGET"; exit 1
  fi
fi

function bmrt_test_case(){
   calculate_time_log=$(bmrt_test --bmodel $1 --devid $TPUID | grep "calculate" 2>&1)
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
    if test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/yoloworld_v2_fp32_1b.bmodel
      bmrt_test_case BM1684X/yoloworld_v2_fp16_1b.bmodel
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
    echo "Passed: $2"; echo ""
    if test $PYTEST = "pytest"; then echo "Passed: $2" >> ${top_dir}auto_test_result.txt; echo "#######Debug Info Start#######" >> ${top_dir}auto_test_result.txt; fi
  else
    echo "Failed: $2"; ALL_PASS=0
    if test $PYTEST = "pytest"; then echo "Failed: $2" >> ${top_dir}auto_test_result.txt; echo "#######Debug Info Start#######" >> ${top_dir}auto_test_result.txt; fi
  fi
  if test $PYTEST = "pytest"; then
    if [[ $3 != 0 ]] && [[ $3 != "" ]];then tail -n ${ECHO_LINES} $3 >> ${top_dir}auto_test_result.txt; fi
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

function test_python()
{
  if [ ! -d log ];then mkdir log; fi
  python3 python/yoloworld_$1.py --input $3 --bmodel models/$TARGET/$2 --clip_bmodel models/$TARGET/$4 --dev_id $TPUID > log/$1_$2_python_test.log 2>&1
  judge_ret $? "python3 python/yoloworld_$1.py --input $3 --bmodel models/$TARGET/$2 --clip_bmodel models/$TARGET/$4 --dev_id $TPUID" log/$1_$2_python_test.log
  tail -n 20 log/$1_$2_python_test.log
  if test $3 = "datasets/coco/val2017_1000"; then
    echo "==================="; echo "Comparing statis..."
    python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yoloworld_$1.py --language=python --input=log/$1_$2_python_test.log --bmodel=$2
    judge_ret $? "compare_statis $1 $2"
    echo "==================="
  fi
}

function eval_python()
{
  echo -e "\n########################\nCase Start: eval python\n########################"
  if [ ! -d python/log ];then mkdir python/log; fi
  python3 python/yoloworld_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --clip_bmodel models/$TARGET/$4 --dev_id $TPUID --conf_thresh 0.001 --nms_thresh 0.7 --class_names "all" > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/yoloworld_$1.py ... --conf_thresh 0.001 --nms_thresh 0.7 --class_names all" python/log/$1_$2_debug.log
  tail -n 20 python/log/$1_$2_debug.log

  echo "Evaluating..."
  res=$(python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/$2_val2017_1000_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
  judge_ret $? "$2_val2017_1000_$1_python_result: Precision compare!" python/log/$1_$2_eval.log
  ap0=$(echo -e "$res"| grep "Average Precision  (AP) @\[ IoU\=0.50:0.95 | area\=   all | maxDets\=100 \]" | grep -oP ' = \K\d+\.\d+' | awk '{printf "%.3f \n", $1}')
  ap1=$(echo -e "$res"| grep "Average Precision  (AP) @\[ IoU\=0.50      | area\=   all | maxDets\=100 \]" | grep -oP ' = \K\d+\.\d+' | awk '{printf "%.3f \n", $1}')
  printf "| %-12s | %-14s | %-22s | %8.3f | %8.3f |\n" "$PLATFORM" "yoloworld_$1.py" "$2" "$(printf "%.3f" $ap0)" "$(printf "%.3f" $ap1)" >> scripts/acc.txt
  echo -e "########################\nCase End: eval python\n########################\n"
}

function compile_mlir()
{
  ./scripts/gen_fp32bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  ./scripts/gen_fp16bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp16bmodel" 0
}

function compare_res(){
    ret=`awk -v x=$1 -v y=$2 'BEGIN{print(x-y<0.001 && y-x<0.001)?1:0}'`
    if [ $ret -eq 0 ]; then
        ALL_PASS=0
        echo "***************************************"
        echo "Ground truth is $2, your result is: $1"
        echo -e "\e[41m compare wrong! \e[0m"
        echo "***************************************"
        return 1
    else
        echo "***************************************"
        echo -e "\e[42m compare right! \e[0m"
        echo "***************************************"
        return 0
    fi
}

CLIP=clip_text_vitb32_bm1684x_f16_1b.bmodel

if test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_test"
then
  download
  pip3 install pycocotools opencv-python-headless ftfy regex torch setuptools==69.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
  bmrt_test_benchmark
  if test $TARGET = "BM1684X"; then
    if test $CASE_MODE = "fully"; then
      test_python opencv yoloworld_v2_fp32_1b.bmodel datasets/test_car_person_1080P.mp4 $CLIP
      test_python opencv yoloworld_v2_fp16_1b.bmodel datasets/test_car_person_1080P.mp4 $CLIP
      test_python bmcv   yoloworld_v2_fp32_1b.bmodel datasets/test_car_person_1080P.mp4 $CLIP
      test_python bmcv   yoloworld_v2_fp16_1b.bmodel datasets/test_car_person_1080P.mp4 $CLIP
      #performance test
      test_python opencv yoloworld_v2_fp32_1b.bmodel datasets/coco/val2017_1000 $CLIP
      test_python opencv yoloworld_v2_fp16_1b.bmodel datasets/coco/val2017_1000 $CLIP
      test_python bmcv   yoloworld_v2_fp32_1b.bmodel datasets/coco/val2017_1000 $CLIP
      test_python bmcv   yoloworld_v2_fp16_1b.bmodel datasets/coco/val2017_1000 $CLIP
      eval_python opencv yoloworld_v2_fp32_1b.bmodel 0.376 $CLIP
      eval_python opencv yoloworld_v2_fp16_1b.bmodel 0.376 $CLIP
      eval_python bmcv   yoloworld_v2_fp32_1b.bmodel 0.376 $CLIP
      eval_python bmcv   yoloworld_v2_fp16_1b.bmodel 0.371 $CLIP
    elif test $CASE_MODE = "partly"; then
      test_python opencv yoloworld_v2_fp16_1b.bmodel datasets/coco/val2017_1000 $CLIP
      eval_python opencv yoloworld_v2_fp16_1b.bmodel 0.376 $CLIP
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  fi
elif test $MODE = "soc_test"
then
  download
  pip3 install pycocotools opencv-python-headless ftfy regex torch -i https://pypi.tuna.tsinghua.edu.cn/simple
  bmrt_test_benchmark
  if test $TARGET = "BM1684X"; then
    if test $CASE_MODE = "fully"; then
      test_python opencv yoloworld_v2_fp32_1b.bmodel datasets/test_car_person_1080P.mp4 $CLIP
      test_python opencv yoloworld_v2_fp16_1b.bmodel datasets/test_car_person_1080P.mp4 $CLIP
      test_python bmcv   yoloworld_v2_fp32_1b.bmodel datasets/test_car_person_1080P.mp4 $CLIP
      test_python bmcv   yoloworld_v2_fp16_1b.bmodel datasets/test_car_person_1080P.mp4 $CLIP
      test_python opencv yoloworld_v2_fp32_1b.bmodel datasets/coco/val2017_1000 $CLIP
      test_python opencv yoloworld_v2_fp16_1b.bmodel datasets/coco/val2017_1000 $CLIP
      test_python bmcv   yoloworld_v2_fp32_1b.bmodel datasets/coco/val2017_1000 $CLIP
      test_python bmcv   yoloworld_v2_fp16_1b.bmodel datasets/coco/val2017_1000 $CLIP
      eval_python opencv yoloworld_v2_fp32_1b.bmodel 0.376 $CLIP
      eval_python opencv yoloworld_v2_fp16_1b.bmodel 0.376 $CLIP
      eval_python bmcv   yoloworld_v2_fp32_1b.bmodel 0.376 $CLIP
      eval_python bmcv   yoloworld_v2_fp16_1b.bmodel 0.371 $CLIP
    elif test $CASE_MODE = "partly"; then
      test_python opencv yoloworld_v2_fp16_1b.bmodel datasets/coco/val2017_1000 $CLIP
      eval_python opencv yoloworld_v2_fp16_1b.bmodel 0.376 $CLIP
    fi
  fi
else
  echo "unknown MODE: $MODE"; usage; exit 1
fi

popd
echo ""
cat scripts/acc.txt
echo ""
if [ $ALL_PASS -eq 1 ]; then
  echo -e "\e[42m all tests passed! \e[0m"
else
  echo -e "\e[41m some tests failed! \e[0m"
  exit 1
fi
