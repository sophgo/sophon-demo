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
if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi

CASE_MODE="fully"

usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_build|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X|BM1688|CV186X] [ -s SOCSDK] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2 
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
if [ -f "scripts/acc.txt" ]; then
  rm scripts/acc.txt
fi
echo "|   测试平台   |     测试程序     |              测试模型              | AP@IoU=0.5:0.95 | AP@IoU=0.5 |" >> scripts/acc.txt
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
     printf "| %-35s| % 15s |\n" "$1" "$time"
   done
}
function bmrt_test_benchmark(){
    pushd models
    printf "| %-35s| % 15s |\n" "测试模型" "calculate time(ms)"
    printf "| %-35s| % 15s |\n" "-------------------" "--------------"
   
    if test $TARGET = "BM1684"; then
      bmrt_test_case BM1684/yolov8s-pose_fp32_1b.bmodel
      bmrt_test_case BM1684/yolov8s-pose_int8_1b.bmodel
      bmrt_test_case BM1684/yolov8s-pose_int8_4b.bmodel

    elif test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/yolov8s-pose_fp32_1b.bmodel
      bmrt_test_case BM1684X/yolov8s-pose_fp16_1b.bmodel
      bmrt_test_case BM1684X/yolov8s-pose_int8_1b.bmodel
      bmrt_test_case BM1684X/yolov8s-pose_int8_4b.bmodel

    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/yolov8s-pose_fp32_1b.bmodel
      bmrt_test_case BM1688/yolov8s-pose_fp16_1b.bmodel
      bmrt_test_case BM1688/yolov8s-pose_int8_1b.bmodel
      bmrt_test_case BM1688/yolov8s-pose_int8_4b.bmodel
      bmrt_test_case BM1688/yolov8s-pose_fp32_1b_2core.bmodel
      bmrt_test_case BM1688/yolov8s-pose_fp16_1b_2core.bmodel
      bmrt_test_case BM1688/yolov8s-pose_int8_1b_2core.bmodel
      bmrt_test_case BM1688/yolov8s-pose_int8_4b_2core.bmodel

    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/yolov8s-pose_fp32_1b.bmodel
      bmrt_test_case CV186X/yolov8s-pose_fp16_1b.bmodel
      bmrt_test_case CV186X/yolov8s-pose_int8_1b.bmodel
      bmrt_test_case CV186X/yolov8s-pose_int8_4b.bmodel
   
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
    python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$1.py --language=python --input=log/$1_$2_python_test.log --bmodel=$2
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
  res=$(python3 tools/eval_coco.py --gt_path datasets/coco/person_keypoints_val2017_1000.json --result_json results/$2_val2017_1000_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
  judge_ret $? "$2_val2017_1000_$1_python_result: Precision compare!" python/log/$1_$2_eval.log
  ap0=$(echo -e "$res"| grep "Average Precision  (AP) @\[ IoU\=0.50:0.95 | area\=   all | maxDets\= 20 \]" | grep -oP ' = \K\d+\.\d+' | awk '{printf "%.3f \n", $1}')
  ap1=$(echo -e "$res"| grep "Average Precision  (AP) @\[ IoU\=0.50      | area\=   all | maxDets\= 20 \]" | grep -oP ' = \K\d+\.\d+' | awk '{printf "%.3f \n", $1}')
  printf "| %-12s | %-16s | %-34s |   %8.3f      | %8.3f   |\n" "$PLATFORM" "yolov8_$1.py" "$2" "$(printf "%.3f" $ap0)" "$(printf "%.3f" $ap1)" >> scripts/acc.txt
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
    python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$2.$1 --language=cpp --input=log/$1_$2_$3_cpp_test.log --bmodel=$3
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
  res=$(python3 ../../tools/eval_coco.py --gt_path ../../datasets/coco/person_keypoints_val2017_1000.json --result_json results/$3_val2017_1000_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log)

  echo -e "$res"

  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  judge_ret $? "$3_val2017_1000_$2_cpp_result: Precision compare!" log/$1_$2_$3_eval.log
  ap0=$(echo -e "$res"| grep "Average Precision  (AP) @\[ IoU\=0.50:0.95 | area\=   all | maxDets\= 20 \]" | grep -oP ' = \K\d+\.\d+' | awk '{printf "%.3f \n", $1}')
  ap1=$(echo -e "$res"| grep "Average Precision  (AP) @\[ IoU\=0.50      | area\=   all | maxDets\= 20 \]" | grep -oP ' = \K\d+\.\d+' | awk '{printf "%.3f \n", $1}')
  printf "| %-12s | %-16s | %-34s |   %8.3f      | %8.3f   |\n" "$PLATFORM" "yolov8_$2.$1" "$3" "$(printf "%.3f" $ap0)" "$(printf "%.3f" $ap1)" >> ../../scripts/acc.txt

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

if test $MODE = "compile_nntc"
then
  download
  compile_nntc
elif test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_build"
then
  build_pcie bmcv
elif test $MODE = "pcie_test"
then
  download
  pip3 install pycocotools opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684"
  then
    if test $CASE_MODE = "fully"
    then
      test_python opencv yolov8s-pose_fp32_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_int8_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/test
      test_python bmcv yolov8s-pose_fp32_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/test
      test_cpp pcie bmcv yolov8s-pose_fp32_1b.bmodel ../../datasets/test
      test_cpp pcie bmcv yolov8s-pose_int8_1b.bmodel ../../datasets/test
      test_cpp pcie bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/test

      test_python opencv yolov8s-pose_fp32_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_fp32_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_cpp pcie bmcv yolov8s-pose_fp32_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp pcie bmcv yolov8s-pose_int8_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp pcie bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_fp32_1b.bmodel 0.575
      eval_python opencv yolov8s-pose_int8_1b.bmodel 0.491
      eval_python opencv yolov8s-pose_int8_4b.bmodel 0.491
      eval_python bmcv yolov8s-pose_fp32_1b.bmodel   0.576
      eval_python bmcv yolov8s-pose_int8_1b.bmodel   0.490
      eval_python bmcv yolov8s-pose_int8_4b.bmodel   0.490
      eval_cpp pcie bmcv yolov8s-pose_fp32_1b.bmodel  0.576
      eval_cpp pcie bmcv yolov8s-pose_int8_1b.bmodel  0.491
      eval_cpp pcie bmcv yolov8s-pose_int8_4b.bmodel  0.491

    elif test $CASE_MODE = "partly"
    then
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/test
      test_cpp pcie bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/test

      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_cpp pcie bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_int8_4b.bmodel 0.491
      eval_python bmcv yolov8s-pose_int8_4b.bmodel   0.490
      eval_cpp pcie bmcv yolov8s-pose_int8_4b.bmodel  0.491    
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  elif test $TARGET = "BM1684X"
  then
    if test $CASE_MODE = "fully"
    then
   
      test_python opencv yolov8s-pose_fp32_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_fp16_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_int8_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/test
      test_python bmcv yolov8s-pose_fp32_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_fp16_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/test
      test_cpp pcie bmcv yolov8s-pose_fp32_1b.bmodel ../../datasets/test
      test_cpp pcie bmcv yolov8s-pose_fp16_1b.bmodel ../../datasets/test
      test_cpp pcie bmcv yolov8s-pose_int8_1b.bmodel ../../datasets/test
      test_cpp pcie bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/test

      test_python opencv yolov8s-pose_fp32_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_fp16_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_fp32_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_fp16_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_cpp pcie bmcv yolov8s-pose_fp32_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp pcie bmcv yolov8s-pose_fp16_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp pcie bmcv yolov8s-pose_int8_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp pcie bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_fp32_1b.bmodel 0.575
      eval_python opencv yolov8s-pose_fp16_1b.bmodel 0.574
      eval_python opencv yolov8s-pose_int8_1b.bmodel 0.553
      eval_python opencv yolov8s-pose_int8_4b.bmodel 0.553
      eval_python bmcv yolov8s-pose_fp32_1b.bmodel   0.574
      eval_python bmcv yolov8s-pose_fp16_1b.bmodel   0.574
      eval_python bmcv yolov8s-pose_int8_1b.bmodel   0.554
      eval_python bmcv yolov8s-pose_int8_4b.bmodel   0.554
      eval_cpp pcie bmcv yolov8s-pose_fp32_1b.bmodel  0.574
      eval_cpp pcie bmcv yolov8s-pose_fp16_1b.bmodel  0.575
      eval_cpp pcie bmcv yolov8s-pose_int8_1b.bmodel  0.553
      eval_cpp pcie bmcv yolov8s-pose_int8_4b.bmodel  0.553

    elif test $CASE_MODE = "partly"
    then
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/test
      test_cpp pcie bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/test

      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_cpp pcie bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_int8_4b.bmodel 0.553
      eval_python bmcv yolov8s-pose_int8_4b.bmodel   0.554
      eval_cpp pcie bmcv yolov8s-pose_int8_4b.bmodel  0.553
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  download
  pip3 install pycocotools opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684"
  then
    if test $CASE_MODE = "fully"
    then
      test_python opencv yolov8s-pose_fp32_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_int8_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/test
      test_python bmcv yolov8s-pose_fp32_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/test
      test_cpp soc bmcv yolov8s-pose_fp32_1b.bmodel ../../datasets/test
      test_cpp soc bmcv yolov8s-pose_int8_1b.bmodel ../../datasets/test
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/test

      test_python opencv yolov8s-pose_fp32_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_fp32_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_fp32_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_fp32_1b.bmodel 0.575
      eval_python opencv yolov8s-pose_int8_1b.bmodel 0.491
      eval_python opencv yolov8s-pose_int8_4b.bmodel 0.491
      eval_python bmcv yolov8s-pose_fp32_1b.bmodel   0.576
      eval_python bmcv yolov8s-pose_int8_1b.bmodel   0.490
      eval_python bmcv yolov8s-pose_int8_4b.bmodel   0.490
      eval_cpp soc bmcv yolov8s-pose_fp32_1b.bmodel  0.576
      eval_cpp soc bmcv yolov8s-pose_int8_1b.bmodel  0.491
      eval_cpp soc bmcv yolov8s-pose_int8_4b.bmodel  0.491

    elif test $CASE_MODE = "partly"
    then
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/test
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/test

      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_int8_4b.bmodel 0.491
      eval_python bmcv yolov8s-pose_int8_4b.bmodel   0.490
      eval_cpp soc bmcv yolov8s-pose_int8_4b.bmodel  0.491
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  elif test $TARGET = "BM1684X"
  then
    if test $CASE_MODE = "fully"
    then
      test_python opencv yolov8s-pose_fp32_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_fp16_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_int8_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/test
      test_python bmcv yolov8s-pose_fp32_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_fp16_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/test
      test_cpp soc bmcv yolov8s-pose_fp32_1b.bmodel ../../datasets/test
      test_cpp soc bmcv yolov8s-pose_fp16_1b.bmodel ../../datasets/test
      test_cpp soc bmcv yolov8s-pose_int8_1b.bmodel ../../datasets/test
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/test

      test_python opencv yolov8s-pose_fp32_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_fp16_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_fp32_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_fp16_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_fp32_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_fp16_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_fp32_1b.bmodel 0.575
      eval_python opencv yolov8s-pose_fp16_1b.bmodel 0.574
      eval_python opencv yolov8s-pose_int8_1b.bmodel 0.553
      eval_python opencv yolov8s-pose_int8_4b.bmodel 0.553
      eval_python bmcv yolov8s-pose_fp32_1b.bmodel   0.574
      eval_python bmcv yolov8s-pose_fp16_1b.bmodel   0.574
      eval_python bmcv yolov8s-pose_int8_1b.bmodel   0.554
      eval_python bmcv yolov8s-pose_int8_4b.bmodel   0.554
      eval_cpp soc bmcv yolov8s-pose_fp32_1b.bmodel  0.574
      eval_cpp soc bmcv yolov8s-pose_fp16_1b.bmodel  0.575
      eval_cpp soc bmcv yolov8s-pose_int8_1b.bmodel  0.553
      eval_cpp soc bmcv yolov8s-pose_int8_4b.bmodel  0.553

    elif test $CASE_MODE = "partly"
    then
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/test
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/test

      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_int8_4b.bmodel 0.553
      eval_python bmcv yolov8s-pose_int8_4b.bmodel   0.554
      eval_cpp soc bmcv yolov8s-pose_int8_4b.bmodel  0.553
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  elif test $TARGET = "BM1688"
  then
    if test $CASE_MODE = "fully"
    then
    
      test_python opencv yolov8s-pose_fp32_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_fp16_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_int8_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/test
      test_python bmcv yolov8s-pose_fp32_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_fp16_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/test
      test_cpp soc bmcv yolov8s-pose_fp32_1b.bmodel ../../datasets/test
      test_cpp soc bmcv yolov8s-pose_fp16_1b.bmodel ../../datasets/test
      test_cpp soc bmcv yolov8s-pose_int8_1b.bmodel ../../datasets/test
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/test
 
      test_python opencv yolov8s-pose_fp32_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_fp16_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_fp32_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_fp16_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_fp32_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_fp16_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_fp32_1b.bmodel 0.575 
      eval_python opencv yolov8s-pose_fp16_1b.bmodel 0.574
      eval_python opencv yolov8s-pose_int8_1b.bmodel 0.553
      eval_python opencv yolov8s-pose_int8_4b.bmodel 0.553
      eval_python bmcv yolov8s-pose_fp32_1b.bmodel   0.574
      eval_python bmcv yolov8s-pose_fp16_1b.bmodel   0.574
      eval_python bmcv yolov8s-pose_int8_1b.bmodel   0.550
      eval_python bmcv yolov8s-pose_int8_4b.bmodel   0.550
      eval_cpp soc bmcv yolov8s-pose_fp32_1b.bmodel  0.574
      eval_cpp soc bmcv yolov8s-pose_fp16_1b.bmodel  0.574
      eval_cpp soc bmcv yolov8s-pose_int8_1b.bmodel  0.553
      eval_cpp soc bmcv yolov8s-pose_int8_4b.bmodel  0.553

      test_python opencv yolov8s-pose_fp32_1b_2core.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_fp16_1b_2core.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_1b_2core.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_4b_2core.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_fp32_1b_2core.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_fp16_1b_2core.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_1b_2core.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b_2core.bmodel datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_fp32_1b_2core.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_fp16_1b_2core.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_1b_2core.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_4b_2core.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_fp32_1b_2core.bmodel 0.575
      eval_python opencv yolov8s-pose_fp16_1b_2core.bmodel 0.574
      eval_python opencv yolov8s-pose_int8_1b_2core.bmodel 0.553
      eval_python opencv yolov8s-pose_int8_4b_2core.bmodel 0.553
      eval_python bmcv yolov8s-pose_fp32_1b_2core.bmodel   0.574
      eval_python bmcv yolov8s-pose_fp16_1b_2core.bmodel   0.574
      eval_python bmcv yolov8s-pose_int8_1b_2core.bmodel   0.550
      eval_python bmcv yolov8s-pose_int8_4b_2core.bmodel   0.550
      eval_cpp soc bmcv yolov8s-pose_fp32_1b_2core.bmodel  0.574
      eval_cpp soc bmcv yolov8s-pose_fp16_1b_2core.bmodel  0.574
      eval_cpp soc bmcv yolov8s-pose_int8_1b_2core.bmodel  0.551
      eval_cpp soc bmcv yolov8s-pose_int8_4b_2core.bmodel  0.551

    elif test $CASE_MODE = "partly"
    then
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/test
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/test
 
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_int8_4b.bmodel 0.553
      eval_python bmcv yolov8s-pose_int8_4b.bmodel   0.550
      eval_cpp soc bmcv yolov8s-pose_int8_4b.bmodel  0.553

      test_python opencv yolov8s-pose_int8_4b_2core.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b_2core.bmodel datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_4b_2core.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_int8_4b_2core.bmodel 0.553
      eval_python bmcv yolov8s-pose_int8_4b_2core.bmodel   0.550
      eval_cpp soc bmcv yolov8s-pose_int8_4b_2core.bmodel  0.551
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  elif test $TARGET = "CV186X"
  then
    if test $CASE_MODE = "fully"
    then
      test_python opencv yolov8s-pose_fp32_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_fp16_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_int8_1b.bmodel datasets/test
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/test
      test_python bmcv yolov8s-pose_fp32_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_fp16_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_1b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/test
      test_cpp soc bmcv yolov8s-pose_fp32_1b.bmodel ../../datasets/test
      test_cpp soc bmcv yolov8s-pose_fp16_1b.bmodel ../../datasets/test
      test_cpp soc bmcv yolov8s-pose_int8_1b.bmodel ../../datasets/test
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/test
 
      test_python opencv yolov8s-pose_fp32_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_fp16_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_1b.bmodel datasets/coco/val2017_1000
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_fp32_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_fp16_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_1b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_fp32_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_fp16_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_1b.bmodel ../../datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_fp32_1b.bmodel 0.575
      eval_python opencv yolov8s-pose_fp16_1b.bmodel 0.574
      eval_python opencv yolov8s-pose_int8_1b.bmodel 0.553
      eval_python opencv yolov8s-pose_int8_4b.bmodel 0.553
      eval_python bmcv yolov8s-pose_fp32_1b.bmodel   0.574
      eval_python bmcv yolov8s-pose_fp16_1b.bmodel   0.574
      eval_python bmcv yolov8s-pose_int8_1b.bmodel   0.550
      eval_python bmcv yolov8s-pose_int8_4b.bmodel   0.550
      eval_cpp soc bmcv yolov8s-pose_fp32_1b.bmodel  0.574
      eval_cpp soc bmcv yolov8s-pose_fp16_1b.bmodel  0.574
      eval_cpp soc bmcv yolov8s-pose_int8_1b.bmodel  0.551
      eval_cpp soc bmcv yolov8s-pose_int8_4b.bmodel  0.551
    elif test $CASE_MODE = "partly"
    then
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/test
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/test
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/test
 
      test_python opencv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_python bmcv yolov8s-pose_int8_4b.bmodel datasets/coco/val2017_1000
      test_cpp soc bmcv yolov8s-pose_int8_4b.bmodel ../../datasets/coco/val2017_1000

      eval_python opencv yolov8s-pose_int8_4b.bmodel 0.553
      eval_python bmcv yolov8s-pose_int8_4b.bmodel   0.550
      eval_cpp soc bmcv yolov8s-pose_int8_4b.bmodel  0.551
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------yolov8_det mAP----------"
  cat scripts/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------yolov8_det performance-----------"
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