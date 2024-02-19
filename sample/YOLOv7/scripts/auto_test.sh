#!/bin/bash
scripts_dir=$(cd `dirname $BASH_SOURCE[0]`/ && pwd)
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684X"
MODE="pcie_test"
TPUID=0
ALL_PASS=1
if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi

usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [ -d TPUID]" 1>&2 
}

while getopts ":m:t:s:d:" opt
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
    ?)
      usage
      exit 1;;
  esac
done

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
  else
    echo "Failed: $2"
    ALL_PASS=0
  fi
  sleep 3
}

function download()
{
  chmod -R +x scripts/
  ./scripts/download.sh
  judge_ret $? "download"
}

function compile_nntc()
{
  ./scripts/gen_fp32bmodel_nntc.sh BM1684
  judge_ret $? "generate BM1684 fp32bmodel"
  ./scripts/gen_int8bmodel_nntc.sh BM1684
  judge_ret $? "generate BM1684 int8bmodel"
}
function compile_mlir()
{
  ./scripts/gen_fp32bmodel_mlir.sh bm1684x
  judge_ret $? "generate BM1684X fp32bmodel"
  ./scripts/gen_int8bmodel_mlir.sh bm1684x
  judge_ret $? "generate BM1684X int8bmodel"
  ./scripts/gen_fp16bmodel_mlir.sh bm1684x
  judge_ret $? "generate BM1684X fp16bmodel"
}
function build_pcie()
{
  pushd cpp/yolov7_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build yolov7_$1"
  popd
}

function build_soc()
{
  pushd cpp/yolov7_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc yolov7_$1"
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
    else
        echo "***************************************"
        echo -e "\e[42m compare right! \e[0m" #green
        echo "***************************************"
    fi
}

function test_cpp()
{
  pushd cpp/yolov7_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./yolov7_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_cpp_test.log 2>&1
  judge_ret $? "./yolov7_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID"
  tail -n 15 log/$1_$2_$3_cpp_test.log
  if test $4 = "../../datasets/coco/val2017_1000"; then
    echo "==================="
    echo "Comparing statis..."
    python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov7_$2.$1 --language=cpp --input=log/$1_$2_$3_cpp_test.log --bmodel=$3
    judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov7_$2.$1 --language=cpp --input=log/$1_$2_$3_cpp_test.log --bmodel=$3"
    echo "==================="
  fi
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/yolov7_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./yolov7_$2.$1 --input=../../datasets/coco/val2017_1000 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.001 --nms_thresh=0.65 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  tail -n 15 log/$1_$2_$3_debug.log
  judge_ret $? "./yolov7_$2.$1 --input=../../datasets/coco/val2017_1000 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.001 --nms_thresh=0.65 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1"
  
  echo "Evaluating..."
  res=$(python3 ../../tools/eval_coco.py --gt_path ../../datasets/coco/instances_val2017_1000.json --result_json results/$3_val2017_1000_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"

  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function test_python()
{
  if [ ! -d log ];then
    mkdir log
  fi
  python3 python/yolov7_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id=$TPUID > log/$1_$2_python_test.log 2>&1
  judge_ret $? "python3 python/yolov7_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id=$TPUID"
  tail -n 20 log/$1_$2_python_test.log
  if test $3 = "datasets/coco/val2017_1000"; then
    echo "==================="
    echo "Comparing statis..."
    python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov7_$1.py --language=python --input=log/$1_$2_python_test.log --bmodel=$2
    judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov7_$1.py --language=python --input=log/$1_$2_python_test.log --bmodel=$2"
    echo "==================="
  fi
}

function eval_python()
{  
  echo -e "\n########################\nCase Start: eval python\n########################"
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  python3 python/yolov7_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --dev_id=$TPUID --conf_thresh 0.001 --nms_thresh 0.65 > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/yolov7_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --dev_id=$TPUID --conf_thresh 0.001 --nms_thresh 0.65 > python/log/$1_$2_debug.log 2>&1"
  tail -n 20 python/log/$1_$2_debug.log

  echo "Evaluating..."
  res=$(python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/$2_val2017_1000_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
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
  download
  if test $TARGET = "BM1684"
  then
    test_python opencv yolov7_v0.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov7_v0.1_3output_int8_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov7_v0.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov7_v0.1_3output_int8_1b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov7_v0.1_3output_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov7_v0.1_3output_int8_1b.bmodel ../../datasets/test_car_person_1080P.mp4

    #performence test
    test_python opencv yolov7_v0.1_3output_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolov7_v0.1_3output_int8_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolov7_v0.1_3output_int8_4b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_int8_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_int8_4b.bmodel datasets/coco/val2017_1000
    test_cpp pcie bmcv yolov7_v0.1_3output_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie bmcv yolov7_v0.1_3output_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie bmcv yolov7_v0.1_3output_int8_4b.bmodel ../../datasets/coco/val2017_1000

    eval_python opencv yolov7_v0.1_3output_fp32_1b.bmodel 0.5141659367922798
    eval_python opencv yolov7_v0.1_3output_int8_1b.bmodel 0.5054503987722289
    eval_python opencv yolov7_v0.1_3output_int8_4b.bmodel 0.5054503987722289
    eval_python bmcv yolov7_v0.1_3output_fp32_1b.bmodel 0.504417540926352
    eval_python bmcv yolov7_v0.1_3output_int8_1b.bmodel 0.49731983349296716
    eval_python bmcv yolov7_v0.1_3output_int8_4b.bmodel 0.49731983349296716
    eval_cpp pcie bmcv yolov7_v0.1_3output_fp32_1b.bmodel 0.493618547647431
    eval_cpp pcie bmcv yolov7_v0.1_3output_int8_1b.bmodel 0.4865848847182539
    eval_cpp pcie bmcv yolov7_v0.1_3output_int8_4b.bmodel 0.4865848847182539

  elif test $TARGET = "BM1684X"
  then
    test_python opencv yolov7_v0.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov7_v0.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov7_v0.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov7_v0.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov7_v0.1_3output_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov7_v0.1_3output_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    #performence test
    test_python opencv yolov7_v0.1_3output_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolov7_v0.1_3output_fp16_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolov7_v0.1_3output_int8_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolov7_v0.1_3output_int8_4b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_fp16_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_int8_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_int8_4b.bmodel datasets/coco/val2017_1000
    test_cpp pcie bmcv yolov7_v0.1_3output_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie bmcv yolov7_v0.1_3output_fp16_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie bmcv yolov7_v0.1_3output_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie bmcv yolov7_v0.1_3output_int8_4b.bmodel ../../datasets/coco/val2017_1000

    eval_python opencv yolov7_v0.1_3output_fp32_1b.bmodel 0.5139793861023544
    eval_python opencv yolov7_v0.1_3output_fp16_1b.bmodel 0.5136247112750388
    eval_python opencv yolov7_v0.1_3output_int8_1b.bmodel 0.5107850878437884
    eval_python opencv yolov7_v0.1_3output_int8_4b.bmodel 0.5107850878437884
    eval_python bmcv yolov7_v0.1_3output_fp32_1b.bmodel 0.5040489236493023
    eval_python bmcv yolov7_v0.1_3output_fp16_1b.bmodel 0.5039191983893554
    eval_python bmcv yolov7_v0.1_3output_int8_1b.bmodel 0.5008662797559423
    eval_python bmcv yolov7_v0.1_3output_int8_4b.bmodel 0.5008662797559423
    eval_cpp pcie bmcv yolov7_v0.1_3output_fp32_1b.bmodel 0.4933993162110466
    eval_cpp pcie bmcv yolov7_v0.1_3output_fp16_1b.bmodel 0.49356109975723356
    eval_cpp pcie bmcv yolov7_v0.1_3output_int8_1b.bmodel 0.4916107344392691
    eval_cpp pcie bmcv yolov7_v0.1_3output_int8_4b.bmodel 0.4916107344392691
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  download
  if test $TARGET = "BM1684"
  then
    test_python opencv yolov7_v0.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov7_v0.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov7_v0.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov7_v0.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov7_v0.1_3output_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov7_v0.1_3output_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    #performence test
    test_python opencv yolov7_v0.1_3output_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolov7_v0.1_3output_int8_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolov7_v0.1_3output_int8_4b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_int8_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_int8_4b.bmodel datasets/coco/val2017_1000
    test_cpp soc bmcv yolov7_v0.1_3output_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolov7_v0.1_3output_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolov7_v0.1_3output_int8_4b.bmodel ../../datasets/coco/val2017_1000

    eval_python opencv yolov7_v0.1_3output_fp32_1b.bmodel 0.5141659367922798
    eval_python opencv yolov7_v0.1_3output_int8_1b.bmodel 0.5054503987722289
    eval_python opencv yolov7_v0.1_3output_int8_4b.bmodel 0.5054503987722289
    eval_python bmcv yolov7_v0.1_3output_fp32_1b.bmodel 0.504417540926352
    eval_python bmcv yolov7_v0.1_3output_int8_1b.bmodel 0.49731983349296716
    eval_python bmcv yolov7_v0.1_3output_int8_4b.bmodel 0.49731983349296716
    eval_cpp soc bmcv yolov7_v0.1_3output_fp32_1b.bmodel 0.493618547647431 
    eval_cpp soc bmcv yolov7_v0.1_3output_int8_1b.bmodel 0.4865848847182539
    eval_cpp soc bmcv yolov7_v0.1_3output_int8_4b.bmodel 0.4865848847182539
  elif test $TARGET = "BM1684X"
  then
    test_python opencv yolov7_v0.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov7_v0.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov7_v0.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov7_v0.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov7_v0.1_3output_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov7_v0.1_3output_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    #performence test
    test_python opencv yolov7_v0.1_3output_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolov7_v0.1_3output_fp16_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolov7_v0.1_3output_int8_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolov7_v0.1_3output_int8_4b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_fp16_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_int8_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolov7_v0.1_3output_int8_4b.bmodel datasets/coco/val2017_1000
    test_cpp soc bmcv yolov7_v0.1_3output_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolov7_v0.1_3output_fp16_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolov7_v0.1_3output_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolov7_v0.1_3output_int8_4b.bmodel ../../datasets/coco/val2017_1000

    eval_python opencv yolov7_v0.1_3output_fp32_1b.bmodel 0.514165938340715
    eval_python opencv yolov7_v0.1_3output_fp16_1b.bmodel 0.5142747966547465
    eval_python opencv yolov7_v0.1_3output_int8_1b.bmodel 0.5107850878437884
    eval_python opencv yolov7_v0.1_3output_int8_4b.bmodel 0.5107850878437884
    eval_python bmcv yolov7_v0.1_3output_fp32_1b.bmodel 0.5040489236493023
    eval_python bmcv yolov7_v0.1_3output_fp16_1b.bmodel 0.5039191983893554
    eval_python bmcv yolov7_v0.1_3output_int8_1b.bmodel 0.5008662797559423
    eval_python bmcv yolov7_v0.1_3output_int8_4b.bmodel 0.5008662797559423 
    eval_cpp soc bmcv yolov7_v0.1_3output_fp32_1b.bmodel 0.49339931066030207
    eval_cpp soc bmcv yolov7_v0.1_3output_fp16_1b.bmodel 0.49356109975723356
    eval_cpp soc bmcv yolov7_v0.1_3output_int8_1b.bmodel 0.4916107344392691
    eval_cpp soc bmcv yolov7_v0.1_3output_int8_4b.bmodel 0.4916107344392691
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