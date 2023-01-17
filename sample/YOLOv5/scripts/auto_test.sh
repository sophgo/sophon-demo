#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684X"
MODE="pcie_test"
TPUID=0
ALL_PASS=1

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
  ./scripts/gen_fp16bmodel_mlir.sh bm1684x
  judge_ret $? "generate BM1684X fp16bmodel"
  ./scripts/gen_int8bmodel_mlir.sh bm1684x
  judge_ret $? "generate BM1684X int8bmodel"
}

function build_pcie()
{
  pushd cpp/yolov5_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build yolov5_$1"
  popd
}

function build_soc()
{
  pushd cpp/yolov5_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc yolov5_$1"
  popd
}

function compare_res(){
    ret=`awk -v x=$1 -v y=$2 'BEGIN{print(x-y<0.0001 && y-x<0.0001)?1:0}'`
    if [ $ret -eq 0 ]
    then
        ALL_PASS=0
        echo "compare wrong!"
    else
        echo "compare right!"
    fi
}

function test_cpp()
{
  pushd cpp/yolov5_$2
  ./yolov5_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID
  judge_ret $? "./yolov5_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID"
  popd
}

function eval_cpp()
{
  pushd cpp/yolov5_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./yolov5_$2.$1 --input=../../datasets/coco/val2017 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.001 --nms_thresh=0.6 --obj_thresh=0.001 --dev_id $TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./yolov5_$2.$1 --input=../../datasets/coco/val2017 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.001 --nms_thresh=0.6 --obj_thresh=0.001 --dev_id $TPUID > log/$1_$2_$3_debug.log 2>&1"
  echo "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
  echo "python3 ../../tools/eval_coco.py --label_json ../../datasets/coco/instances_val2017.json --result_json results/$3_val2017_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log"
  echo "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
  res=$(python3 ../../tools/eval_coco.py --label_json ../../datasets/coco/instances_val2017.json --result_json results/$3_val2017_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log)
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  popd
}

function test_python()
{
  python3 python/yolov5_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID
  judge_ret $? "python3 python/yolov5_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID"
}

function eval_python()
{  
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  python3 python/yolov5_$1.py --input datasets/coco/val2017 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.001 --nms_thresh 0.6 > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/yolov5_$1.py --input datasets/coco/val2017 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.001 --nms_thresh 0.6 > python/log/$1_$2_debug.log 2>&1"
  echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
  echo "python3 tools/eval_coco.py --label_json datasets/coco/instances_val2017.json --result_json results/$2_val2017_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log"
  echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
  res=$(python3 tools/eval_coco.py --label_json datasets/coco/instances_val2017.json --result_json results/$2_val2017_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
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
    test_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test
    test_python opencv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test
    test_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test
    test_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test
    test_cpp pcie bmcv yolov5s_v6.1_3output_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv yolov5s_v6.1_3output_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov5s_v6.1_3output_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov5s_v6.1_3output_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel 0.3728494154948667
    eval_python opencv yolov5s_v6.1_3output_int8_1b.bmodel 0.3415060943647802
    eval_python opencv yolov5s_v6.1_3output_int8_4b.bmodel 0.3415060943647802
    eval_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.3627150044630879
    eval_python bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.3302358654479303
    eval_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.3302358654479303
    eval_cpp pcie bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.35903516521721296
    eval_cpp pcie bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.3317881837937532
    eval_cpp pcie bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.3317881837937532

  elif test $TARGET = "BM1684X"
  then
    test_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test
    test_python opencv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test
    test_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test
    test_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test
    test_cpp pcie bmcv yolov5s_v6.1_3output_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv yolov5s_v6.1_3output_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov5s_v6.1_3output_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov5s_v6.1_3output_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel 0.372849263961394
    eval_python opencv yolov5s_v6.1_3output_fp16_1b.bmodel 0.3727984936413759
    eval_python opencv yolov5s_v6.1_3output_int8_1b.bmodel 0.356805022950668
    eval_python opencv yolov5s_v6.1_3output_int8_4b.bmodel 0.35783828515718147
    eval_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.36282681409398276
    eval_python bmcv yolov5s_v6.1_3output_fp16_1b.bmodel 0.3627372334554521
    eval_python bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.34641547670369127
    eval_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.34641547670369127
    eval_cpp pcie bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.3590649468018816
    eval_cpp pcie bmcv yolov5s_v6.1_3output_fp16_1b.bmodel 0.35900734830593534
    eval_cpp pcie bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.344157647308505
    eval_cpp pcie bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.344157647308505
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  download
  if test $TARGET = "BM1684"
  then
    test_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test
    test_python opencv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test
    test_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test
    test_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test
    test_cpp soc bmcv yolov5s_v6.1_3output_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv yolov5s_v6.1_3output_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov5s_v6.1_3output_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov5s_v6.1_3output_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel 0.3728494154948667
    eval_python opencv yolov5s_v6.1_3output_int8_1b.bmodel 0.3415060943647802
    eval_python opencv yolov5s_v6.1_3output_int8_4b.bmodel 0.3415060943647802
    eval_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.3627150044630879
    eval_python bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.3302358654479303
    eval_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.3302358654479303
    eval_cpp soc bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.35903516521721296 
    eval_cpp soc bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.3317881837937532
    eval_cpp soc bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.3317881837937532
  elif test $TARGET = "BM1684X"
  then
    test_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test
    test_python opencv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test
    test_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test
    test_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test
    test_cpp soc bmcv yolov5s_v6.1_3output_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv yolov5s_v6.1_3output_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov5s_v6.1_3output_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov5s_v6.1_3output_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel 0.37284926941692986
    # eval_python opencv yolov5s_v6.1_3output_fp16_1b.bmodel 0.37279000669409373
    eval_python opencv yolov5s_v6.1_3output_int8_1b.bmodel 0.3568049563925935
    eval_python opencv yolov5s_v6.1_3output_int8_4b.bmodel 0.3568049563925935
    eval_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.3628276247456819
    # eval_python bmcv yolov5s_v6.1_3output_fp16_1b.bmodel 0.3628276247456819
    eval_python bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.3464152996599053
    eval_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.3464152996599053 
    eval_cpp soc bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.3590649468018816
    # eval_cpp soc bmcv yolov5s_v6.1_3output_fp16_1b.bmodel 0.3590649468018816
    eval_cpp soc bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.344157647308505
    eval_cpp soc bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.344157647308505
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