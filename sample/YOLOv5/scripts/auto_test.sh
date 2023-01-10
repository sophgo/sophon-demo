#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../

TARGET="BM1684"
MODE="pcie_test"

usage() 
{
  echo "Usage: $0 [ -m MODE compile|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK]" 1>&2 
}

while getopts ":m:t:s:" opt
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
    exit 1
  fi
  sleep 3
}

function download()
{
  chmod -R +x scripts/
  ./scripts/download.sh
  judge_ret $? "download"
}

function compile()
{
  ./scripts/gen_fp32bmodel.sh BM1684
  judge_ret $? "generate BM1684 fp32bmodel"
  ./scripts/gen_fp32bmodel.sh BM1684X
  judge_ret $? "generate BM1684X fp32bmodel"
  ./scripts/gen_int8bmodel.sh BM1684
  judge_ret $? "generate BM1684 int8bmodel"
  ./scripts/gen_int8bmodel.sh BM1684X
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

function test_cpp()
{
  pushd cpp/yolov5_$2
  ./yolov5_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3
  judge_ret $? "./yolov5_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3"
  popd
}

function eval_cpp()
{
  pushd cpp/yolov5_$2
  ./yolov5_$2.$1 --input=../../datasets/coco/val2017 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.001 --nms_thresh=0.6 --obj_thresh=0.001
  judge_ret $? "./yolov5_$2.$1 --input=../../datasets/coco/val2017 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.001 --nms_thresh=0.6 --obj_thresh=0.001"
  res=$(python3 ../../tools/eval_coco.py --label_json ../../datasets/coco/instances_val2017.json --result_json results/$3_val2017_$2_cpp_result.json 2>&1)
  echo $res
  array=(${res//=/ })
  acc=${array[1]}
  if test $acc = $4
  then
    echo 'compare right!'
  else
    echo 'compare wrong!'
    exit 1
  fi
  popd
}

function test_python()
{
  python3 python/yolov5_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id 0
  judge_ret $? "python3 python/yolov5_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id 0"
}

function eval_python()
{
  python3 python/yolov5_$1.py --input datasets/coco/val2017 --bmodel models/$TARGET/$2 --dev_id 0 --conf_thresh 0.001 --nms_thresh 0.6
  judge_ret $? "python3 python/yolov5_$1.py --input datasets/coco/val2017 --bmodel models/$TARGET/$2 --dev_id 0 --conf_thresh 0.001 --nms_thresh 0.6"
  res=$(python3 tools/eval_coco.py --label_json datasets/coco/instances_val2017.json --result_json results/$2_val2017_$1_python_result.json 2>&1)
  echo $res
  array=(${res//=/ })
  acc=${array[1]}
  if test $acc = $3
  then
    echo 'compare right!'
  else
    echo 'compare wrong!'
    exit 1
  fi
}

pushd $top_dir

if test $MODE = "compile"
then
  download
  compile
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
    eval_python opencv yolov5s_v6.1_3output_int8_1b.bmodel 0.35589387754525864
    eval_python opencv yolov5s_v6.1_3output_int8_4b.bmodel 0.35589387754525864
    eval_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.36282681409398276
    eval_python bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.3449644595315241
    eval_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.3449644595315241
    eval_cpp pcie bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.3590649468018816
    eval_cpp pcie bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.34471780636574756
    eval_cpp pcie bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.34471780636574756
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
    eval_python opencv yolov5s_v6.1_3output_int8_1b.bmodel 0.3558938777923215
    eval_python opencv yolov5s_v6.1_3output_int8_4b.bmodel 0.3558938777923215
    eval_python bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.3628276247456819
    eval_python bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.34496446870309155
    eval_python bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.34496446870309155 
    eval_cpp soc bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.3590649468018816
    eval_cpp soc bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.34471780636574756
    eval_cpp soc bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.34471780636574756
  fi
fi
popd

echo 'test pass!!!'