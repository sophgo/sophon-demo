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
  pushd cpp/openpose_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build openpose_$1"
  popd
}

function build_soc()
{
  pushd cpp/openpose_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc openpose_$1"
  popd
}

function test_cpp()
{
  pushd cpp/openpose_$2
  ./openpose_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3
  judge_ret $? "./openpose_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3"
  popd
}

function eval_cpp()
{
  pushd cpp/openpose_$2
  ./openpose_$2.$1 --input=../../datasets/coco/val2017 --bmodel=../../models/$TARGET/$3
  judge_ret $? "./openpose_$2.$1 --input=../../datasets/coco/val2017 --bmodel=../../models/$TARGET/$3"
  res=$(python3 ../../tools/eval_coco.py --label_json ../../datasets/coco/person_keypoints_val2017.json --result_json results/$3_val2017_$2_cpp_result.json 2>&1)
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
  python3 python/openpose_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id 0
  judge_ret $? "python3 python/openpose_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id 0"
}

function eval_python()
{
  python3 python/openpose_$1.py --input datasets/coco/val2017 --bmodel models/$TARGET/$2 --dev_id 0
  judge_ret $? "python3 python/openpose_$1.py --input datasets/coco/val2017 --bmodel models/$TARGET/$2 --dev_id 0"
  res=$(python3 tools/eval_coco.py --label_json datasets/coco/person_keypoints_val2017.json --result_json results/$2_val2017_$1_python_result.json 2>&1)
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
    test_python opencv pose_coco_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_int8_4b.bmodel datasets/test
    test_python opencv pose_body_25_fp32_1b.bmodel datasets/test
    test_cpp pcie bmcv pose_coco_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv pose_coco_int8_4b.bmodel ../../datasets/test
    test_cpp pcie bmcv pose_body_25_fp32_1b.bmodel ../../datasets/test
    test_python opencv pose_coco_fp32_1b.bmodel datasets/dance_1080P.mp4
    test_python opencv pose_coco_int8_4b.bmodel datasets/dance_1080P.mp4
    test_cpp pcie bmcv pose_coco_fp32_1b.bmodel ../../datasets/dance_1080P.mp4
    test_cpp pcie bmcv pose_coco_int8_4b.bmodel ../../datasets/dance_1080P.mp4

    eval_python opencv pose_coco_fp32_1b.bmodel 0.4079548951880353
    eval_python opencv pose_coco_int8_1b.bmodel 0.3867737295620169
    eval_python opencv pose_coco_int8_4b.bmodel 0.3867737295620169
    eval_cpp pcie bmcv pose_coco_fp32_1b.bmodel 0.3946956331797582
    eval_cpp pcie bmcv pose_coco_int8_1b.bmodel 0.3742159412587507
    eval_cpp pcie bmcv pose_coco_int8_4b.bmodel 0.3742159412587507

  elif test $TARGET = "BM1684X"
  then
    test_python opencv pose_coco_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_int8_4b.bmodel datasets/test
    test_python opencv pose_body_25_fp32_1b.bmodel datasets/test
    test_cpp pcie bmcv pose_coco_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv pose_coco_int8_4b.bmodel ../../datasets/test
    test_cpp pcie bmcv pose_body_25_fp32_1b.bmodel ../../datasets/test
    test_python opencv pose_coco_fp32_1b.bmodel datasets/dance_1080P.mp4
    test_python opencv pose_coco_int8_4b.bmodel datasets/dance_1080P.mp4
    test_cpp pcie bmcv pose_coco_fp32_1b.bmodel ../../datasets/dance_1080P.mp4
    test_cpp pcie bmcv pose_coco_int8_4b.bmodel ../../datasets/dance_1080P.mp4

    eval_python opencv pose_coco_fp32_1b.bmodel 0.4079558333212888
    eval_python opencv pose_coco_int8_1b.bmodel 0.38610751552648426
    eval_python opencv pose_coco_int8_4b.bmodel 0.38610751552648426
    eval_cpp pcie bmcv pose_coco_fp32_1b.bmodel 0.3948540354884685
    eval_cpp pcie bmcv pose_coco_int8_1b.bmodel 0.3744470284180181
    eval_cpp pcie bmcv pose_coco_int8_4b.bmodel 0.3744470284180181
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  download
  if test $TARGET = "BM1684"
  then
    test_python opencv pose_coco_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_int8_4b.bmodel datasets/test
    test_python opencv pose_body_25_fp32_1b.bmodel datasets/test
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_body_25_fp32_1b.bmodel ../../datasets/test
    test_python opencv pose_coco_fp32_1b.bmodel datasets/dance_1080P.mp4
    test_python opencv pose_coco_int8_4b.bmodel datasets/dance_1080P.mp4
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/dance_1080P.mp4
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/dance_1080P.mp4

    eval_python opencv pose_coco_fp32_1b.bmodel 0.40847134571076393
    eval_python opencv pose_coco_int8_1b.bmodel 0.3856922046874407
    eval_python opencv pose_coco_int8_4b.bmodel 0.3856922046874407
    eval_cpp soc bmcv pose_coco_fp32_1b.bmodel 0.39452899322916646
    eval_cpp soc bmcv pose_coco_int8_1b.bmodel 0.3742180524078875
    eval_cpp soc bmcv pose_coco_int8_4b.bmodel 0.3742180524078875
  elif test $TARGET = "BM1684X"
  then
    test_python opencv pose_coco_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_int8_4b.bmodel datasets/test
    test_python opencv pose_body_25_fp32_1b.bmodel datasets/test
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_body_25_fp32_1b.bmodel ../../datasets/test
    test_python opencv pose_coco_fp32_1b.bmodel datasets/dance_1080P.mp4
    test_python opencv pose_coco_int8_4b.bmodel datasets/dance_1080P.mp4
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/dance_1080P.mp4
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/dance_1080P.mp4

    eval_python opencv pose_coco_fp32_1b.bmodel 0.4084711176538226
    eval_python opencv pose_coco_int8_1b.bmodel 0.3856922046874407
    eval_python opencv pose_coco_int8_4b.bmodel 0.3856922046874407
    eval_cpp soc bmcv pose_coco_fp32_1b.bmodel 0.3948943742856824
    eval_cpp soc bmcv pose_coco_int8_1b.bmodel 0.37446997024676
    eval_cpp soc bmcv pose_coco_int8_4b.bmodel 0.37446997024676
  fi
fi
popd

echo 'test pass!!!'