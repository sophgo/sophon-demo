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
  pushd cpp/lprnet_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build lprnet_$1"
  popd
}

function build_soc()
{
  pushd cpp/lprnet_$1
  if [ ! -d build ]; then
      mkdir build
  fi
  cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc lprnet_$1"
  popd
}

function test_cpp()
{
  pushd cpp/lprnet_$2/build
  ./lprnet_$2.$1 ../../../data/images/test ../../../data/models/$TARGET/lprnet_$3_$4.bmodel 0
  judge_ret $? "test_cpp lprnet_$2_$1 by lprnet_$3_$4.bmodel"
  res=$(python3 ../../../tools/eval.py --label_json ../../../data/images/test_label.json --result_json results/lprnet_$3_$4.bmodel_test_$2_cpp_result.json 2>&1)
  echo $res
  array=(${res//=/ })
  acc=${array[-1]}
  if test $acc = $5
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
  python3 python/lprnet_$1.py --input_path data/images/test --bmodel data/models/$TARGET/lprnet_$2_$3.bmodel --tpu_id 0
  judge_ret $? "test_python lprnet_$1 by lprnet_$2_$3.bmodel"
  res=$(python3 tools/eval.py --label_json data/images/test_label.json --result_json results/lprnet_$2_$3.bmodel_test_$1_python_result.json 2>&1)
  echo $res
  array=(${res//=/ })
  acc=${array[-1]}
  if test $acc = $4
  then
    echo 'compare right!'
  else
    echo 'compare wrong!'
    exit 1
  fi
}


pushd $top_dir
if [ ! -d data ]; then
  download
fi

if test $MODE = "compile"
then
  compile
elif test $MODE = "pcie_test"
then
  build_pcie opencv
  build_pcie bmcv
  if test $TARGET = "BM1684"
  then
    test_cpp pcie opencv fp32 1b 0.88
    test_cpp pcie opencv fp32 4b 0.892
    test_cpp pcie opencv int8 1b 0.873
    test_cpp pcie opencv int8 4b 0.884
    test_cpp pcie bmcv fp32 1b 0.88
    test_cpp pcie bmcv fp32 4b 0.892
    test_cpp pcie bmcv int8 1b 0.873
    test_cpp pcie bmcv int8 4b 0.884
    test_python opencv fp32 1b 0.894
    test_python opencv fp32 4b 0.901
    test_python opencv int8 1b 0.887
    test_python opencv int8 4b 0.898
    test_python bmcv fp32 1b 0.882
    test_python bmcv fp32 4b 0.882
    test_python bmcv int8 1b 0.871
    test_python bmcv int8 4b 0.878
  elif test $TARGET = "BM1684X"
  then
    test_cpp pcie opencv fp32 1b 0.882
    test_cpp pcie opencv fp32 4b 0.893
    test_cpp pcie opencv int8 1b 0.874
    test_cpp pcie opencv int8 4b 0.879
    test_cpp pcie bmcv fp32 1b 0.882
    test_cpp pcie bmcv fp32 4b 0.893
    test_cpp pcie bmcv int8 1b 0.874
    test_cpp pcie bmcv int8 4b 0.879
    test_python opencv fp32 1b 0.894
    test_python opencv fp32 4b 0.901
    test_python opencv int8 1b 0.887
    test_python opencv int8 4b 0.898
    test_python bmcv fp32 1b 0.879
    test_python bmcv fp32 4b 0.882
    test_python bmcv int8 1b 0.876
    test_python bmcv int8 4b 0.878
  fi
elif test $MODE = "soc_build"
then
  build_soc opencv
  build_soc bmcv
elif test $MODE = "soc_test"
then
  if test $TARGET = "BM1684"
  then
    test_cpp soc opencv fp32 1b 0.88
    test_cpp soc opencv fp32 4b 0.892
    test_cpp soc opencv int8 1b 0.873
    test_cpp soc opencv int8 4b 0.884
    test_cpp soc bmcv fp32 1b 0.88
    test_cpp soc bmcv fp32 4b 0.892
    test_cpp soc bmcv int8 1b 0.873
    test_cpp soc bmcv int8 4b 0.884
    test_python opencv fp32 1b 0.894
    test_python opencv fp32 4b 0.901
    test_python opencv int8 1b 0.887
    test_python opencv int8 4b 0.898
    test_python bmcv fp32 1b 0.882
    test_python bmcv fp32 4b 0.882
    test_python bmcv int8 1b 0.871
    test_python bmcv int8 4b 0.878
  elif test $TARGET = "BM1684X"
  then
    test_cpp soc opencv fp32 1b 0.882
    test_cpp soc opencv fp32 4b 0.893
    test_cpp soc opencv int8 1b 0.874
    test_cpp soc opencv int8 4b 0.879
    test_cpp soc bmcv fp32 1b 0.882
    test_cpp soc bmcv fp32 4b 0.893
    test_cpp soc bmcv int8 1b 0.874
    test_cpp soc bmcv int8 4b 0.879
    test_python opencv fp32 1b 0.894
    test_python opencv fp32 4b 0.901
    test_python opencv int8 1b 0.887
    test_python opencv int8 4b 0.898
    test_python bmcv fp32 1b 0.879
    test_python bmcv fp32 4b 0.882
    test_python bmcv int8 1b 0.876
    test_python bmcv int8 4b 0.878
  fi
fi
popd

echo 'test pass!!!'