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

usage()
{
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [-a SAIL] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2
}

while getopts ":m:t:s:a:d:p:" opt
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
    ?)
      usage
      exit 1;;
  esac
done

if test $PYTEST = "pytest"
then
  >${top_dir}auto_test_result.txt
fi

function judge_ret() {
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

function compile_mlir()
{
  ./scripts/gen_fp32bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  ./scripts/gen_fp16bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp16bmodel" 0
  ./scripts/gen_int8bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET int8bmodel" 0
}

function build_pcie()
{
  pushd cpp/p2pnet_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build p2pnet_$1" 0
  popd
}

function build_soc()
{
  pushd cpp/p2pnet_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc p2pnet_$1" 0
  popd
}

function compare_res()
{
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

function test_cpp()
{
  echo -e "\n########################\nCase Start: test cpp\n########################"
  pushd cpp/p2pnet_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./p2pnet_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID > log/$1_$2_$3_cpp_test.log
  judge_ret $? "./p2pnet_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID"  log/$1_$2_$3_cpp_test.log
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/p2pnet_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./p2pnet_$2.$1 --input=../../datasets/test/images --bmodel=../../models/$TARGET/$3 --dev_id $TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./p2pnet_$2.$1 --input=../../datasets/test/images --bmodel=../../models/$TARGET/$3 --dev_id $TPUID > log/$1_$2_$3_debug.log 2>&1" log/$1_$2_$3_debug.log
  tail -n 15 log/$1_$2_$3_debug.log

  echo "Evaluating..."
  res=$(python3 ../../tools/eval_acc.py --gt_path ../../datasets/test/ground-truth --result_path results 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  judge_ret $? "cpp result: Precision compare!" log/$1_$2_$3_eval.log
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function test_python()
{
  echo -e "\n########################\nCase Start: test python\n########################"
  pushd python
  if [ ! -d log ];then
    mkdir log
  fi
  python3 p2pnet_$1.py --input $3 --bmodel ../models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_python_test.log
  judge_ret $? "python3 p2pnet_$1.py --input $3 --bmodel ../models/$TARGET/$2 --dev_id $TPUID" log/$1_$2_python_test.log
  popd
}

function eval_python()
{
  echo -e "\n########################\nCase Start: eval python\n########################"
  pushd python
  if [ ! -d log ];then
    mkdir log
  fi
  python3 p2pnet_$1.py --input ../datasets/test/images --bmodel ../models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 p2pnet_$1.py --input ../datasets/test/images --bmodel ../models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_debug.log 2>&1" log/$1_$2_debug.log
  tail -n 20 log/$1_$2_debug.log

  echo "Evaluating..."
  res=$(python3 ../tools/eval_acc.py --gt_path ../datasets/test/ground-truth --result_path results 2>&1 | tee log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
  judge_ret $? "python result: Precision compare!" log/$1_$2_eval.log
  popd
  echo -e "########################\nCase End: eval python\n########################\n"
}

if test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_build"
then
  build_pcie bmcv
elif test $MODE = "pcie_test"
then
  download
  build_pcie bmcv
  pip3 install -r python/requirements.txt
  if test $TARGET = "BM1684"
  then
    test_python opencv p2pnet_bm1684_fp32_1b.bmodel ../datasets/test/images
    test_python opencv p2pnet_bm1684_int8_1b.bmodel ../datasets/test/images
    test_python opencv p2pnet_bm1684_int8_4b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684_fp32_1b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684_int8_1b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684_int8_4b.bmodel ../datasets/test/images
    test_cpp pcie bmcv p2pnet_bm1684_fp32_1b.bmodel ../../datasets/test/images
    test_cpp pcie bmcv p2pnet_bm1684_int8_1b.bmodel ../../datasets/test/images
    test_cpp pcie bmcv p2pnet_bm1684_int8_4b.bmodel ../../datasets/test/images

    eval_python opencv p2pnet_bm1684_fp32_1b.bmodel 18.35126582278481
    eval_python opencv p2pnet_bm1684_int8_1b.bmodel 20.443037974683545
    eval_python opencv p2pnet_bm1684_int8_4b.bmodel 20.443037974683545
    eval_python bmcv p2pnet_bm1684_fp32_1b.bmodel 20.199367088607595
    eval_python bmcv p2pnet_bm1684_int8_1b.bmodel 20.664556962025316
    eval_python bmcv p2pnet_bm1684_int8_4b.bmodel 20.664556962025316
    eval_cpp pcie bmcv p2pnet_bm1684_fp32_1b.bmodel 18.14873417721519
    eval_cpp pcie bmcv p2pnet_bm1684_int8_1b.bmodel 19.911392405063292
    eval_cpp pcie bmcv p2pnet_bm1684_int8_4b.bmodel 19.911392405063292
  elif test $TARGET = "BM1684X"
  then
    test_python opencv p2pnet_bm1684x_fp32_1b.bmodel ../datasets/test/images
    test_python opencv p2pnet_bm1684x_fp16_1b.bmodel ../datasets/test/images
    test_python opencv p2pnet_bm1684x_int8_1b.bmodel ../datasets/test/images
    test_python opencv p2pnet_bm1684x_int8_4b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684x_fp32_1b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684x_fp16_1b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684x_int8_1b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684x_int8_4b.bmodel ../datasets/test/images
    test_cpp pcie bmcv p2pnet_bm1684x_fp32_1b.bmodel ../../datasets/test/images
    test_cpp pcie bmcv p2pnet_bm1684x_fp16_1b.bmodel ../../datasets/test/images
    test_cpp pcie bmcv p2pnet_bm1684x_int8_1b.bmodel ../../datasets/test/images
    test_cpp pcie bmcv p2pnet_bm1684x_int8_4b.bmodel ../../datasets/test/images

    eval_python opencv p2pnet_bm1684x_fp32_1b.bmodel 18.35126582278481
    eval_python opencv p2pnet_bm1684x_fp16_1b.bmodel 18.341772151898734
    eval_python opencv p2pnet_bm1684x_int8_1b.bmodel 18.490506329113924
    eval_python opencv p2pnet_bm1684x_int8_4b.bmodel 18.490506329113924
    eval_python bmcv p2pnet_bm1684x_fp32_1b.bmodel 20.21518987341772
    eval_python bmcv p2pnet_bm1684x_fp16_1b.bmodel 20.20886075949367
    eval_python bmcv p2pnet_bm1684x_int8_1b.bmodel 20.335443037974684
    eval_python bmcv p2pnet_bm1684x_int8_4b.bmodel 20.335443037974684
    eval_cpp pcie bmcv p2pnet_bm1684x_fp32_1b.bmodel 18.056962025316455
    eval_cpp pcie bmcv p2pnet_bm1684x_fp16_1b.bmodel 18.088607594936708
    eval_cpp pcie bmcv p2pnet_bm1684x_int8_1b.bmodel 17.990506329113924
    eval_cpp pcie bmcv p2pnet_bm1684x_int8_4b.bmodel 17.990506329113924
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  download
  if test $TARGET = "BM1684"
  then
    test_python opencv p2pnet_bm1684_fp32_1b.bmodel ../datasets/test/images
    test_python opencv p2pnet_bm1684_int8_1b.bmodel ../datasets/test/images
    test_python opencv p2pnet_bm1684_int8_4b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684_fp32_1b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684_int8_1b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684_int8_4b.bmodel ../datasets/test/images
    test_cpp soc bmcv p2pnet_bm1684_fp32_1b.bmodel ../../datasets/test/images
    test_cpp soc bmcv p2pnet_bm1684_int8_1b.bmodel ../../datasets/test/images
    test_cpp soc bmcv p2pnet_bm1684_int8_4b.bmodel ../../datasets/test/images

    eval_python opencv p2pnet_bm1684_fp32_1b.bmodel 18.35126582278481
    eval_python opencv p2pnet_bm1684_int8_1b.bmodel 20.443037974683545
    eval_python opencv p2pnet_bm1684_int8_4b.bmodel 20.443037974683545
    eval_python bmcv p2pnet_bm1684_fp32_1b.bmodel 20.199367088607595
    eval_python bmcv p2pnet_bm1684_int8_1b.bmodel 20.664556962025316
    eval_python bmcv p2pnet_bm1684_int8_4b.bmodel 20.664556962025316
    eval_cpp soc bmcv p2pnet_bm1684_fp32_1b.bmodel 18.14873417721519
    eval_cpp soc bmcv p2pnet_bm1684_int8_1b.bmodel 19.911392405063292
    eval_cpp soc bmcv p2pnet_bm1684_int8_4b.bmodel 19.911392405063292
  elif test $TARGET = "BM1684X"
  then
    test_python opencv p2pnet_bm1684x_fp32_1b.bmodel ../datasets/test/images
    test_python opencv p2pnet_bm1684x_fp16_1b.bmodel ../datasets/test/images
    test_python opencv p2pnet_bm1684x_int8_1b.bmodel ../datasets/test/images
    test_python opencv p2pnet_bm1684x_int8_4b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684x_fp32_1b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684x_fp16_1b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684x_int8_1b.bmodel ../datasets/test/images
    test_python bmcv p2pnet_bm1684x_int8_4b.bmodel ../datasets/test/images
    test_cpp soc bmcv p2pnet_bm1684x_fp32_1b.bmodel ../../datasets/test/images
    test_cpp soc bmcv p2pnet_bm1684x_fp16_1b.bmodel ../../datasets/test/images
    test_cpp soc bmcv p2pnet_bm1684x_int8_1b.bmodel ../../datasets/test/images
    test_cpp soc bmcv p2pnet_bm1684x_int8_4b.bmodel ../../datasets/test/images

    eval_python opencv p2pnet_bm1684x_fp32_1b.bmodel 18.35126582278481
    eval_python opencv p2pnet_bm1684x_fp16_1b.bmodel 18.341772151898734
    eval_python opencv p2pnet_bm1684x_int8_1b.bmodel 18.490506329113924
    eval_python opencv p2pnet_bm1684x_int8_4b.bmodel 18.490506329113924
    eval_python bmcv p2pnet_bm1684x_fp32_1b.bmodel 20.21518987341772
    eval_python bmcv p2pnet_bm1684x_fp16_1b.bmodel 20.20886075949367
    eval_python bmcv p2pnet_bm1684x_int8_1b.bmodel 20.335443037974684
    eval_python bmcv p2pnet_bm1684x_int8_4b.bmodel 20.335443037974684
    eval_cpp soc bmcv p2pnet_bm1684x_fp32_1b.bmodel 18.056962025316455
    eval_cpp soc bmcv p2pnet_bm1684x_fp16_1b.bmodel 18.088607594936708
    eval_cpp soc bmcv p2pnet_bm1684x_int8_1b.bmodel 17.990506329113924
    eval_cpp soc bmcv p2pnet_bm1684x_int8_4b.bmodel 17.990506329113924
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
