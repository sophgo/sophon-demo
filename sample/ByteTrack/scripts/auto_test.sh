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

usage()
{
  echo "Usage: $0 [ -m MODE |pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2
}

while getopts ":m:t:s:d:p:" opt
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

function build_pcie()
{
  pushd cpp/bytetrack_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build bytetrack_$1" 0
  popd
}

function build_soc()
{
  pushd cpp/bytetrack_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc bytetrack_$1" 0
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
  pushd cpp/bytetrack_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./bytetrack_$2.$1 --input=../../datasets/test_car_person_1080P.mp4 --bmodel_detector=../../models/$TARGET/$3 --dev_id=$TPUID  > log/$1_$2_$3_cpp_test.log 2>&1
  judge_ret $? "./bytetrack_$2.$1 --input=../../datasets/test_car_person_1080P.mp4 --bmodel_detector=../../models/$TARGET/$3 --dev_id=$TPUID" log/$1_$2_$3_cpp_test.log
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/bytetrack_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./bytetrack_$2.$1 --input=../../datasets/mot15_trainset/ADL-Rundle-6/img1 --bmodel_detector=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./bytetrack_$2.$1 --input=../../datasets/mot15_trainset/ADL-Rundle-6/img1 --bmodel_detector=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1" log/$1_$2_$3_debug.log
  tail -n 15 log/$1_$2_$3_debug.log

  echo "Evaluating..."
  res=$(python3 ../../tools/eval_mot15.py --gt_file ../../datasets/mot15_trainset/ADL-Rundle-6/gt/gt.txt --ts_file results/mot_eval/ADL-Rundle-6_$3.txt 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"

  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  judge_ret $? "cpp mot_eval/ADL-Rundle-6_$3: Precision compare!" log/$1_$2_$3_eval.log
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function test_python()
{
  pushd python
  if [ ! -d log ];then
    mkdir log
  fi
  python3 bytetrack_$1.py --input ../datasets/test_car_person_1080P.mp4 --bmodel_detector ../models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_python_test.log 2>&1
  judge_ret $? "bytetrack_$1.py --input ../datasets/test_car_person_1080P.mp4 --bmodel_detector ../models/$TARGET/$2 --dev_id $TPUID" log/$1_$2_python_test.log
  popd
}

function eval_python()
{
  echo -e "\n########################\nCase Start: eval python\n########################"
  pushd python
  if [ ! -d log ];then
    mkdir log
  fi
  python3 bytetrack_$1.py --input ../datasets/mot15_trainset/ADL-Rundle-6/img1 --bmodel_detector ../models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 bytetrack_$1.py --input ../datasets/mot15_trainset/ADL-Rundle-6/img1  --bmodel_detector ../models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_debug.log 2>&1" log/$1_$2_debug.log
  tail -n 20 log/$1_$2_debug.log

  echo "Evaluating..."
  res=$(python3 ../tools/eval_mot15.py --gt_file ../datasets/mot15_trainset/ADL-Rundle-6/gt/gt.txt --ts_file results/mot_eval/ADL-Rundle-6_$2.txt 2>&1 | tee log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
  judge_ret $? "python mot_eval/ADL-Rundle-6_$2: Precision compare!" log/$1_$2_eval.log
  popd
  echo -e "########################\nCase End: eval python\n########################\n"
}

if test $MODE = "pcie_test"
then
  download
  build_pcie bmcv
  pip3 install -r python/requirements.txt
  pip3 install motmetrics
  if test $TARGET = "BM1684"
  then
    test_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel
    test_python opencv yolov5s_v6.1_3output_int8_1b.bmodel
    test_python opencv yolov5s_v6.1_3output_int8_4b.bmodel
    test_cpp pcie bmcv yolov5s_v6.1_3output_fp32_1b.bmodel
    test_cpp pcie bmcv yolov5s_v6.1_3output_int8_1b.bmodel
    test_cpp pcie bmcv yolov5s_v6.1_3output_int8_4b.bmodel

    eval_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel 0.5140746656019166
    eval_python opencv yolov5s_v6.1_3output_int8_1b.bmodel 0.5040926332601318
    eval_python opencv yolov5s_v6.1_3output_int8_4b.bmodel 0.5040926332601318
    eval_cpp pcie bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.5022958674386104
    eval_cpp pcie bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.5000998203234178
    eval_cpp pcie bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.5000998203234178

  elif test $TARGET = "BM1684X"
  then
    test_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel
    test_python opencv yolov5s_v6.1_3output_fp16_1b.bmodel
    test_python opencv yolov5s_v6.1_3output_int8_1b.bmodel
    test_python opencv yolov5s_v6.1_3output_int8_4b.bmodel
    test_cpp pcie bmcv yolov5s_v6.1_3output_fp32_1b.bmodel
    test_cpp pcie bmcv yolov5s_v6.1_3output_fp16_1b.bmodel
    test_cpp pcie bmcv yolov5s_v6.1_3output_int8_1b.bmodel
    test_cpp pcie bmcv yolov5s_v6.1_3output_int8_4b.bmodel

    eval_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel 0.5140746656019166
    eval_python opencv yolov5s_v6.1_3output_fp16_1b.bmodel 0.5142743062487523
    eval_python opencv yolov5s_v6.1_3output_int8_1b.bmodel 0.5226592134158514
    eval_python opencv yolov5s_v6.1_3output_int8_4b.bmodel 0.5226592134158514
    eval_cpp pcie bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.5028947893791176
    eval_cpp pcie bmcv yolov5s_v6.1_3output_fp16_1b.bmodel 0.5024955080854462
    eval_cpp pcie bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.5244559792373727
    eval_cpp pcie bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.5244559792373727
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  download
  pip3 install -r python/requirements.txt --upgrade
  pip3 install opencv-python-headless motmetrics
  if test $TARGET = "BM1684"
  then
    test_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel
    test_python opencv yolov5s_v6.1_3output_int8_1b.bmodel
    test_python opencv yolov5s_v6.1_3output_int8_4b.bmodel
    test_cpp soc bmcv yolov5s_v6.1_3output_fp32_1b.bmodel
    test_cpp soc bmcv yolov5s_v6.1_3output_int8_1b.bmodel
    test_cpp soc bmcv yolov5s_v6.1_3output_int8_4b.bmodel

    eval_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel 0.5140746656019166
    eval_python opencv yolov5s_v6.1_3output_int8_1b.bmodel 0.5040926332601318
    eval_python opencv yolov5s_v6.1_3output_int8_4b.bmodel 0.5040926332601318
    eval_cpp soc bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.5022958674386104
    eval_cpp soc bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.5000998203234178
    eval_cpp soc bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.5000998203234178
  elif test $TARGET = "BM1684X"
  then
    test_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel
    test_python opencv yolov5s_v6.1_3output_fp16_1b.bmodel
    test_python opencv yolov5s_v6.1_3output_int8_1b.bmodel
    test_python opencv yolov5s_v6.1_3output_int8_4b.bmodel
    test_cpp soc bmcv yolov5s_v6.1_3output_fp32_1b.bmodel
    test_cpp soc bmcv yolov5s_v6.1_3output_fp16_1b.bmodel
    test_cpp soc bmcv yolov5s_v6.1_3output_int8_1b.bmodel
    test_cpp soc bmcv yolov5s_v6.1_3output_int8_4b.bmodel

    eval_python opencv yolov5s_v6.1_3output_fp32_1b.bmodel 0.5140746656019166
    eval_python opencv yolov5s_v6.1_3output_fp16_1b.bmodel 0.5142743062487523
    eval_python opencv yolov5s_v6.1_3output_int8_1b.bmodel 0.5226592134158514
    eval_python opencv yolov5s_v6.1_3output_int8_4b.bmodel 0.5226592134158514
    eval_cpp soc bmcv yolov5s_v6.1_3output_fp32_1b.bmodel 0.5028947893791176
    eval_cpp soc bmcv yolov5s_v6.1_3output_fp16_1b.bmodel 0.5024955080854462
    eval_cpp soc bmcv yolov5s_v6.1_3output_int8_1b.bmodel 0.5244559792373727
    eval_cpp soc bmcv yolov5s_v6.1_3output_int8_4b.bmodel 0.5244559792373727
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