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
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2 
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

function build_pcie()
{
  pushd cpp/deepsort_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build deepsort_$1" 0
  popd
}

function build_soc()
{
  pushd cpp/deepsort_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc deepsort_$1" 0
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
  pushd cpp/deepsort_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./deepsort_$2.$1 --input=../../datasets/test_car_person_1080P.mp4 --bmodel_detector=../../models/$TARGET/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_cpp_test.log 2>&1
  judge_ret $? "./deepsort_$2.$1 --input=../../datasets/test_car_person_1080P.mp4 --bmodel_detector=../../models/$TARGET/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor=../../models/$TARGET/$3 --dev_id=$TPUID" log/$1_$2_$3_cpp_test.log
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/deepsort_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./deepsort_$2.$1 --input=../../datasets/mot15_trainset/ADL-Rundle-6/img1 --bmodel_detector=../../models/$TARGET/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./deepsort_$2.$1 --input=../../datasets/mot15_trainset/ADL-Rundle-6/img1 --bmodel_detector=../../models/$TARGET/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1" log/$1_$2_$3_debug.log
  tail -n 20 log/$1_$2_$3_debug.log

  echo "Evaluating..."
  res=$(python3 ../../tools/eval_mot15.py --gt_file ../../datasets/mot15_trainset/ADL-Rundle-6/gt/gt.txt --ts_file results/mot_eval/ADL-Rundle-6_$3.txt 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"

  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  judge_ret $? "mot_eval/ADL-Rundle-6_$3: Precision compare!" log/$1_$2_$3_eval.log
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function test_python()
{
  pushd python
  if [ ! -d log ];then
    mkdir log
  fi
  python3 deepsort_$1.py --input ../datasets/test_car_person_1080P.mp4 --bmodel_detector ../models/$TARGET/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor ../models/$TARGET/$2 --dev_id=$TPUID > log/$1_$2_python_test.log 2>&1
  judge_ret $? "deepsort_$1.py --input ../datasets/test_car_person_1080P.mp4 --bmodel_detector ../models/$TARGET/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor ../models/$TARGET/$2 --dev_id=$TPUID" log/$1_$2_python_test.log
  popd
}

function eval_python()
{ 
  echo -e "\n########################\nCase Start: eval python\n########################"
  pushd python
  if [ ! -d log ];then
    mkdir log
  fi
  python3 deepsort_$1.py --input ../datasets/mot15_trainset/ADL-Rundle-6/img1 --bmodel_detector ../models/$TARGET/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor ../models/$TARGET/$2 --dev_id=$TPUID > log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 deepsort_$1.py --input ../datasets/mot15_trainset/ADL-Rundle-6/img1 --bmodel_detector ../models/$TARGET/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor ../models/$TARGET/$2 --dev_id=$TPUID > log/$1_$2_debug.log 2>&1" log/$1_$2_debug.log
  tail -n 30 log/$1_$2_debug.log | head -n 20
  
  echo "Evaluating..."
  res=$(python3 ../tools/eval_mot15.py --gt_file ../datasets/mot15_trainset/ADL-Rundle-6/gt/gt.txt --ts_file results/mot_eval/ADL-Rundle-6_$2.txt 2>&1 | tee log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
  judge_ret $? "mot_eval/ADL-Rundle-6_$2: Precision compare!" log/$1_$2_eval.log
  popd
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
  download
  build_pcie bmcv
  pip3 install -r python/requirements.txt
  pip3 install motmetrics
  if test $TARGET = "BM1684"
  then
    test_python opencv extractor_fp32_1b.bmodel
    test_python opencv extractor_fp32_4b.bmodel
    test_python opencv extractor_int8_1b.bmodel
    test_python opencv extractor_int8_4b.bmodel
    test_cpp pcie bmcv extractor_fp32_1b.bmodel
    test_cpp pcie bmcv extractor_fp32_4b.bmodel
    test_cpp pcie bmcv extractor_int8_1b.bmodel
    test_cpp pcie bmcv extractor_int8_4b.bmodel

    eval_python opencv extractor_fp32_1b.bmodel 0.45717708125374323
    eval_python opencv extractor_fp32_4b.bmodel 0.45717708125374323
    eval_python opencv extractor_int8_1b.bmodel 0.45897384707526456
    eval_python opencv extractor_int8_4b.bmodel 0.45897384707526456
    eval_cpp pcie bmcv extractor_fp32_1b.bmodel 0.4497903773208225
    eval_cpp pcie bmcv extractor_fp32_4b.bmodel 0.4497903773208225
    eval_cpp pcie bmcv extractor_int8_1b.bmodel 0.4523857057296866
    eval_cpp pcie bmcv extractor_int8_4b.bmodel 0.4523857057296866

  elif test $TARGET = "BM1684X"
  then
    test_python opencv extractor_fp32_1b.bmodel
    test_python opencv extractor_fp32_4b.bmodel
    test_python opencv extractor_fp16_1b.bmodel
    test_python opencv extractor_fp16_4b.bmodel
    test_python opencv extractor_int8_1b.bmodel
    test_python opencv extractor_int8_4b.bmodel
    test_cpp pcie bmcv extractor_fp32_1b.bmodel
    test_cpp pcie bmcv extractor_fp32_4b.bmodel
    test_cpp pcie bmcv extractor_fp16_1b.bmodel
    test_cpp pcie bmcv extractor_fp16_4b.bmodel
    test_cpp pcie bmcv extractor_int8_1b.bmodel
    test_cpp pcie bmcv extractor_int8_4b.bmodel

    eval_python opencv extractor_fp32_1b.bmodel 0.43940906368536636
    eval_python opencv extractor_fp32_4b.bmodel 0.43940906368536636
    eval_python opencv extractor_fp16_1b.bmodel 0.43940906368536636
    eval_python opencv extractor_fp16_4b.bmodel 0.43940906368536636
    eval_python opencv extractor_int8_1b.bmodel 0.43601517268915946
    eval_python opencv extractor_int8_4b.bmodel 0.43601517268915946
    eval_cpp pcie bmcv extractor_fp32_1b.bmodel 0.44200439209423037
    eval_cpp pcie bmcv extractor_fp32_4b.bmodel 0.44200439209423037
    eval_cpp pcie bmcv extractor_fp16_1b.bmodel 0.44200439209423037
    eval_cpp pcie bmcv extractor_fp16_4b.bmodel 0.44200439209423037
    eval_cpp pcie bmcv extractor_int8_1b.bmodel 0.43761229786384503
    eval_cpp pcie bmcv extractor_int8_4b.bmodel 0.43761229786384503
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
    test_python opencv extractor_fp32_1b.bmodel
    test_python opencv extractor_fp32_4b.bmodel
    test_python opencv extractor_int8_1b.bmodel
    test_python opencv extractor_int8_4b.bmodel
    test_cpp soc bmcv extractor_fp32_1b.bmodel
    test_cpp soc bmcv extractor_fp32_4b.bmodel
    test_cpp soc bmcv extractor_int8_1b.bmodel
    test_cpp soc bmcv extractor_int8_4b.bmodel

    eval_python opencv extractor_fp32_1b.bmodel 0.45717708125374323
    eval_python opencv extractor_fp32_4b.bmodel 0.45717708125374323
    eval_python opencv extractor_int8_1b.bmodel 0.45897384707526456
    eval_python opencv extractor_int8_4b.bmodel 0.45897384707526456
    eval_cpp soc bmcv extractor_fp32_1b.bmodel 0.4497903773208225
    eval_cpp soc bmcv extractor_fp32_4b.bmodel 0.4497903773208225
    eval_cpp soc bmcv extractor_int8_1b.bmodel 0.4523857057296866
    eval_cpp soc bmcv extractor_int8_4b.bmodel 0.4523857057296866
  elif test $TARGET = "BM1684X"
  then
    test_python opencv extractor_fp32_1b.bmodel
    test_python opencv extractor_fp32_4b.bmodel
    test_python opencv extractor_fp16_1b.bmodel
    test_python opencv extractor_fp16_4b.bmodel
    test_python opencv extractor_int8_1b.bmodel
    test_python opencv extractor_int8_4b.bmodel
    test_cpp soc bmcv extractor_fp32_1b.bmodel
    test_cpp soc bmcv extractor_fp32_4b.bmodel
    test_cpp soc bmcv extractor_fp16_1b.bmodel
    test_cpp soc bmcv extractor_fp16_4b.bmodel
    test_cpp soc bmcv extractor_int8_1b.bmodel
    test_cpp soc bmcv extractor_int8_4b.bmodel

    eval_python opencv extractor_fp32_1b.bmodel 0.43940906368536636
    eval_python opencv extractor_fp32_4b.bmodel 0.43940906368536636
    eval_python opencv extractor_fp16_1b.bmodel 0.43940906368536636
    eval_python opencv extractor_fp16_4b.bmodel 0.43940906368536636
    eval_python opencv extractor_int8_1b.bmodel 0.43601517268915946
    eval_python opencv extractor_int8_4b.bmodel 0.43601517268915946
    eval_cpp soc bmcv extractor_fp32_1b.bmodel  0.44200439209423037
    eval_cpp soc bmcv extractor_fp32_4b.bmodel  0.44200439209423037
    eval_cpp soc bmcv extractor_fp16_1b.bmodel  0.44200439209423037
    eval_cpp soc bmcv extractor_fp16_4b.bmodel  0.44200439209423037
    eval_cpp soc bmcv extractor_int8_1b.bmodel  0.43761229786384503
    eval_cpp soc bmcv extractor_int8_4b.bmodel  0.43761229786384503
  elif test $TARGET = "BM1688"
  then
    test_python opencv extractor_fp32_1b.bmodel
    test_python opencv extractor_fp32_4b.bmodel
    test_python opencv extractor_fp16_1b.bmodel
    test_python opencv extractor_fp16_4b.bmodel
    test_python opencv extractor_int8_1b.bmodel
    test_python opencv extractor_int8_4b.bmodel
    test_cpp soc bmcv extractor_fp32_1b.bmodel
    test_cpp soc bmcv extractor_fp32_4b.bmodel
    test_cpp soc bmcv extractor_fp16_1b.bmodel
    test_cpp soc bmcv extractor_fp16_4b.bmodel
    test_cpp soc bmcv extractor_int8_1b.bmodel
    test_cpp soc bmcv extractor_int8_4b.bmodel

    eval_python opencv extractor_fp32_1b.bmodel 0.441
    eval_python opencv extractor_fp32_4b.bmodel 0.441
    eval_python opencv extractor_fp16_1b.bmodel 0.441
    eval_python opencv extractor_fp16_4b.bmodel 0.441
    eval_python opencv extractor_int8_1b.bmodel 0.440
    eval_python opencv extractor_int8_4b.bmodel 0.440
    eval_cpp soc bmcv extractor_fp32_1b.bmodel  0.430
    eval_cpp soc bmcv extractor_fp32_4b.bmodel  0.430
    eval_cpp soc bmcv extractor_fp16_1b.bmodel  0.430
    eval_cpp soc bmcv extractor_fp16_4b.bmodel  0.430
    eval_cpp soc bmcv extractor_int8_1b.bmodel  0.429
    eval_cpp soc bmcv extractor_int8_4b.bmodel  0.429

    eval_python opencv extractor_fp32_1b_2core.bmodel 0.441
    eval_python opencv extractor_fp32_4b_2core.bmodel 0.441
    eval_python opencv extractor_fp16_1b_2core.bmodel 0.441
    eval_python opencv extractor_fp16_4b_2core.bmodel 0.441
    eval_python opencv extractor_int8_1b_2core.bmodel 0.440
    eval_python opencv extractor_int8_4b_2core.bmodel 0.440
    eval_cpp soc bmcv extractor_fp32_1b_2core.bmodel  0.430
    eval_cpp soc bmcv extractor_fp32_4b_2core.bmodel  0.430
    eval_cpp soc bmcv extractor_fp16_1b_2core.bmodel  0.430
    eval_cpp soc bmcv extractor_fp16_4b_2core.bmodel  0.430
    eval_cpp soc bmcv extractor_int8_1b_2core.bmodel  0.429
    eval_cpp soc bmcv extractor_int8_4b_2core.bmodel  0.429
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