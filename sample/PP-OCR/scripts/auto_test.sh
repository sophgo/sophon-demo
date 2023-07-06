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
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [-a SAIL] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2 
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
  pushd cpp/ppocr_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build ppocr_$1" 0
  popd
}

function build_soc()
{
  pushd cpp/ppocr_$1
  if [ -d build ]; then
      rm -rf build
  fi
  if test $1 = "sail"; then
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK -DSAIL_PATH=$SAIL_PATH && make
    judge_ret $? "build soc ppocr_$1" 0
  else
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
    judge_ret $? "build soc ppocr_$1" 0
  fi
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

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/ppocr_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./ppocr_$2.$1 --input=../../datasets/train_full_images_0 --batch_size=4 --bmodel_det=../../models/$TARGET/ch_PP-OCRv3_det_$3.bmodel \
                                                                --bmodel_cls=../../models/$TARGET/ch_PP-OCRv3_cls_$3.bmodel \
                                                                --bmodel_rec=../../models/$TARGET/ch_PP-OCRv3_rec_$3.bmodel \
                                                                --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "$1 $2 $3 cpp debug" log/$1_$2_$3_debug.log
  tail -n 15 log/$1_$2_$3_debug.log
  echo "Evaluating..."
  res=$(python3 ../../tools/eval_icdar.py --gt_path ../../datasets/train_full_images_0.json --result_json results/ppocr_system_results_b4.json 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"
  f_score=$(echo "$res" | grep -oE "F-score: ([0-9.]+)" | cut -d ' ' -f 2)
  compare_res $f_score $4
  judge_ret $? "$3_$2_cpp_result: Precision compare!" log/$1_$2_$3_eval.log
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function eval_python()
{  
  echo -e "\n########################\nCase Start: eval python\n########################"
  pushd python
  if [ ! -d log ];then
    mkdir log
  fi
  python3 ppocr_system_$1.py --input=../datasets/train_full_images_0 --batch_size=4 --bmodel_det=../models/$TARGET/ch_PP-OCRv3_det_$2.bmodel \
                                                                --bmodel_cls=../models/$TARGET/ch_PP-OCRv3_cls_$2.bmodel \
                                                                --bmodel_rec=../models/$TARGET/ch_PP-OCRv3_rec_$2.bmodel \
                                                                --dev_id=$TPUID > log/$1_$2_debug.log 2>&1
  judge_ret $? "$1 $2 python debug" log/$1_$2_debug.log
  tail -n 20 log/$1_$2_debug.log
  
  echo "Evaluating..."
  res=$(python3 ../tools/eval_icdar.py --gt_path ../datasets/train_full_images_0.json --result_json results/ppocr_system_results_b4.json 2>&1 | tee log/$1_$2_eval.log)
  echo -e "$res"
  f_score=$(echo "$res" | grep -oE "F-score: ([0-9.]+)" | cut -d ' ' -f 2)
  compare_res $f_score $3
  judge_ret $? "$2_$1_python_result: Precision compare!" log/$1_$2_eval.log

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
  build_pcie bmcv
  pip3 install -r python/requirements.txt
  download
  if test $TARGET = "BM1684"
  then
    eval_python opencv fp32 0.57461
    eval_cpp pcie bmcv fp32 0.57303
  elif test $TARGET = "BM1684X"
  then
    eval_python opencv fp32 0.57422
    eval_python opencv fp16 0.57488
    eval_cpp pcie bmcv fp32 0.57194
    eval_cpp pcie bmcv fp16 0.57153
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  pip3 install -r python/requirements.txt
  download
  if test $TARGET = "BM1684"
  then
    eval_python opencv fp32 0.57461
    eval_cpp soc bmcv fp32 0.57303
  elif test $TARGET = "BM1684X"
  then
    eval_python opencv fp32 0.57422
    eval_python opencv fp16 0.57488
    eval_cpp soc bmcv fp32 0.57194
    eval_cpp soc bmcv fp16 0.57153
  fi
fi

if [[ $ALL_PASS -eq 0 ]]
then
    echo "===================================================================="
    echo "Some process produced unexpected results, please look out their logs!"
    echo "===================================================================="
else
    echo "===================="
    echo "Test cases all pass!"
    echo "===================="
fi

popd