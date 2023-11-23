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

function compile_mlir()
{
  ./scripts/yolov3_fp32bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  ./scripts/yolov3_fp16bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp16bmodel" 0
  ./scripts/yolov3_int8bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET int8bmodel" 0
  ./scripts/yolov4_fp32bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  ./scripts/yolov4_fp16bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp16bmodel" 0
  ./scripts/yolov4_int8bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET int8bmodel" 0
}

function build_pcie()
{
  pushd cpp/yolov34_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build yolov34_$1" 0
  popd
}

function build_soc()
{
  pushd cpp/yolov34_$1
  if [ -d build ]; then
      rm -rf build
  fi
  if test $1 = "sail"; then
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK -DSAIL_PATH=$SAIL_PATH && make
    judge_ret $? "build soc yolov34_$1" 0
  else
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
    judge_ret $? "build soc yolov34_$1" 0
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

function test_cpp()
{
  pushd cpp/yolov34_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./yolov34_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_cpp_test.log
  judge_ret $? "./yolov34_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID" log/$1_$2_$3_cpp_test.log
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/yolov34_$2
  if [ ! -d log ];then
    mkdir log
  fi
  yolov4_flag=$(echo $3 | grep "yolov4")
  if ["$yolov4_flag" != ""];then
    conf_thresh = 0.3
    nms_thresh = 0.5
  else
    conf_thresh = 0.001
    nms_thresh = 0.6
  fi
  ./yolov34_$2.$1 --input=../../datasets/coco/val2017_1000 --bmodel=../../models/$TARGET/$3 --conf_thresh=$conf_thresh --nms_thresh=$nms_thresh --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./yolov34_$2.$1 --input=../../datasets/coco/val2017_1000 --bmodel=../../models/$TARGET/$3 --conf_thresh=$conf_thresh --nms_thresh=$nms_thresh --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1" log/$1_$2_$3_debug.log
  tail -n 15 log/$1_$2_$3_debug.log
  
  echo "Evaluating..."
  res=$(python3 ../../tools/eval_coco.py --gt_path ../../datasets/coco/instances_val2017_1000.json --result_json results/$3_val2017_1000_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"

  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  judge_ret $? "$3_val2017_1000_$2_cpp_result: Precision compare!" log/$1_$2_$3_eval.log
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function test_python()
{
  if [ ! -d log ];then
    mkdir log
  fi
  python3 python/yolov34_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_python_test.log
  judge_ret $? "python3 python/yolov34_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID" log/$1_$2_python_test.log
}

function eval_python()
{  
  echo -e "\n########################\nCase Start: eval python\n########################"
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  yolov4_flag=$(echo $3 | grep "yolov4")
  if ["$yolov4_flag" != ""];then
    conf_thresh = 0.3
    nms_thresh = 0.5
  else
    conf_thresh = 0.001
    nms_thresh = 0.6
  fi
  python3 python/yolov34_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh $conf_thresh --nms_thresh $nms_thresh > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/yolov34_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh $conf_thresh --nms_thresh $nms_thresh > python/log/$1_$2_debug.log 2>&1" python/log/$1_$2_debug.log
  tail -n 20 python/log/$1_$2_debug.log
  
  echo "Evaluating..."
  res=$(python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/$2_val2017_1000_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
  judge_ret $? "$2_val2017_1000_$1_python_result: Precision compare!" python/log/$1_$2_eval.log
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
  build_pcie sail
  download
  if test $TARGET = "BM1684"
  then
    test_python opencv yolov3_fp32_1b.bmodel datasets/test
    test_python opencv yolov3_int8_4b.bmodel datasets/test
    test_python bmcv yolov3_fp32_1b.bmodel datasets/test
    test_python bmcv yolov3_int8_4b.bmodel datasets/test
    test_cpp pcie bmcv yolov3_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv yolov3_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov3_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov3_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov3_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov3_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov3_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov3_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail yolov3_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail yolov3_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv yolov3_fp32_1b.bmodel 0.47067943487124275
    eval_python opencv yolov3_int8_1b.bmodel 0.38308073092209927
    eval_python opencv yolov3_int8_4b.bmodel 0.38694755826639443
    eval_python bmcv yolov3_fp32_1b.bmodel 0.4577038294654958
    eval_python bmcv yolov3_int8_1b.bmodel 0.37139915199303436
    eval_python bmcv yolov3_int8_4b.bmodel 0.37742256558011966
    eval_cpp pcie bmcv yolov3_fp32_1b.bmodel 0.44618513920646896 
    eval_cpp pcie bmcv yolov3_int8_1b.bmodel 0.3698011287077706
    eval_cpp pcie bmcv yolov3_int8_4b.bmodel 0.3718812568970122
    eval_cpp pcie sail yolov3_fp32_1b.bmodel 0.44618513920646896 
    eval_cpp pcie sail yolov3_int8_1b.bmodel 0.3698011287077706
    eval_cpp pcie sail yolov3_int8_4b.bmodel 0.3718812568970122

    test_python opencv yolov4_fp32_1b.bmodel datasets/test
    test_python opencv yolov4_int8_4b.bmodel datasets/test
    test_python bmcv yolov4_fp32_1b.bmodel datasets/test
    test_python bmcv yolov4_int8_4b.bmodel datasets/test
    test_cpp pcie bmcv yolov4_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv yolov4_int8_4b.bmodel ../../datasets/test
    test_cpp pcie sail yolov4_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie sail yolov4_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov4_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov4_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov4_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov4_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov4_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov4_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail yolov4_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail yolov4_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv yolov4_fp32_1b.bmodel 0.25203026855911836
    eval_python opencv yolov4_int8_1b.bmodel 0.20823138814457098
    eval_python bmcv yolov4_fp32_1b.bmodel 0.24192182474497695
    eval_python bmcv yolov4_int8_1b.bmodel 0.1996143666478323
    eval_cpp pcie bmcv yolov4_fp32_1b.bmodel 0.24784360996361854 
    eval_cpp pcie bmcv yolov4_int8_1b.bmodel 0.2037278836882346
    eval_cpp pcie sail yolov4_fp32_1b.bmodel 0.24642136125683783 
    eval_cpp pcie sail yolov4_int8_1b.bmodel 0.2037278836882346
  elif test $TARGET = "BM1684X"
  then
    test_python opencv yolov3_fp32_1b.bmodel datasets/test
    test_python opencv yolov3_int8_4b.bmodel datasets/test
    test_python bmcv yolov3_fp32_1b.bmodel datasets/test
    test_python bmcv yolov3_int8_4b.bmodel datasets/test
    test_cpp pcie bmcv yolov3_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv yolov3_int8_4b.bmodel ../../datasets/test
    test_cpp pcie sail yolov3_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie sail yolov3_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov3_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov3_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov3_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov3_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov3_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov3_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail yolov3_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail yolov3_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv yolov3_fp32_1b.bmodel 0.47027688124870315
    eval_python opencv yolov3_fp16_1b.bmodel 0.47040430835003094
    eval_python opencv yolov3_int8_1b.bmodel 0.44636334239794856
    eval_python opencv yolov3_int8_4b.bmodel 0.44516431603019774
    eval_python bmcv yolov3_fp32_1b.bmodel 0.45776991019464947
    eval_python bmcv yolov3_fp16_1b.bmodel 0.45789262373865924 
    eval_python bmcv yolov3_int8_1b.bmodel 0.4357100056911882
    eval_python bmcv yolov3_int8_4b.bmodel 0.43482066132334396 
    eval_cpp pcie bmcv yolov3_fp32_1b.bmodel 0.44618446753553126 
    eval_cpp pcie bmcv yolov3_fp16_1b.bmodel 0.4461005235647297
    eval_cpp pcie bmcv yolov3_int8_1b.bmodel 0.42650106280397765 
    eval_cpp pcie bmcv yolov3_int8_4b.bmodel 0.42702455761891606 
    eval_cpp pcie sail yolov3_fp32_1b.bmodel 0.44618446753553126 
    eval_cpp pcie sail yolov3_fp16_1b.bmodel 0.4461005235647297
    eval_cpp pcie sail yolov3_int8_1b.bmodel 0.42650106280397765 
    eval_cpp pcie sail yolov3_int8_4b.bmodel 0.42702455761891606 
    
    test_python opencv yolov4_fp32_1b.bmodel datasets/test
    test_python opencv yolov4_int8_4b.bmodel datasets/test
    test_python bmcv yolov4_fp32_1b.bmodel datasets/test
    test_python bmcv yolov4_int8_4b.bmodel datasets/test
    test_cpp pcie bmcv yolov4_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv yolov4_int8_4b.bmodel ../../datasets/test
    test_cpp pcie sail yolov4_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie sail yolov4_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov4_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov4_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov4_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov4_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov4_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolov4_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail yolov4_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail yolov4_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    
    eval_python opencv yolov4_fp32_1b.bmodel 0.25208730541638674
    eval_python opencv yolov4_fp16_1b.bmodel 0.2518793738965063
    eval_python opencv yolov4_int8_1b.bmodel 0.23691730869480723
    eval_python opencv yolov4_int8_4b.bmodel 0.23604207342233868
    eval_python bmcv yolov4_fp32_1b.bmodel 0.24404540934567398
    eval_python bmcv yolov4_fp16_1b.bmodel 0.23986616984939513 
    eval_python bmcv yolov4_int8_1b.bmodel 0.22469758875830578
    eval_python bmcv yolov4_int8_4b.bmodel 0.22577518766126672 
    eval_cpp pcie bmcv yolov4_fp32_1b.bmodel 0.24749183135236155 
    eval_cpp pcie bmcv yolov4_fp16_1b.bmodel 0.24715251586618836
    eval_cpp pcie bmcv yolov4_int8_1b.bmodel 0.23207870486180743 
    eval_cpp pcie bmcv yolov4_int8_4b.bmodel 0.23092078998731425 
    eval_cpp pcie sail yolov4_fp32_1b.bmodel 0.24749183135236155 
    eval_cpp pcie sail yolov4_fp16_1b.bmodel 0.24715251586618836
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
  build_soc sail
elif test $MODE = "soc_test"
then
  download
  if test $TARGET = "BM1684"
  then
    test_python opencv yolov3_fp32_1b.bmodel datasets/test
    test_python opencv yolov3_int8_4b.bmodel datasets/test
    test_python bmcv yolov3_fp32_1b.bmodel datasets/test
    test_python bmcv yolov3_int8_4b.bmodel datasets/test
    test_cpp soc bmcv yolov3_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv yolov3_int8_4b.bmodel ../../datasets/test
    test_cpp soc sail yolov3_fp32_1b.bmodel ../../datasets/test
    test_cpp soc sail yolov3_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov3_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov3_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov3_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov3_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov3_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov3_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolov3_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolov3_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv yolov3_fp32_1b.bmodel 0.47067943487124275
    eval_python opencv yolov3_int8_1b.bmodel 0.38308073092209927
    eval_python opencv yolov3_int8_4b.bmodel 0.38694755826639443
    eval_python bmcv yolov3_fp32_1b.bmodel 0.4577038294654958
    eval_python bmcv yolov3_int8_1b.bmodel 0.37139915199303436
    eval_python bmcv yolov3_int8_4b.bmodel 0.37742256558011966
    eval_cpp soc bmcv yolov3_fp32_1b.bmodel 0.44618513920646896 
    eval_cpp soc bmcv yolov3_int8_1b.bmodel 0.3698011287077706
    eval_cpp soc bmcv yolov3_int8_4b.bmodel 0.3718812568970122
    eval_cpp soc sail yolov3_fp32_1b.bmodel 0.44618513920646896 
    eval_cpp soc sail yolov3_int8_1b.bmodel 0.3698011287077706
    eval_cpp soc sail yolov3_int8_4b.bmodel 0.3718812568970122

    test_python opencv yolov4_fp32_1b.bmodel datasets/test
    test_python opencv yolov4_int8_4b.bmodel datasets/test
    test_python bmcv yolov4_fp32_1b.bmodel datasets/test
    test_python bmcv yolov4_int8_4b.bmodel datasets/test
    test_cpp soc bmcv yolov4_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv yolov4_int8_4b.bmodel ../../datasets/test
    test_cpp soc sail yolov4_fp32_1b.bmodel ../../datasets/test
    test_cpp soc sail yolov4_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov4_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov4_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov4_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov4_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov4_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov4_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolov4_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolov4_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv yolov4_fp32_1b.bmodel 0.25203026855911836
    eval_python opencv yolov4_int8_1b.bmodel 0.20823138814457098
    eval_python bmcv yolov4_fp32_1b.bmodel 0.24192182474497695
    eval_python bmcv yolov4_int8_1b.bmodel 0.1996143666478323
    eval_cpp soc bmcv yolov4_fp32_1b.bmodel 0.24784360996361854 
    eval_cpp soc bmcv yolov4_int8_1b.bmodel 0.2037278836882346
    eval_cpp soc sail yolov4_fp32_1b.bmodel 0.24642136125683783 
    eval_cpp soc sail yolov4_int8_1b.bmodel 0.2037278836882346
  elif test $TARGET = "BM1684X"
  then
    test_python opencv yolov3_fp32_1b.bmodel datasets/test
    test_python opencv yolov3_int8_4b.bmodel datasets/test
    test_python bmcv yolov3_fp32_1b.bmodel datasets/test
    test_python bmcv yolov3_int8_4b.bmodel datasets/test
    test_cpp soc bmcv yolov3_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv yolov3_int8_4b.bmodel ../../datasets/test
    test_cpp soc sail yolov3_fp32_1b.bmodel ../../datasets/test
    test_cpp soc sail yolov3_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov3_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov3_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov3_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov3_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov3_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov3_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolov3_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolov3_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    
    eval_python opencv yolov3_fp32_1b.bmodel 0.47027688124870315
    eval_python opencv yolov3_fp16_1b.bmodel 0.47040430835003094
    eval_python opencv yolov3_int8_1b.bmodel 0.44636334239794856
    eval_python opencv yolov3_int8_4b.bmodel 0.44516431603019774
    eval_python bmcv yolov3_fp32_1b.bmodel 0.45776991019464947
    eval_python bmcv yolov3_fp16_1b.bmodel 0.45789262373865924 
    eval_python bmcv yolov3_int8_1b.bmodel 0.4357100056911882
    eval_python bmcv yolov3_int8_4b.bmodel 0.43482066132334396 
    eval_cpp soc bmcv yolov3_fp32_1b.bmodel 0.44618446753553126 
    eval_cpp soc bmcv yolov3_fp16_1b.bmodel 0.4461005235647297
    eval_cpp soc bmcv yolov3_int8_1b.bmodel 0.42650106280397765 
    eval_cpp soc bmcv yolov3_int8_4b.bmodel 0.42702455761891606 
    eval_cpp soc sail yolov3_fp32_1b.bmodel 0.44618446753553126 
    eval_cpp soc sail yolov3_fp16_1b.bmodel 0.4461005235647297
    eval_cpp soc sail yolov3_int8_1b.bmodel 0.42650106280397765 
    eval_cpp soc sail yolov3_int8_4b.bmodel 0.42702455761891606 

    test_python opencv yolov4_fp32_1b.bmodel datasets/test
    test_python opencv yolov4_int8_4b.bmodel datasets/test
    test_python bmcv yolov4_fp32_1b.bmodel datasets/test
    test_python bmcv yolov4_int8_4b.bmodel datasets/test
    test_cpp soc bmcv yolov4_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv yolov4_int8_4b.bmodel ../../datasets/test
    test_cpp soc sail yolov4_fp32_1b.bmodel ../../datasets/test
    test_cpp soc sail yolov4_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov4_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov4_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov4_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov4_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov4_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov4_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolov4_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolov4_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    
    eval_python opencv yolov4_fp32_1b.bmodel 0.25208730541638674
    eval_python opencv yolov4_fp16_1b.bmodel 0.2518793738965063
    eval_python opencv yolov4_int8_1b.bmodel 0.23691730869480723
    eval_python opencv yolov4_int8_4b.bmodel 0.23604207342233868
    eval_python bmcv yolov4_fp32_1b.bmodel 0.24404540934567398
    eval_python bmcv yolov4_fp16_1b.bmodel 0.23986616984939513 
    eval_python bmcv yolov4_int8_1b.bmodel 0.22469758875830578
    eval_python bmcv yolov4_int8_4b.bmodel 0.22577518766126672 
    eval_cpp soc bmcv yolov4_fp32_1b.bmodel 0.24749183135236155 
    eval_cpp soc bmcv yolov4_fp16_1b.bmodel 0.24715251586618836
    eval_cpp soc bmcv yolov4_int8_1b.bmodel 0.23207870486180743 
    eval_cpp soc bmcv yolov4_int8_4b.bmodel 0.23092078998731425 
    eval_cpp soc sail yolov4_fp32_1b.bmodel 0.24749183135236155 
    eval_cpp soc sail yolov4_fp16_1b.bmodel 0.24715251586618836
    eval_cpp soc sail yolov4_int8_1b.bmodel 0.23207870486180743 
    eval_cpp soc sail yolov4_int8_4b.bmodel 0.23092078998731425 
  elif test $TARGET = "BM1688"
  then
    test_python opencv yolov3_fp32_1b.bmodel datasets/test
    test_python opencv yolov3_int8_4b.bmodel datasets/test
    test_python bmcv yolov3_fp32_1b.bmodel datasets/test
    test_python bmcv yolov3_int8_4b.bmodel datasets/test
    test_cpp soc bmcv yolov3_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv yolov3_int8_4b.bmodel ../../datasets/test
    test_cpp soc sail yolov3_fp32_1b.bmodel ../../datasets/test
    test_cpp soc sail yolov3_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov3_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov3_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov3_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov3_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov3_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov3_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolov3_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolov3_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    
    eval_python opencv yolov3_fp32_1b.bmodel 0.47067938105420654
    eval_python opencv yolov3_fp16_1b.bmodel 0.47058393631571005
    eval_python opencv yolov3_int8_1b.bmodel 0.4454269452645311
    eval_python opencv yolov3_int8_4b.bmodel 0.4454269452645311
    eval_python bmcv yolov3_fp32_1b.bmodel 0.45810570099893505
    eval_python bmcv yolov3_fp16_1b.bmodel 0.4582569041268784
    eval_python bmcv yolov3_int8_1b.bmodel 0.43533974214986876
    eval_python bmcv yolov3_int8_4b.bmodel 0.43533974214986876
    eval_cpp soc bmcv yolov3_fp32_1b.bmodel 0.4461306562762746
    eval_cpp soc bmcv yolov3_fp16_1b.bmodel 0.4458266476021808
    eval_cpp soc bmcv yolov3_int8_1b.bmodel 0.4269113112724515
    eval_cpp soc bmcv yolov3_int8_4b.bmodel 0.4269113112724515
    eval_cpp soc sail yolov3_fp32_1b.bmodel 0.4461306562762746
    eval_cpp soc sail yolov3_fp16_1b.bmodel 0.4458266476021808
    eval_cpp soc sail yolov3_int8_1b.bmodel 0.4269113112724515
    eval_cpp soc sail yolov3_int8_4b.bmodel 0.4269113112724515

    test_python opencv yolov4_fp32_1b.bmodel datasets/test
    test_python opencv yolov4_int8_4b.bmodel datasets/test
    test_python bmcv yolov4_fp32_1b.bmodel datasets/test
    test_python bmcv yolov4_int8_4b.bmodel datasets/test
    test_cpp soc bmcv yolov4_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv yolov4_int8_4b.bmodel ../../datasets/test
    test_cpp soc sail yolov4_fp32_1b.bmodel ../../datasets/test
    test_cpp soc sail yolov4_int8_4b.bmodel ../../datasets/test
    test_python opencv yolov4_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolov4_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov4_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolov4_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov4_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolov4_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolov4_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolov4_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv yolov4_fp32_1b.bmodel 0.25203026855911836
    eval_python opencv yolov4_fp16_1b.bmodel 0.2519212256627636
    eval_python opencv yolov4_int8_1b.bmodel 0.23368842036037576
    eval_python opencv yolov4_int8_4b.bmodel 0.23368842036037576
    eval_python bmcv yolov4_fp32_1b.bmodel 0.24382584373876187
    eval_python bmcv yolov4_fp16_1b.bmodel 0.24424650826218103
    eval_python bmcv yolov4_int8_1b.bmodel 0.2264664789368982
    eval_python bmcv yolov4_int8_4b.bmodel 0.2264664789368982
    eval_cpp soc bmcv yolov4_fp32_1b.bmodel 0.24678471040047345
    eval_cpp soc bmcv yolov4_fp16_1b.bmodel 0.24737630315278233
    eval_cpp soc bmcv yolov4_int8_1b.bmodel 0.22929578290926206
    eval_cpp soc bmcv yolov4_int8_4b.bmodel 0.22929578290926206
    eval_cpp soc sail yolov4_fp32_1b.bmodel 0.24643350255307597
    eval_cpp soc sail yolov4_fp16_1b.bmodel 0.24737630315278233
    eval_cpp soc sail yolov4_int8_1b.bmodel 0.22929578290926206
    eval_cpp soc sail yolov4_int8_4b.bmodel 0.22929578290926206
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