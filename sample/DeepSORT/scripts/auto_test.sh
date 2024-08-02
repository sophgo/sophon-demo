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
CASE_MODE="fully"
usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_build|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X|BM1688|CV186X] [ -s SOCSDK] [ -d TPUID] [ -p PYTEST auto_test|pytest] [ -c CASE_MODE fully|partly]" 1>&2 
}

while getopts ":m:t:s:d:p:c:" opt
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
    c)
      CASE_MODE=${OPTARG}
      echo "case mode is $CASE_MODE";;
    ?)
      usage
      exit 1;;
  esac
done

if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi
if [ -f "scripts/acc.txt" ]; then
  rm scripts/acc.txt
fi

PLATFORM=$TARGET
if test $MODE = "soc_test"; then
  if test $TARGET = "BM1684X"; then
    PLATFORM="SE7-32"
  elif test $TARGET = "BM1684"; then
    PLATFORM="SE5-16"
  elif test $TARGET = "BM1688"; then
    PLATFORM="SE9-16"
    cpu_core_num=$(nproc)
    if [ "$cpu_core_num" -eq 6 ]; then
      PLATFORM="SE9-8"
    fi
  elif test $TARGET = "CV186X"; then
    PLATFORM="SE9-8"
  else
    echo "Unknown TARGET type: $TARGET"
  fi
fi


function bmrt_test_case(){
   calculate_time_log=$(bmrt_test --bmodel $1 --devid $TPUID | grep "calculate" 2>&1)
   is_4b=$(echo $1 |grep "4b")

   if [ "$is_4b" != "" ]; then
    readarray -t calculate_times < <(echo "$calculate_time_log" | grep -oP 'calculate  time\(s\): \K\d+\.\d+' | awk '{printf "%.2f \n", $1 * 250}')
   else
    readarray -t calculate_times < <(echo "$calculate_time_log" | grep -oP 'calculate  time\(s\): \K\d+\.\d+' | awk '{printf "%.2f \n", $1 * 1000}')
   fi
   for time in "${calculate_times[@]}"
   do
     printf "| %-35s| % 15s |\n" "$1" "$time"
   done
}
function bmrt_test_benchmark(){
    pushd models
    printf "| %-35s| % 15s |\n" "测试模型" "calculate time(ms)"
    printf "| %-35s| % 15s |\n" "-------------------" "--------------"
   
    if test $TARGET = "BM1684"; then
      bmrt_test_case BM1684/extractor_fp32_1b.bmodel
      bmrt_test_case BM1684/extractor_fp32_4b.bmodel
      bmrt_test_case BM1684/extractor_int8_1b.bmodel
      bmrt_test_case BM1684/extractor_int8_4b.bmodel
    elif test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/extractor_fp32_1b.bmodel
      bmrt_test_case BM1684X/extractor_fp32_4b.bmodel
      bmrt_test_case BM1684X/extractor_fp16_1b.bmodel
      bmrt_test_case BM1684X/extractor_fp16_4b.bmodel
      bmrt_test_case BM1684X/extractor_int8_1b.bmodel
      bmrt_test_case BM1684X/extractor_int8_4b.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/extractor_fp32_1b.bmodel
      bmrt_test_case BM1688/extractor_fp32_4b.bmodel
      bmrt_test_case BM1688/extractor_fp16_1b.bmodel
      bmrt_test_case BM1688/extractor_fp16_4b.bmodel
      bmrt_test_case BM1688/extractor_int8_1b.bmodel
      bmrt_test_case BM1688/extractor_int8_4b.bmodel
      if test "$PLATFORM" = "SE9-16"; then 
        bmrt_test_case BM1688/extractor_fp32_1b_2core.bmodel
        bmrt_test_case BM1688/extractor_fp32_4b_2core.bmodel
        bmrt_test_case BM1688/extractor_fp16_1b_2core.bmodel
        bmrt_test_case BM1688/extractor_fp16_4b_2core.bmodel
        bmrt_test_case BM1688/extractor_int8_1b_2core.bmodel
        bmrt_test_case BM1688/extractor_int8_4b_2core.bmodel
      fi
    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/extractor_fp32_1b.bmodel
      bmrt_test_case CV186X/extractor_fp32_4b.bmodel
      bmrt_test_case CV186X/extractor_fp16_1b.bmodel
      bmrt_test_case CV186X/extractor_fp16_4b.bmodel
      bmrt_test_case CV186X/extractor_int8_1b.bmodel
      bmrt_test_case CV186X/extractor_int8_4b.bmodel
    fi
  
    popd
}



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
    if [[ $3 != 0 ]] && [[ $3 != "" ]];then
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
  ./deepsort_$2.$1 --input=$4 --bmodel_detector=../../models/$TARGET/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_cpp_test.log 2>&1
  judge_ret $? "./deepsort_$2.$1 --input=$4  --bmodel_detector=../../models/$TARGET/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor=../../models/$TARGET/$3 --dev_id=$TPUID" log/$1_$2_$3_cpp_test.log
  tail -n 20 log/$1_$2_$3_cpp_test.log
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
  echo "==================="
  echo "Comparing statis..."
  python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=deepsort_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3
  judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=deepsort_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../../tools/eval_mot15.py --gt_file ../../datasets/mot15_trainset/ADL-Rundle-6/gt/gt.txt --ts_file results/mot_eval/ADL-Rundle-6_$3.txt 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  judge_ret $? "mot_eval/ADL-Rundle-6_$3: Precision compare!" log/$1_$2_$3_eval.log
  mota=$(echo "$res" | grep -oP 'MOTA = \K\d+\.\d+'| xargs printf "%.3f")
  printf "| %-12s | %-18s | %-40s | %8s |\n" "$PLATFORM" "deepsort_$2.$1" "$3" "$mota" >> ../../scripts/acc.txt

  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function test_python()
{
  pushd python
  if [ ! -d log ];then
    mkdir log
  fi
  python3 deepsort_$1.py --input $3 --bmodel_detector ../models/$TARGET/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor ../models/$TARGET/$2 --dev_id=$TPUID > log/$1_$2_python_test.log 2>&1
  judge_ret $? "deepsort_$1.py --input $3 --bmodel_detector ../models/$TARGET/yolov5s_v6.1_3output_int8_1b.bmodel --bmodel_extractor ../models/$TARGET/$2 --dev_id=$TPUID" log/$1_$2_python_test.log
  tail -n 30 log/$1_$2_python_test.log
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

  echo "==================="
  echo "Comparing statis..."
  python3 ../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=deepsort_$1.py --language=python --input=log/$1_$2_debug.log --bmodel=$2
  judge_ret $? "python3 ../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=deepsort_$1.py --language=python --input=log/$1_$2_debug.log --bmodel=$2"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../tools/eval_mot15.py --gt_file ../datasets/mot15_trainset/ADL-Rundle-6/gt/gt.txt --ts_file results/mot_eval/ADL-Rundle-6_$2.txt 2>&1 | tee log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
  judge_ret $? "mot_eval/ADL-Rundle-6_$2: Precision compare!" log/$1_$2_eval.log
  mota=$(echo "$res" | grep -oP 'MOTA = \K\d+\.\d+'| xargs printf "%.3f")
  printf "| %-12s | %-18s | %-40s | %8s |\n" "$PLATFORM" "deepsort_$1.py" "$2" "$mota" >> ../scripts/acc.txt
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
elif test $MODE = "pcie_build"
then
  build_pcie bmcv
elif test $MODE = "pcie_test"
then
  download
  pip3 install -r python/requirements.txt
  pip3 install motmetrics
  if test $TARGET = "BM1684"
  then
    if test $CASE_MODE = "fully"
    then
      test_python opencv extractor_fp32_1b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_fp32_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_int8_1b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_int8_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_cpp pcie bmcv extractor_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp pcie bmcv extractor_fp32_4b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp pcie bmcv extractor_int8_1b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp pcie bmcv extractor_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

      eval_python opencv extractor_fp32_1b.bmodel 0.45717708125374323
      eval_python opencv extractor_fp32_4b.bmodel 0.45717708125374323
      eval_python opencv extractor_int8_1b.bmodel 0.45897384707526456
      eval_python opencv extractor_int8_4b.bmodel 0.45897384707526456
      eval_cpp pcie bmcv extractor_fp32_1b.bmodel 0.4497903773208225
      eval_cpp pcie bmcv extractor_fp32_4b.bmodel 0.4497903773208225
      eval_cpp pcie bmcv extractor_int8_1b.bmodel 0.4523857057296866
      eval_cpp pcie bmcv extractor_int8_4b.bmodel 0.4523857057296866
    elif test $CASE_MODE = "partly"
    then
      test_python opencv extractor_int8_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_cpp pcie bmcv extractor_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

      eval_python opencv extractor_int8_4b.bmodel 0.45897384707526456
      eval_cpp pcie bmcv extractor_int8_4b.bmodel 0.4523857057296866
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi

  elif test $TARGET = "BM1684X"
  then
    if test $CASE_MODE = "fully"
    then
      test_python opencv extractor_fp32_1b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_fp32_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_int8_1b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_int8_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_cpp pcie bmcv extractor_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp pcie bmcv extractor_fp32_4b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp pcie bmcv extractor_int8_1b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp pcie bmcv extractor_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

      eval_python opencv extractor_fp32_1b.bmodel 0.43801157915751643
      eval_python opencv extractor_fp32_4b.bmodel 0.43801157915751643
      eval_python opencv extractor_fp16_1b.bmodel 0.43801157915751643
      eval_python opencv extractor_fp16_4b.bmodel 0.43801157915751643
      eval_python opencv extractor_int8_1b.bmodel 0.43162307845877423
      eval_python opencv extractor_int8_4b.bmodel 0.43162307845877423
      eval_cpp pcie bmcv extractor_fp32_1b.bmodel 0.44320223597524455
      eval_cpp pcie bmcv extractor_fp32_4b.bmodel 0.44320223597524455
      eval_cpp pcie bmcv extractor_fp16_1b.bmodel 0.44320223597524455
      eval_cpp pcie bmcv extractor_fp16_4b.bmodel 0.44320223597524455
      eval_cpp pcie bmcv extractor_int8_1b.bmodel 0.43761229786384503
      eval_cpp pcie bmcv extractor_int8_4b.bmodel 0.43761229786384503
    elif test $CASE_MODE = "partly"
    then
      test_python opencv extractor_int8_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_cpp pcie bmcv extractor_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

      eval_python opencv extractor_int8_4b.bmodel 0.43162307845877423
      eval_cpp pcie bmcv extractor_int8_4b.bmodel 0.43761229786384503
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
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
    if test $CASE_MODE = "fully"
    then
      test_python opencv extractor_fp32_1b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_fp32_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_int8_1b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_int8_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_fp32_4b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_int8_1b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

      eval_python opencv extractor_fp32_1b.bmodel 0.45717708125374323
      eval_python opencv extractor_fp32_4b.bmodel 0.45717708125374323
      eval_python opencv extractor_int8_1b.bmodel 0.45897384707526456
      eval_python opencv extractor_int8_4b.bmodel 0.45897384707526456
      eval_cpp soc bmcv extractor_fp32_1b.bmodel 0.4497903773208225
      eval_cpp soc bmcv extractor_fp32_4b.bmodel 0.4497903773208225
      eval_cpp soc bmcv extractor_int8_1b.bmodel 0.4523857057296866
      eval_cpp soc bmcv extractor_int8_4b.bmodel 0.4523857057296866
    elif test $CASE_MODE = "partly"
    then
      test_python opencv extractor_int8_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

      eval_python opencv extractor_int8_4b.bmodel 0.45897384707526456
      eval_cpp soc bmcv extractor_int8_4b.bmodel 0.4523857057296866
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  elif test $TARGET = "BM1684X"
  then
    if test $CASE_MODE = "fully"
    then
      test_python opencv extractor_fp32_1b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_fp32_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_int8_1b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_int8_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_fp32_4b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_int8_1b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

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
    elif test $CASE_MODE = "partly"
    then
      test_python opencv extractor_int8_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

      eval_python opencv extractor_int8_4b.bmodel 0.43601517268915946
      eval_cpp soc bmcv extractor_int8_4b.bmodel  0.43761229786384503
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  elif test $TARGET = "CV186X"
  then
    if test $CASE_MODE = "fully"
    then
      test_python opencv extractor_fp32_1b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_fp32_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_int8_1b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_int8_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_fp32_4b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_int8_1b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

      eval_python opencv extractor_fp32_1b.bmodel 0.441
      eval_python opencv extractor_fp32_4b.bmodel 0.441
      eval_python opencv extractor_fp16_1b.bmodel 0.441
      eval_python opencv extractor_fp16_4b.bmodel 0.441
      eval_python opencv extractor_int8_1b.bmodel 0.441
      eval_python opencv extractor_int8_4b.bmodel 0.441
      eval_cpp soc bmcv extractor_fp32_1b.bmodel  0.4298263126372529
      eval_cpp soc bmcv extractor_fp32_4b.bmodel  0.4298263126372529
      eval_cpp soc bmcv extractor_fp16_1b.bmodel  0.43002595328408866
      eval_cpp soc bmcv extractor_fp16_4b.bmodel  0.43002595328408866
      eval_cpp soc bmcv extractor_int8_1b.bmodel  0.4294270313435815
      eval_cpp soc bmcv extractor_int8_4b.bmodel  0.4294270313435815
    elif test $CASE_MODE = "partly"
    then
      test_python opencv extractor_int8_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

      eval_python opencv extractor_int8_4b.bmodel 0.441
      eval_cpp soc bmcv extractor_int8_4b.bmodel  0.4294270313435815
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  elif [ "$TARGET" = "BM1688" ] || [ "$TARGET" = "CV186X" ]
  then
    if test $CASE_MODE = "fully"
    then
      test_python opencv extractor_fp32_1b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_fp32_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_int8_1b.bmodel ../datasets/test_car_person_1080P.mp4
      test_python opencv extractor_int8_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_fp32_4b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_int8_1b.bmodel ../../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

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
      if test "$PLATFORM" = "SE9-16"; then 
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
    elif test $CASE_MODE = "partly"
    then
      test_python opencv extractor_int8_4b.bmodel ../datasets/test_car_person_1080P.mp4
      test_cpp soc bmcv extractor_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

      eval_python opencv extractor_int8_4b.bmodel 0.440
      eval_cpp soc bmcv extractor_int8_4b.bmodel  0.429
      if test "$PLATFORM" = "SE9-16"; then 
        eval_python opencv extractor_int8_4b_2core.bmodel 0.440
        eval_cpp soc bmcv extractor_int8_4b_2core.bmodel  0.429
      fi
    else
      echo "unknown CASE_MODE: $CASE_MODE"
    fi
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]
then
  echo "--------DeepSORT MOTA----------"
  cat scripts/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------DeepSORT performance-----------"
  cat tools/benchmark.txt
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
