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
CASE_MODE="fully"
usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_build|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X|BM1688|CV186X] [ -s SOCSDK] [-a SAIL] [ -d TPUID] [ -p PYTEST auto_test|pytest] [ -c fully|partly]" 1>&2 
}

while getopts ":m:t:s:a:d:p:c:" opt
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
echo "|   测试平台    |      测试程序           |   模型精度   | F-score |" >> scripts/acc.txt
PLATFORM=$TARGET

if test $MODE = "soc_test"; then
  if test $TARGET = "BM1684X"; then
    PLATFORM="SE7-32"
  elif test $TARGET = "BM1684"; then
    PLATFORM="SE5-16"
  elif test $TARGET = "BM1688"; then
    PLATFORM="SE9-16"
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

   stage=0
   for time in "${calculate_times[@]}"
   do
     if test $stage = 0; then
        printf "| %-41s| % 7s | % 15s |\n" "$1" "$stage" "$time"
     else
        printf "| %-41s| % 7s | % 15s |\n" "^" "$stage" "$time"
     fi
     stage=$(expr $stage + 1)
   done
}
function bmrt_test_benchmark(){
    pushd models
    printf "| %-35s| % 7s | % 15s |\n" "测试模型" "stage" "calculate time(ms)"
   
    if test $TARGET = "BM1684"; then
      bmrt_test_case BM1684/ch_PP-OCRv4_det_fp32.bmodel
      bmrt_test_case BM1684/ch_PP-OCRv4_rec_fp32.bmodel
    elif test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/ch_PP-OCRv4_det_fp32.bmodel
      bmrt_test_case BM1684X/ch_PP-OCRv4_rec_fp32.bmodel
      bmrt_test_case BM1684X/ch_PP-OCRv4_det_fp16.bmodel
      bmrt_test_case BM1684X/ch_PP-OCRv4_rec_fp16.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/ch_PP-OCRv4_det_fp32.bmodel
      bmrt_test_case BM1688/ch_PP-OCRv4_rec_fp32.bmodel
      bmrt_test_case BM1688/ch_PP-OCRv4_det_fp16.bmodel
      bmrt_test_case BM1688/ch_PP-OCRv4_rec_fp16.bmodel
      bmrt_test_case BM1688/ch_PP-OCRv4_det_fp32_2core.bmodel
      bmrt_test_case BM1688/ch_PP-OCRv4_rec_fp32_2core.bmodel
      bmrt_test_case BM1688/ch_PP-OCRv4_det_fp16_2core.bmodel
      bmrt_test_case BM1688/ch_PP-OCRv4_rec_fp16_2core.bmodel
    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/ch_PP-OCRv4_det_fp32.bmodel
      bmrt_test_case CV186X/ch_PP-OCRv4_rec_fp32.bmodel
      bmrt_test_case CV186X/ch_PP-OCRv4_det_fp16.bmodel
      bmrt_test_case CV186X/ch_PP-OCRv4_rec_fp16.bmodel
    fi
    popd
}


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
  # ./scripts/gen_int8bmodel_mlir.sh $TARGET
  # judge_ret $? "generate $TARGET int8bmodel" 0
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
    ret=`awk -v x=$1 -v y=$2 'BEGIN{print(x-y<0.01 && y-x<0.01)?1:0}'`
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
  ./ppocr_$2.$1 --input=../../datasets/train_full_images_0 --batch_size=4 --bmodel_det=../../models/$TARGET/ch_PP-OCRv4_det_$3.bmodel \
                                                                --bmodel_rec=../../models/$TARGET/ch_PP-OCRv4_rec_$3.bmodel \
                                                                --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "$1 $2 $3 cpp debug" log/$1_$2_$3_debug.log
  tail -n 15 log/$1_$2_$3_debug.log

  echo "==================="
  echo "Comparing statis..."
  python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=ppocr_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel_det=ch_PP-OCRv4_det_$3.bmodel --bmodel_rec=ch_PP-OCRv4_rec_$3.bmodel
  judge_ret $? "../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=ppocr_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel_det=ch_PP-OCRv4_det_$3.bmodel --bmodel_rec=ch_PP-OCRv4_rec_$3.bmodel"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../../tools/eval_icdar.py --gt_path ../../datasets/train_full_images_0.json --result_json results/ppocr_system_results_b4.json 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"
  f_score=$(echo "$res" | grep -oE "F-score: ([0-9.]+)" | cut -d ' ' -f 2)
  compare_res $f_score $4
  judge_ret $? "$3_$2_cpp_result: Precision compare!" log/$1_$2_$3_eval.log
  popd

  printf "| %-12s | %-25s | %-10s | %8.3f |\n" "$PLATFORM" "ppocr_$2.$1" "$3" "$(printf "%.3f" $f_score)" >> scripts/acc.txt

  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function eval_python()
{  
  echo -e "\n########################\nCase Start: eval python\n########################"
  pushd python
  if [ ! -d log ];then
    mkdir log
  fi
  python3 ppocr_system_$1.py --input=../datasets/train_full_images_0 --batch_size=4 --bmodel_det=../models/$TARGET/ch_PP-OCRv4_det_$2.bmodel \
                                                                --bmodel_rec=../models/$TARGET/ch_PP-OCRv4_rec_$2.bmodel \
                                                                --dev_id=$TPUID > log/$1_$2_debug.log 2>&1
  judge_ret $? "$1 $2 python debug" log/$1_$2_debug.log
  tail -n 30 log/$1_$2_debug.log

  echo "==================="
  echo "Comparing statis..."
  python3 ../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=ppocr_system_$1.py --language=python --input=log/$1_$2_debug.log --bmodel_det=ch_PP-OCRv4_det_$2.bmodel --bmodel_rec=ch_PP-OCRv4_rec_$2.bmodel
  judge_ret $? "python3 ../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=ppocr_system_$1.py --language=python --input=log/$1_$2_debug.log --bmodel_det=ch_PP-OCRv4_det_$2.bmodel --bmodel_rec=ch_PP-OCRv4_rec_$2.bmodel"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../tools/eval_icdar.py --gt_path ../datasets/train_full_images_0.json --result_json results/ppocr_system_results_b4.json 2>&1 | tee log/$1_$2_eval.log)
  echo -e "$res"
  f_score=$(echo "$res" | grep -oE "F-score: ([0-9.]+)" | cut -d ' ' -f 2)
  compare_res $f_score $3
  judge_ret $? "$2_$1_python_result: Precision compare!" log/$1_$2_eval.log

  popd

  printf "| %-12s | %-25s | %-10s | %8.3f |\n" "$PLATFORM" "ppocr_system_$1.py" "$2" "$(printf "%.3f" $f_score)" >> scripts/acc.txt

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
  pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  download
  if test $TARGET = "BM1684"
  then
    eval_python opencv fp32 0.608
    eval_cpp pcie bmcv fp32 0.608
  elif test $TARGET = "BM1684X"
  then
    eval_python opencv fp32 0.608
    eval_python opencv fp16 0.608
    eval_cpp pcie bmcv fp32 0.604
    eval_cpp pcie bmcv fp16 0.604
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  download
  if test $TARGET = "BM1684"
  then
    eval_python opencv fp32 0.608
    eval_cpp soc bmcv fp32 0.608
  elif test $TARGET = "BM1684X"
  then
    eval_python opencv fp32 0.608
    eval_python opencv fp16 0.608
    eval_cpp soc bmcv fp32 0.604
    eval_cpp soc bmcv fp16 0.604
  elif [ "$TARGET" = "BM1688" ] || [ "$TARGET" = "CV186X" ]
  then
    eval_python opencv fp32 0.608
    eval_python opencv fp16 0.608
    eval_cpp soc bmcv fp32  0.604
    eval_cpp soc bmcv fp16  0.604

    if test "$PLATFORM" = "SE9-16"; then 
      eval_python opencv fp32_2core 0.608
      eval_python opencv fp16_2core 0.608
      eval_cpp soc bmcv fp32_2core  0.604
      eval_cpp soc bmcv fp16_2core  0.604
    fi
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------ppocr mAP----------"
  cat scripts/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------ppocr performance-----------"
  cat tools/benchmark.txt
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