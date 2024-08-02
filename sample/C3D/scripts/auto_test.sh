#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684X"
MODE="pcie_test"
SOCSDK="/home/lihengfang/work/sophonsdk/soc-sdk"
TPUID=0
ALL_PASS=1
PYTEST="auto_test"
ECHO_LINES=20
usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X|BM1688|CV186X] [ -s SOCSDK] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2 
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


if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi
if [ -f "scripts/acc.txt" ]; then
  rm scripts/acc.txt
fi
echo "|   测试平台    |      测试程序     |    测试模型        | ACC |" >> scripts/acc.txt
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
      bmrt_test_case BM1684/c3d_fp32_1b.bmodel
      bmrt_test_case BM1684/c3d_fp32_4b.bmodel
      bmrt_test_case BM1684/c3d_int8_1b.bmodel
      bmrt_test_case BM1684/c3d_int8_4b.bmodel
    elif test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/c3d_fp32_1b.bmodel
      bmrt_test_case BM1684X/c3d_fp32_4b.bmodel
      bmrt_test_case BM1684X/c3d_fp16_1b.bmodel
      bmrt_test_case BM1684X/c3d_fp16_4b.bmodel
      bmrt_test_case BM1684X/c3d_int8_1b.bmodel
      bmrt_test_case BM1684X/c3d_int8_4b.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/c3d_fp32_1b.bmodel
      bmrt_test_case BM1688/c3d_fp32_4b.bmodel
      bmrt_test_case BM1688/c3d_fp16_1b.bmodel
      bmrt_test_case BM1688/c3d_fp16_4b.bmodel
      bmrt_test_case BM1688/c3d_int8_1b.bmodel
      bmrt_test_case BM1688/c3d_int8_4b.bmodel
      bmrt_test_case BM1688/c3d_fp32_1b_2core.bmodel
      bmrt_test_case BM1688/c3d_fp32_4b_2core.bmodel
      bmrt_test_case BM1688/c3d_fp16_1b_2core.bmodel
      bmrt_test_case BM1688/c3d_fp16_4b_2core.bmodel
      bmrt_test_case BM1688/c3d_int8_1b_2core.bmodel
      bmrt_test_case BM1688/c3d_int8_4b_2core.bmodel
    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/c3d_fp32_1b.bmodel
      bmrt_test_case CV186X/c3d_fp32_4b.bmodel
      bmrt_test_case CV186X/c3d_fp16_1b.bmodel
      bmrt_test_case CV186X/c3d_fp16_4b.bmodel
      bmrt_test_case CV186X/c3d_int8_1b.bmodel
      bmrt_test_case CV186X/c3d_int8_4b.bmodel
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

function download(){
    #download dataset and models.
    if [ ! -d './datasets/UCF_test_01' -o ! -d './models' ]; then
        echo "preparing datasets and models......"
        chmod +x ./scripts/download.sh
        ./scripts/download.sh
        judge_ret $? "download" 0
    else
        echo "data already exists!"
    fi
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

function build_pcie(){
    pushd cpp/c3d_$1
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. && make
    judge_ret $? "build c3d_$1" 0
    popd
}

function build_soc()
{
    pushd cpp/c3d_$1
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
    judge_ret $? "build soc c3d_$1" 0
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
#e.g.: test_cpp opencv pcie c3d_int8_1b.bmodel 0.715
function test_cpp(){
    echo -e "\n########################\nCase Start: eval cpp\n########################"
    pushd cpp/c3d_$2
    if [ ! -d log ];then
        mkdir log
    fi
    echo "testing cpp $2 $3:"
    chmod +x ./c3d_$2.$1
    ./c3d_$2.$1 --input=../../datasets/UCF_test_01 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$2_$3.log 2>&1
    judge_ret $? "./c3d_$2.$1 --input=../../datasets/UCF_test_01 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$2_$3.log 2>&1" log/$2_$3.log
    tail -n 25 log/$2_$3.log

    echo "==================="
    echo "Comparing statis..."
    python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=c3d_$2.$1 --language=cpp --input=log/$2_$3.log --bmodel=$3
    judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=c3d_$2.$1 --language=cpp --input=log/$2_$3.log --bmodel=$3"
    echo "==================="
    
    echo "Evaluating..."
    res=$(python3 ../../tools/eval_ucf.py --result_json results/$3_$2_cpp.json --gt_path ../../datasets/ground_truth.json 2>&1 | tee log/$2_$1_$3_eval.log)
    echo -e "$res"
    array=(${res//=/ })
    acc=${array[1]}
    compare_res $acc $4
    judge_ret $? "$3_$1_cpp: Precision compare!" log/$2_$1_$3_eval.log

    printf "| %-12s | %-14s | %-22s | %8.3f |\n" "$PLATFORM" "c3d_$2.$1" "$3" "$(printf "%.3f" $acc)" >> ../../scripts/acc.txt
    popd
    echo -e "########################\nCase End: eval cpp\n########################\n"
}

#e.g.: test_python opencv c3d_int8_1b.bmodel 0.715
function test_python(){
    echo -e "\n########################\nCase Start: eval python\n########################"
    pushd python
    if [ ! -d log ];then
        mkdir log
    fi
    echo "testing python $1 $2:"
    python3 c3d_$1.py --input ../datasets/UCF_test_01 --bmodel ../models/$TARGET/$2 --dev_id $TPUID > log/$1_$2.log 2>&1
    judge_ret $? "python3 c3d_$1.py --input ../datasets/UCF_test_01 --bmodel ../models/$TARGET/$2 --dev_id $TPUID > log/$1_$2.log 2>&1" log/$1_$2.log
    tail -n 25 log/$1_$2.log 

    echo "==================="
    echo "Comparing statis..."
    python3 ../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=c3d_$1.py --language=python --input=log/$1_$2.log --bmodel=$2
    judge_ret $? "python3 ../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=c3d_$1.py --language=python --input=log/$1_$2.log --bmodel=$2"
    echo "==================="

    echo "Evaluating..."
    res=$(python3 ../tools/eval_ucf.py --result_json results/$2_$1_python.json --gt_path ../datasets/ground_truth.json 2>&1 | tee log/$1_$2_eval.log)
    echo -e "$res"
    array=(${res//=/ })
    acc=${array[1]}
    compare_res $acc $3
    judge_ret $? "$2_$1_python: Precision compare!" log/$1_$2_eval.log
    printf "| %-12s | %-14s | %-22s | %8.3f |\n" "$PLATFORM" "c3d_$1.py" "$2" "$(printf "%.3f" $acc)" >> ../scripts/acc.txt
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
  build_pcie opencv
  download
  pip3 install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684"
  then
    test_python opencv c3d_fp32_1b.bmodel   0.715
    test_python opencv c3d_fp32_4b.bmodel   0.715
    test_python opencv c3d_int8_1b.bmodel   0.712
    test_python opencv c3d_int8_4b.bmodel   0.712
    test_cpp pcie opencv c3d_fp32_1b.bmodel 0.715
    test_cpp pcie opencv c3d_fp32_4b.bmodel 0.715
    test_cpp pcie opencv c3d_int8_1b.bmodel 0.712
    test_cpp pcie opencv c3d_int8_4b.bmodel 0.712
    test_cpp pcie bmcv c3d_fp32_1b.bmodel   0.715
    test_cpp pcie bmcv c3d_fp32_4b.bmodel   0.715
    test_cpp pcie bmcv c3d_int8_1b.bmodel   0.710
    test_cpp pcie bmcv c3d_int8_4b.bmodel   0.710
  elif test $TARGET = "BM1684X"
  then
    test_python opencv c3d_fp32_1b.bmodel   0.715
    test_python opencv c3d_fp32_4b.bmodel   0.715
    test_python opencv c3d_fp16_1b.bmodel   0.715
    test_python opencv c3d_fp16_4b.bmodel   0.715
    test_python opencv c3d_int8_1b.bmodel   0.715
    test_python opencv c3d_int8_4b.bmodel   0.715
    test_cpp pcie opencv c3d_fp32_1b.bmodel 0.715
    test_cpp pcie opencv c3d_fp32_4b.bmodel 0.715
    test_cpp pcie opencv c3d_fp16_1b.bmodel 0.715
    test_cpp pcie opencv c3d_fp16_4b.bmodel 0.715
    test_cpp pcie opencv c3d_int8_1b.bmodel 0.715
    test_cpp pcie opencv c3d_int8_4b.bmodel 0.715
    test_cpp pcie bmcv c3d_fp32_1b.bmodel   0.715
    test_cpp pcie bmcv c3d_fp32_4b.bmodel   0.715
    test_cpp pcie bmcv c3d_fp16_1b.bmodel   0.715
    test_cpp pcie bmcv c3d_fp16_4b.bmodel   0.715
    test_cpp pcie bmcv c3d_int8_1b.bmodel   0.712
    test_cpp pcie bmcv c3d_int8_4b.bmodel   0.712
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
  build_soc opencv
elif test $MODE = "soc_test"
then
  download
  pip3 install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684"
  then
    test_python opencv c3d_fp32_1b.bmodel  0.715
    test_python opencv c3d_fp32_4b.bmodel  0.715
    test_python opencv c3d_int8_1b.bmodel  0.712
    test_python opencv c3d_int8_4b.bmodel  0.712
    test_cpp soc opencv c3d_fp32_1b.bmodel 0.715
    test_cpp soc opencv c3d_fp32_4b.bmodel 0.715
    test_cpp soc opencv c3d_int8_1b.bmodel 0.712
    test_cpp soc opencv c3d_int8_4b.bmodel 0.712
    test_cpp soc bmcv c3d_fp32_1b.bmodel   0.715
    test_cpp soc bmcv c3d_fp32_4b.bmodel   0.715
    test_cpp soc bmcv c3d_int8_1b.bmodel   0.710
    test_cpp soc bmcv c3d_int8_4b.bmodel   0.710
    
  elif test $TARGET = "BM1684X"
  then
    test_python opencv c3d_fp32_1b.bmodel  0.715
    test_python opencv c3d_fp32_4b.bmodel  0.715
    test_python opencv c3d_fp16_1b.bmodel  0.715
    test_python opencv c3d_fp16_4b.bmodel  0.715
    test_python opencv c3d_int8_1b.bmodel  0.715
    test_python opencv c3d_int8_4b.bmodel  0.715
    test_cpp soc opencv c3d_fp32_1b.bmodel 0.715
    test_cpp soc opencv c3d_fp32_4b.bmodel 0.715
    test_cpp soc opencv c3d_fp16_1b.bmodel 0.715
    test_cpp soc opencv c3d_fp16_4b.bmodel 0.715
    test_cpp soc opencv c3d_int8_1b.bmodel 0.715
    test_cpp soc opencv c3d_int8_4b.bmodel 0.715
    test_cpp soc bmcv c3d_fp32_1b.bmodel   0.715
    test_cpp soc bmcv c3d_fp32_4b.bmodel   0.715
    test_cpp soc bmcv c3d_fp16_1b.bmodel   0.715
    test_cpp soc bmcv c3d_fp16_4b.bmodel   0.715
    test_cpp soc bmcv c3d_int8_1b.bmodel   0.712
    test_cpp soc bmcv c3d_int8_4b.bmodel   0.712
  elif [ "$TARGET" = "BM1688" ] || [ "$TARGET" = "CV186X" ]
  then
    test_python opencv c3d_fp32_1b.bmodel  0.715
    test_python opencv c3d_fp32_4b.bmodel  0.715
    test_python opencv c3d_fp16_1b.bmodel  0.715
    test_python opencv c3d_fp16_4b.bmodel  0.715
    test_python opencv c3d_int8_1b.bmodel  0.711
    test_python opencv c3d_int8_4b.bmodel  0.711
    test_cpp soc opencv c3d_fp32_1b.bmodel 0.715
    test_cpp soc opencv c3d_fp32_4b.bmodel 0.715
    test_cpp soc opencv c3d_fp16_1b.bmodel 0.715
    test_cpp soc opencv c3d_fp16_4b.bmodel 0.715
    test_cpp soc opencv c3d_int8_1b.bmodel 0.711
    test_cpp soc opencv c3d_int8_4b.bmodel 0.711
    test_cpp soc bmcv c3d_fp32_1b.bmodel   0.715
    test_cpp soc bmcv c3d_fp32_4b.bmodel   0.715
    test_cpp soc bmcv c3d_fp16_1b.bmodel   0.715
    test_cpp soc bmcv c3d_fp16_4b.bmodel   0.715
    test_cpp soc bmcv c3d_int8_1b.bmodel   0.715
    test_cpp soc bmcv c3d_int8_4b.bmodel   0.715
    
    if test "$PLATFORM" = "SE9-16"; then 
      test_python opencv c3d_fp32_1b_2core.bmodel  0.715
      test_python opencv c3d_fp32_4b_2core.bmodel  0.715
      test_python opencv c3d_fp16_1b_2core.bmodel  0.715
      test_python opencv c3d_fp16_4b_2core.bmodel  0.715
      test_python opencv c3d_int8_1b_2core.bmodel  0.711
      test_python opencv c3d_int8_4b_2core.bmodel  0.711
      test_cpp soc opencv c3d_fp32_1b_2core.bmodel 0.715
      test_cpp soc opencv c3d_fp32_4b_2core.bmodel 0.715
      test_cpp soc opencv c3d_fp16_1b_2core.bmodel 0.715
      test_cpp soc opencv c3d_fp16_4b_2core.bmodel 0.715
      test_cpp soc opencv c3d_int8_1b_2core.bmodel 0.711
      test_cpp soc opencv c3d_int8_4b_2core.bmodel 0.711
      test_cpp soc bmcv c3d_fp32_1b_2core.bmodel   0.715
      test_cpp soc bmcv c3d_fp32_4b_2core.bmodel   0.715
      test_cpp soc bmcv c3d_fp16_1b_2core.bmodel   0.715
      test_cpp soc bmcv c3d_fp16_4b_2core.bmodel   0.715
      test_cpp soc bmcv c3d_int8_1b_2core.bmodel   0.715
      test_cpp soc bmcv c3d_int8_4b_2core.bmodel   0.715
    fi
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------C3D acc----------"
  cat scripts/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------C3D performance-----------"
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

popd
