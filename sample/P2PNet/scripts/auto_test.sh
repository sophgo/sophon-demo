#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1688"
MODE="soc_test"
TPUID=0
ALL_PASS=1
PYTEST="auto_test"
ECHO_LINES=20
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib

usage()
{
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X|BM1688|CV186X] [ -s SOCSDK] [-a SAIL] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2
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

if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi

if [ -f "scripts/acc.txt" ]; then
  rm scripts/acc.txt
fi
echo "|   测试平台    |      测试程序       |        测试模型        | ACC(%) |" >> scripts/acc.txt
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
      bmrt_test_case BM1684/p2pnet_bm1684_fp32_1b.bmodel
      bmrt_test_case BM1684/p2pnet_bm1684_int8_1b.bmodel
      bmrt_test_case BM1684/p2pnet_bm1684_int8_4b.bmodel
    elif test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/p2pnet_bm1684x_fp32_1b.bmodel
      bmrt_test_case BM1684X/p2pnet_bm1684x_fp16_1b.bmodel
      bmrt_test_case BM1684X/p2pnet_bm1684x_int8_1b.bmodel
      bmrt_test_case BM1684X/p2pnet_bm1684x_int8_4b.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/p2pnet_bm1688_fp32_1b.bmodel
      bmrt_test_case BM1688/p2pnet_bm1688_fp16_1b.bmodel
      bmrt_test_case BM1688/p2pnet_bm1688_int8_1b.bmodel
      bmrt_test_case BM1688/p2pnet_bm1688_int8_4b.bmodel
    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/p2pnet_cv186x_fp32_1b.bmodel
      bmrt_test_case CV186X/p2pnet_cv186x_fp16_1b.bmodel
      bmrt_test_case CV186X/p2pnet_cv186x_int8_1b.bmodel
      bmrt_test_case CV186X/p2pnet_cv186x_int8_4b.bmodel
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
  
  echo "==================="
  echo "Comparing statis..."
  python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=p2pnet_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3
  judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=p2pnet_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../../tools/eval_acc.py --gt_path ../../datasets/test/ground-truth --result_path results/images 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  array[1]=${array[1]%,}
  acc=${array[1]}
  compare_res $acc $4
  judge_ret $? "cpp result: Precision compare!" log/$1_$2_$3_eval.log
  printf "| %-12s | %-18s | %-25s | %8.2f |\n" "$PLATFORM" "p2pnet_$2.$1" "$3" "$(printf "%.2f" $acc)" >> ${top_dir}scripts/acc.txt
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
  python3 p2pnet_$1.py --input ../datasets/test/images --bmodel ../models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 p2pnet_$1.py --input ../datasets/test/images --bmodel ../models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_debug.log 2>&1" log/$1_$2_debug.log
  tail -n 20 log/$1_$2_debug.log
  
  echo "==================="
  echo "Comparing statis..."
  python3 ../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=p2pnet_$1.py --language=python --input=log/$1_$2_debug.log --bmodel=$2
  judge_ret $? "python3 ../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=p2pnet_$1.py --language=python --input=log/$1_$2_debug.log  --bmodel=$2"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../tools/eval_acc.py --gt_path ../datasets/test/ground-truth --result_path results/images 2>&1 | tee log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  array[1]=${array[1]%,}
  acc=${array[1]}
  compare_res $acc $3
  judge_ret $? "python result: Precision compare!" log/$1_$2_eval.log
  printf "| %-12s | %-18s | %-25s | %8.2f |\n" "$PLATFORM" "p2pnet_$1.py" "$2" "$(printf "%.2f" $acc)" >> ${top_dir}scripts/acc.txt
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
  pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684"
  then
    eval_python opencv p2pnet_bm1684_fp32_1b.bmodel 18.35126582278481
    eval_python opencv p2pnet_bm1684_int8_1b.bmodel 20.443037974683545
    eval_python opencv p2pnet_bm1684_int8_4b.bmodel 20.443037974683545
    eval_python bmcv p2pnet_bm1684_fp32_1b.bmodel 20.199367088607595
    eval_python bmcv p2pnet_bm1684_int8_1b.bmodel 20.664556962025316
    eval_python bmcv p2pnet_bm1684_int8_4b.bmodel 20.664556962025316
    eval_cpp pcie bmcv p2pnet_bm1684_fp32_1b.bmodel 18.21
    eval_cpp pcie bmcv p2pnet_bm1684_int8_1b.bmodel 19.911392405063292
    eval_cpp pcie bmcv p2pnet_bm1684_int8_4b.bmodel 19.911392405063292
  elif test $TARGET = "BM1684X"
  then
    eval_python opencv p2pnet_bm1684x_fp32_1b.bmodel 18.35126582278481
    eval_python opencv p2pnet_bm1684x_fp16_1b.bmodel 18.341772151898734
    eval_python opencv p2pnet_bm1684x_int8_1b.bmodel 18.490506329113924
    eval_python opencv p2pnet_bm1684x_int8_4b.bmodel 18.490506329113924
    eval_python bmcv p2pnet_bm1684x_fp32_1b.bmodel 20.21518987341772
    eval_python bmcv p2pnet_bm1684x_fp16_1b.bmodel 20.20886075949367
    eval_python bmcv p2pnet_bm1684x_int8_1b.bmodel 20.335443037974684
    eval_python bmcv p2pnet_bm1684x_int8_4b.bmodel 20.335443037974684
    eval_cpp pcie bmcv p2pnet_bm1684x_fp32_1b.bmodel 18.056962025316455
    eval_cpp pcie bmcv p2pnet_bm1684x_fp16_1b.bmodel 18.15
    eval_cpp pcie bmcv p2pnet_bm1684x_int8_1b.bmodel 18.01
    eval_cpp pcie bmcv p2pnet_bm1684x_int8_4b.bmodel 17.990506329113924
	elif test $TARGET = "BM1688"
  then
    eval_python opencv p2pnet_bm1688_fp32_1b.bmodel 18.35126582278481
    eval_python opencv p2pnet_bm1688_fp16_1b.bmodel 18.33
    eval_python opencv p2pnet_bm1688_int8_1b.bmodel 18.42
    eval_python opencv p2pnet_bm1688_int8_4b.bmodel 18.42
    eval_python bmcv p2pnet_bm1688_fp32_1b.bmodel 20.21518987341772
    eval_python bmcv p2pnet_bm1688_fp16_1b.bmodel 20.17
    eval_python bmcv p2pnet_bm1688_int8_1b.bmodel 20.335443037974684
    eval_python bmcv p2pnet_bm1688_int8_4b.bmodel 20.335443037974684
    eval_cpp pcie bmcv p2pnet_bm1688_fp32_1b.bmodel 18.056962025316455
    eval_cpp pcie bmcv p2pnet_bm1688_fp16_1b.bmodel 18.15
    eval_cpp pcie bmcv p2pnet_bm1688_int8_1b.bmodel 18.10
    eval_cpp pcie bmcv p2pnet_bm1688_int8_4b.bmodel 18.10
  elif test $TARGET = "CV186X"
  then
    eval_python opencv p2pnet_cv186x_fp32_1b.bmodel 18.35126582278481
    eval_python opencv p2pnet_cv186x_fp16_1b.bmodel 18.33
    eval_python opencv p2pnet_cv186x_int8_1b.bmodel 18.43
    eval_python opencv p2pnet_cv186x_int8_4b.bmodel 18.43
    eval_python bmcv p2pnet_cv186x_fp32_1b.bmodel 20.15
    eval_python bmcv p2pnet_cv186x_fp16_1b.bmodel 20.17
    eval_python bmcv p2pnet_cv186x_int8_1b.bmodel 20.36
    eval_python bmcv p2pnet_cv186x_int8_4b.bmodel 20.36
    eval_cpp pcie bmcv p2pnet_cv186x_fp32_1b.bmodel 18.07
    eval_cpp pcie bmcv p2pnet_cv186x_fp16_1b.bmodel 18.06
    eval_cpp pcie bmcv p2pnet_cv186x_int8_1b.bmodel 18.10
    eval_cpp pcie bmcv p2pnet_cv186x_int8_4b.bmodel 18.10
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  download
  if test $TARGET = "BM1684"
  then
    eval_python opencv p2pnet_bm1684_fp32_1b.bmodel 18.35126582278481
    eval_python opencv p2pnet_bm1684_int8_1b.bmodel 20.443037974683545
    eval_python opencv p2pnet_bm1684_int8_4b.bmodel 20.443037974683545
    eval_python bmcv p2pnet_bm1684_fp32_1b.bmodel 20.199367088607595
    eval_python bmcv p2pnet_bm1684_int8_1b.bmodel 20.664556962025316
    eval_python bmcv p2pnet_bm1684_int8_4b.bmodel 20.664556962025316
    eval_cpp soc bmcv p2pnet_bm1684_fp32_1b.bmodel 18.21
    eval_cpp soc bmcv p2pnet_bm1684_int8_1b.bmodel 19.911392405063292
    eval_cpp soc bmcv p2pnet_bm1684_int8_4b.bmodel 19.911392405063292
  elif test $TARGET = "BM1684X"
  then
    eval_python opencv p2pnet_bm1684x_fp32_1b.bmodel 18.35126582278481
    eval_python opencv p2pnet_bm1684x_fp16_1b.bmodel 18.341772151898734
    eval_python opencv p2pnet_bm1684x_int8_1b.bmodel 18.490506329113924
    eval_python opencv p2pnet_bm1684x_int8_4b.bmodel 18.490506329113924
    eval_python bmcv p2pnet_bm1684x_fp32_1b.bmodel 20.21518987341772
    eval_python bmcv p2pnet_bm1684x_fp16_1b.bmodel 20.20886075949367
    eval_python bmcv p2pnet_bm1684x_int8_1b.bmodel 20.335443037974684
    eval_python bmcv p2pnet_bm1684x_int8_4b.bmodel 20.335443037974684
    eval_cpp soc bmcv p2pnet_bm1684x_fp32_1b.bmodel 18.056962025316455
    eval_cpp soc bmcv p2pnet_bm1684x_fp16_1b.bmodel 18.15
    eval_cpp soc bmcv p2pnet_bm1684x_int8_1b.bmodel 18.01
    eval_cpp soc bmcv p2pnet_bm1684x_int8_4b.bmodel 17.990506329113924
	elif test $TARGET = "BM1688"
  then
    eval_python opencv p2pnet_bm1688_fp32_1b.bmodel 18.35126582278481
    eval_python opencv p2pnet_bm1688_fp16_1b.bmodel 18.33
    eval_python opencv p2pnet_bm1688_int8_1b.bmodel 18.42
    eval_python opencv p2pnet_bm1688_int8_4b.bmodel 18.42
    eval_python bmcv p2pnet_bm1688_fp32_1b.bmodel 20.21518987341772
    eval_python bmcv p2pnet_bm1688_fp16_1b.bmodel 20.17
    eval_python bmcv p2pnet_bm1688_int8_1b.bmodel 20.335443037974684
    eval_python bmcv p2pnet_bm1688_int8_4b.bmodel 20.335443037974684
    eval_cpp soc bmcv p2pnet_bm1688_fp32_1b.bmodel 18.056962025316455
    eval_cpp soc bmcv p2pnet_bm1688_fp16_1b.bmodel 18.15
    eval_cpp soc bmcv p2pnet_bm1688_int8_1b.bmodel 18.10
    eval_cpp soc bmcv p2pnet_bm1688_int8_4b.bmodel 18.10
  elif test $TARGET = "CV186X"
  then
    eval_python opencv p2pnet_cv186x_fp32_1b.bmodel 18.35126582278481
    eval_python opencv p2pnet_cv186x_fp16_1b.bmodel 18.33
    eval_python opencv p2pnet_cv186x_int8_1b.bmodel 18.43
    eval_python opencv p2pnet_cv186x_int8_4b.bmodel 18.43
    eval_python bmcv p2pnet_cv186x_fp32_1b.bmodel 20.15
    eval_python bmcv p2pnet_cv186x_fp16_1b.bmodel 20.17
    eval_python bmcv p2pnet_cv186x_int8_1b.bmodel 20.36
    eval_python bmcv p2pnet_cv186x_int8_4b.bmodel 20.36
    eval_cpp soc bmcv p2pnet_cv186x_fp32_1b.bmodel 18.07
    eval_cpp soc bmcv p2pnet_cv186x_fp16_1b.bmodel 18.06
    eval_cpp soc bmcv p2pnet_cv186x_int8_1b.bmodel 18.10
    eval_cpp soc bmcv p2pnet_cv186x_int8_4b.bmodel 18.10
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------p2pnet acc----------"
  cat ${top_dir}scripts/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------p2pnet performance-----------"
  cat ${top_dir}tools/benchmark.txt
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
