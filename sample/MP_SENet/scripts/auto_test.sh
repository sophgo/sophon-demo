#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684X"
MODE="soc_test"
TPUID=0
ALL_PASS=1
PYTEST="auto_test"
ECHO_LINES=20
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
CASE_MODE="fully"
usage() 
{
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_test|soc_test] [ -t TARGET BM1684X|BM1688|CV186X] [ -s SOCSDK] [-a SAIL] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2 
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

PLATFORM=$TARGET
if test $MODE = "soc_test"; then
  if test $TARGET = "BM1684X"; then
    PLATFORM="SE7-32"
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
   
    if test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/mpsenet_vb_1b_fp32.bmodel
      bmrt_test_case BM1684X/mpsenet_vb_1b_bf16.bmodel
      bmrt_test_case BM1684X/mpsenet_dns_1b_fp32.bmodel
      bmrt_test_case BM1684X/mpsenet_dns_1b_bf16.bmodel
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
  chmod +x scripts/download.sh
  ./scripts/download.sh
  judge_ret $? "download" 0
}

function compile_mlir()
{
  ./scripts/gen_fp32bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  ./scripts/gen_bf16bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET bf16bmodel" 0
}


function test_python()
{
  if [ ! -d log ];then
    mkdir log
  fi
  python3 python/mp_senet_sail.py --mp_senet_model models/$TARGET/$1 --wav_files $2 --dev_id $TPUID  > log/python_test.log 2>&1
  judge_ret $? "python3 python/mp_senet_sail.py --mp_senet_model models/$TARGET/$1 --wav_files $2 --dev_id $TPUID" log/python_test.log
  tail -n 20 log/python_test.log
}

if test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_test"
then
  download
  if test $TARGET = "BM1684X"
  then
    test_python mpsenet_vb_1b_bf16.bmodel ./datasets/
  fi
elif test $MODE = "soc_test"
then
  download
  if test $TARGET = "BM1684X"
  then
    test_python mpsenet_vb_1b_bf16.bmodel ./datasets/
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
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