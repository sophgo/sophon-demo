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
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_build|pcie_test|soc_build|soc_test] [ -t TARGET BM1684X|BM1688|CV186X] [ -s SOCSDK] [-a SAIL] [ -d TPUID] [ -p PYTEST auto_test|pytest] [ -c fully|partly]" 1>&2 
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
echo "| 测试平台      |  测试程序            |    superpoint模型         |           superglue模型             |MScore|" >> scripts/acc.txt

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
     printf "| %-45s| % 15s |\n" "$1" "$time"
   done
}
function bmrt_test_benchmark(){
    pushd models
    printf "| %-45s| % 15s |\n" "测试模型" "calculate time(ms)"
    printf "| %-45s| % 15s |\n" "-------------------" "--------------"
   
    if test $TARGET = "BM1684"; then
      echo "Not support BM1684 now"
    elif test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/superpoint_fp32_1b.bmodel
      bmrt_test_case BM1684X/superpoint_fp16_1b.bmodel
      bmrt_test_case BM1684X/superglue_fp32_1b_iter20_1024.bmodel  
      bmrt_test_case BM1684X/superglue_fp16_1b_iter20_1024.bmodel  
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/superpoint_fp32_1b.bmodel
      bmrt_test_case BM1688/superpoint_fp16_1b.bmodel
      bmrt_test_case BM1688/superglue_fp32_1b_iter20_1024.bmodel  
      bmrt_test_case BM1688/superglue_fp16_1b_iter20_1024.bmodel  
    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/superpoint_fp32_1b.bmodel
      bmrt_test_case CV186X/superpoint_fp16_1b.bmodel
      bmrt_test_case CV186X/superglue_fp32_1b_iter20_1024.bmodel  
      bmrt_test_case CV186X/superglue_fp16_1b_iter20_1024.bmodel  
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
  ./scripts/gen_fp16bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp16bmodel" 0
}

function build_pcie()
{
  pushd cpp/superglue_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build superglue_$1" 0
  popd
}

function build_soc()
{
  pushd cpp/superglue_$1
  if [ -d build ]; then
      rm -rf build
  fi
  if test $1 = "sail"; then
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK -DSAIL_PATH=$SAIL_PATH && make
    judge_ret $? "build soc superglue_$1" 0
  else
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
    judge_ret $? "build soc superglue_$1" 0
  fi
  popd
}

function compare_res(){
    ret=`awk -v x=$1 -v y=$2 'BEGIN{print(x-y<1 && y-x<1)?1:0}'`
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
  pushd cpp/superglue_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./superglue_$2.$1 --bmodel_superpoint=../../models/$TARGET/$3 --bmodel_superglue=../../models/$TARGET/$4 --dev_id=$TPUID > log/$1_$2_$3_$4_debug.log 2>&1
  judge_ret $? "../../datasets/scannet_sample_pairs_with_gt.txt" log/$1_$2_$3_$4_debug.log
  tail -n 15 log/$1_$2_$3_$4_debug.log

  echo "==================="
  echo "Comparing statis..."
  python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=superglue_$2.$1 --language=cpp --input=log/$1_$2_$3_$4_debug.log --bmodel_superpoint=$3 --bmodel_superglue=$4
  judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=superglue_$2.$1 --language=cpp --input=log/$1_$2_$3_$4_debug.log --bmodel_superpoint=$3 --bmodel_superglue=$4"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../../tools/eval.py --input_pairs ../../datasets/scannet_sample_pairs_with_gt.txt --result_json results/result.json 2>&1 | tee log/$1_$2_$3_$4_eval.log)
  echo -e "$res"

  mscore=$(echo "$res" | grep -A 2 MScore | awk 'NR==2 {print $5}')
  compare_res $mscore $5
  judge_ret $? "Precision compare!" log/$1_$2_$3_$4_eval.log

  printf "| %-12s | %-18s | %-20s | %-30s | %8.2f |\n" "$PLATFORM" "superglue_$2.$1" "$3" "$4" "$(printf "%.2f" $mscore)" >> ../../scripts/acc.txt

  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function check_dependencies_ubuntu_amd64(){
    if_exit=false
    res=$(dpkg -l |grep libopenblas-dev:amd64)
    if [ $? != 0 ];
    then
        echo "Please install libopenblas-dev:amd64 on your system!"
        if_exit=true
    fi
    res=$(dpkg -l |grep libhwloc-dev:amd64)
    if [ $? != 0 ];
    then
        echo "Please install libhwloc-dev:amd64 on your system!"
        if_exit=true
    fi
    if [ $if_exit == true ]; then
        exit 1
    fi
}

function check_dependencies_ubuntu_arm64(){
    if_exit=false
    res=$(dpkg -l |grep libopenblas-dev:arm64)
    if [ $? != 0 ];
    then
        echo "Please install libopenblas-dev:arm64 on your system!"
        if_exit=true
    fi
    res=$(dpkg -l |grep ccache:arm64)
    if [ $? != 0 ];
    then
        echo "Please install ccache:arm64 on your system!"
        if_exit=true
    fi
    res=$(dpkg -l |grep numactl:arm64)
    if [ $? != 0 ];
    then
        echo "Please install numactl:arm64 on your system!"
        if_exit=true
    fi
    
    res=$(dpkg -l |grep libhwloc-dev:arm64)
    if [ $? != 0 ];
    then
        echo "Please install libhwloc-dev:arm64 on your system!"
        if_exit=true
    fi
    if [ $if_exit == true ]; then
        exit 1
    fi
}

if test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_build"
then
  check_dependencies_ubuntu_amd64
  download
  build_pcie bmcv
elif test $MODE = "pcie_test"
then
  pip3 install opencv-python-headless matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684"
  then
    echo "Not support BM1684 yet."
  elif test $TARGET = "BM1684X"
  then
    eval_cpp pcie bmcv superpoint_fp32_1b.bmodel superglue_fp32_1b_iter20_1024.bmodel 16.90
    eval_cpp pcie bmcv superpoint_fp16_1b.bmodel superglue_fp16_1b_iter20_1024.bmodel 16.72
  fi
elif test $MODE = "soc_build"
then
  check_dependencies_ubuntu_arm64
  download
  build_soc bmcv
elif test $MODE = "soc_test"
then
  download
  pip3 install opencv-python-headless matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
  yes Y | sudo apt install libopenblas-dev
  export LD_LIBRARY_PATH=$PWD/cpp/aarch64_lib/libtorch/lib:$LD_LIBRARY_PATH
  if test $TARGET = "BM1684"
  then
    echo "Not support BM1684 yet."
  elif test $TARGET = "BM1684X"
  then
    eval_cpp soc bmcv superpoint_fp32_1b.bmodel superglue_fp32_1b_iter20_1024.bmodel 16.90
    eval_cpp soc bmcv superpoint_fp16_1b.bmodel superglue_fp16_1b_iter20_1024.bmodel 16.69
  elif [ "$TARGET" = "BM1688" ] || [ "$TARGET" = "CV186X" ]
  then
    eval_cpp soc bmcv superpoint_fp32_1b.bmodel superglue_fp32_1b_iter20_1024.bmodel 16.90
    eval_cpp soc bmcv superpoint_fp16_1b.bmodel superglue_fp16_1b_iter20_1024.bmodel 16.71
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------superglue mscore----------"
  cat scripts/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------superglue performance-----------"
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