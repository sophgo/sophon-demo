#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir
 
#default config
TARGET="BM1684X"
MODE="pcie_test"
TPUID=0
ALL_PASS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
CASE_MODE="fully"

usage() 
{
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_build|pcie_test|soc_test|soc_build] [ -t TARGET BM1684|BM1684X|BM1688|CV186X] [ -d TPUID] [-s SOCSDK] [-a SAIL_PATH] [ -c fully|partly]" 1>&2 
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
echo "|   测试平台    |    测试程序   |              测试模型                                 | WER    |" >> scripts/acc.txt

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
     printf "| %-50s| % 15s |\n" "$1" "$time"
   done
}

function bmrt_test_benchmark(){
    pushd models
    printf "| %-50s| % 15s |\n" "测试模型" "calculate time(ms)"
    printf "| %-50s| % 15s |\n" "-------------------" "--------------"
   
    if test $TARGET = "BM1684"; then
      bmrt_test_case BM1684/wenet_encoder_streaming_fp32.bmodel
      bmrt_test_case BM1684/wenet_encoder_non_streaming_fp32.bmodel
      bmrt_test_case BM1684/wenet_decoder_fp32.bmodel
    elif test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/wenet_encoder_streaming_fp32.bmodel
      bmrt_test_case BM1684X/wenet_encoder_non_streaming_fp32.bmodel
      bmrt_test_case BM1684X/wenet_decoder_fp32.bmodel
      bmrt_test_case BM1684X/wenet_encoder_streaming_fp16.bmodel
      bmrt_test_case BM1684X/wenet_encoder_non_streaming_fp16.bmodel
      bmrt_test_case BM1684X/wenet_decoder_fp16.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/wenet_encoder_streaming_fp32.bmodel
      bmrt_test_case BM1688/wenet_encoder_non_streaming_fp32.bmodel
      bmrt_test_case BM1688/wenet_decoder_fp32.bmodel
      bmrt_test_case BM1688/wenet_encoder_streaming_fp16.bmodel
      bmrt_test_case BM1688/wenet_encoder_non_streaming_fp16.bmodel
      bmrt_test_case BM1688/wenet_decoder_fp16.bmodel
    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/wenet_encoder_streaming_fp32.bmodel
      bmrt_test_case CV186X/wenet_encoder_non_streaming_fp32.bmodel
      bmrt_test_case CV186X/wenet_decoder_fp32.bmodel
      bmrt_test_case CV186X/wenet_encoder_streaming_fp16.bmodel
      bmrt_test_case CV186X/wenet_encoder_non_streaming_fp16.bmodel
      bmrt_test_case CV186X/wenet_decoder_fp16.bmodel
    fi
  
    popd
}

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
  else
    echo "Failed: $2"
    ALL_PASS=0
  fi
  sleep 3
}
 
function download()
{
  chmod -R +x scripts/
  ./scripts/download.sh
  judge_ret $? "download"
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
  pushd cpp
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build wenet pcie"
  popd
}

function build_soc()
{
  pushd cpp
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake -DTARGET_ARCH=soc -DSDK=$SOCSDK .. && make
  judge_ret $? "build wenet soc"
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
    else
        echo "***************************************"
        echo -e "\e[42m compare right! \e[0m" #green
        echo "***************************************"
    fi
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp \n########################"
  pushd cpp
  if [ ! -d log ];then
    mkdir log
  fi
  ./wenet.$1 --encoder_bmodel=../models/$TARGET/$2 --decoder_bmodel=../models/$TARGET/$4 --dict_file=../config/lang_char.txt --config_file=../config/train_u2++_conformer.yaml --result_file=./result.txt --input=../datasets/aishell_S0764/aishell_S0764.list --mode=$3 --dev_id=$TPUID > log/$1_$2_$3_$4_debug.log 2>&1
  judge_ret $? "./wenet.$1 --encoder_bmodel=../models/$TARGET/$2 --decoder_bmodel=../models/$TARGET/$4 --dict_file=../config/lang_char.txt --config_file=../config/train_u2++_conformer.yaml --result_file=./result.txt --input=../datasets/aishell_S0764/aishell_S0764.list --mode=$3 --dev_id=$TPUID > log/$1_$2_$3_$4_debug.log 2>&1"
  tail -n 15 log/$1_$2_$3_$4_debug.log

  echo "==================="
  echo "Comparing statis..."
  python3 ../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=wenet.$1 --language=cpp --input=log/$1_$2_$3_$4_debug.log --encoder_bmodel=$2 --decoder_bmodel=$4 --mode=$3
  judge_ret $? "python3 ../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=wenet.$1 --language=cpp --input=log/$1_$2_$3_$4_debug.log --encoder_bmodel=$2 --decoder_bmodel=$4 --mode=$3"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../tools/eval_aishell.py --char=1 --v=1 ../datasets/aishell_S0764/ground_truth.txt ./result.txt)
  regex='Overall -> ([0-9]+\.[0-9]+) %'

  if [[ $res =~ $regex ]]; then
    wer="${BASH_REMATCH[1]}"
  fi 
  compare_res $wer $5
  
  bmodel_str=$2
  if [ "$3" = "attention_rescoring" ]; then
    bmodel_str="$2 + $4"
  fi
  printf "| %-12s | %-18s | %-70s | %8.3f%% |\n" "$PLATFORM" "wenet.$1" "$bmodel_str" "$(printf "%.3f" $wer)">> ../scripts/acc.txt
  popd
  echo -e "########################\nCase End: eval cpp \n########################\n"
}
 
function eval_python()
{  
  echo -e "\n########################\nCase Start: eval python \n########################"
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  pushd python
  python3 wenet.py --input ../datasets/aishell_S0764/aishell_S0764.list --encoder_bmodel ../models/$TARGET/$1 --decoder_bmodel ../models/$TARGET/$3 --dev_id $TPUID --result_file ./result.txt --mode $2 > ./log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "python3 wenet.py --input ../datasets/aishell_S0764/aishell_S0764.list --encoder_bmodel ../models/$TARGET/$1 --decoder_bmodel ../models/$TARGET/$3 --dev_id $TPUID --result_file ./result.txt --mode $2 > ./log/$1_$2_$3_debug.log 2>&1"
  tail -n 20 log/$1_$2_$3_debug.log

  echo "==================="
  echo "Comparing statis..."
  python3 ../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=wenet.py --language=python --input=log/$1_$2_$3_debug.log --encoder_bmodel=$1 --decoder_bmodel=$3 --mode=$2
  judge_ret $? "python3 ../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=wenet.py --language=python --input=log/$1_$2_$3_debug.log --encoder_bmodel=$1 --decoder_bmodel=$3 --mode=$2"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../tools/eval_aishell.py --char=1 --v=1 ../datasets/aishell_S0764/ground_truth.txt ./result.txt)
  regex='Overall -> ([0-9]+\.[0-9]+) %'

  if [[ $res =~ $regex ]]; then
    wer="${BASH_REMATCH[1]}"
  fi

  compare_res ${wer} $4

  bmodel_str=$1
  if [ "$2" = "attention_rescoring" ]; then
    bmodel_str="$1 + $3"
  fi
  printf "| %-12s | %-18s | %-70s | %8.3f%% |\n" "$PLATFORM" "wenet.py" "$bmodel_str" "$(printf "%.3f" $wer)">> ../scripts/acc.txt

  echo -e "########################\nCase End: eval python \n########################\n"
  popd
}
 
function check_dependency_ubuntu(){
    if_exit=false
    res=$(dpkg -l |grep $1)
    if [ $? != 0 ];
    then
        echo "Please install $1 on your system!"
        if_exit=true
    fi
}

if test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_build"
then
  download
  check_dependency_ubuntu libsuperlu-dev
  check_dependency_ubuntu liblapack-dev
  check_dependency_ubuntu libblas-dev
  check_dependency_ubuntu libopenblas-dev
  check_dependency_ubuntu libarmadillo-dev
  check_dependency_ubuntu libsndfile1-dev
  check_dependency_ubuntu libopenblas-dev
  check_dependency_ubuntu libyaml-cpp-dev
  check_dependency_ubuntu libyaml-cpp0.6
  check_dependency_ubuntu libbz2-dev
  check_dependency_ubuntu liblzma-dev
  if [ $if_exit == true ]; then
    judge_ret 1 "check_dependency_ubuntu"
    exit 1
  fi
  build_pcie
elif test $MODE = "pcie_test"
then
  download
  pip3 install torch==1.13.1 torchaudio==0.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684"
  then
    eval_python wenet_encoder_streaming_fp32.bmodel ctc_prefix_beam_search . 2.70
    eval_python wenet_encoder_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.80
    eval_cpp pcie wenet_encoder_streaming_fp32.bmodel ctc_prefix_beam_search . 2.55
    eval_cpp pcie wenet_encoder_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.72

    eval_python wenet_encoder_non_streaming_fp32.bmodel ctc_prefix_beam_search . 2.02
    eval_python wenet_encoder_non_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.65
    eval_cpp pcie wenet_encoder_non_streaming_fp32.bmodel ctc_prefix_beam_search . 2.02
    eval_cpp pcie wenet_encoder_non_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.65
  elif test $TARGET = "BM1684X"
  then
    eval_python wenet_encoder_streaming_fp32.bmodel ctc_prefix_beam_search . 2.70
    eval_python wenet_encoder_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.80
    eval_cpp pcie wenet_encoder_streaming_fp32.bmodel ctc_prefix_beam_search . 2.55
    eval_cpp pcie wenet_encoder_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.72

    eval_python wenet_encoder_streaming_fp16.bmodel ctc_prefix_beam_search . 2.70
    eval_python wenet_encoder_streaming_fp16.bmodel attention_rescoring wenet_decoder_fp16.bmodel 3.30
    eval_cpp pcie wenet_encoder_streaming_fp16.bmodel ctc_prefix_beam_search . 2.55
    eval_cpp pcie wenet_encoder_streaming_fp16.bmodel attention_rescoring wenet_decoder_fp16.bmodel 3.15
    
    eval_python wenet_encoder_non_streaming_fp32.bmodel ctc_prefix_beam_search . 2.02
    eval_python wenet_encoder_non_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.65
    eval_cpp pcie wenet_encoder_non_streaming_fp32.bmodel ctc_prefix_beam_search . 2.02
    eval_cpp pcie wenet_encoder_non_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.65

    eval_python wenet_encoder_non_streaming_fp16.bmodel ctc_prefix_beam_search . 2.02
    eval_python wenet_encoder_non_streaming_fp16.bmodel attention_rescoring wenet_decoder_fp16.bmodel 2.62
    eval_cpp pcie wenet_encoder_non_streaming_fp16.bmodel ctc_prefix_beam_search . 2.02
    eval_cpp pcie wenet_encoder_non_streaming_fp16.bmodel attention_rescoring wenet_decoder_fp16.bmodel 2.62
  fi
elif test $MODE = "soc_build"
then
  download
  check_dependency_ubuntu libsuperlu-dev
  if [ $if_exit == true ]; then
    judge_ret 1 "check_dependency_ubuntu"
    exit 1
  fi
  build_soc
elif test $MODE = "soc_test"
then
  export LD_LIBRARY_PATH=$PWD/cpp/cross_compile_module/ctcdecode-cpp/openfst-1.6.3/src/lib/.libs/:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$PWD/cpp/cross_compile_module/ctcdecode-cpp/build/:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$PWD/cpp/cross_compile_module/ctcdecode-cpp/build/3rd_party/kenlm/lib/:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$PWD/cpp/cross_compile_module/3rd_party/lib/:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$PWD/cpp/cross_compile_module/3rd_party/lib/blas/:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$PWD/cpp/cross_compile_module/3rd_party/lib/lapack/:$LD_LIBRARY_PATH
  download
  pip3 install torch==1.13.1 torchaudio==0.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684"
  then
    eval_python wenet_encoder_streaming_fp32.bmodel ctc_prefix_beam_search . 2.70
    eval_python wenet_encoder_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.80
    eval_cpp soc wenet_encoder_streaming_fp32.bmodel ctc_prefix_beam_search . 2.55
    eval_cpp soc wenet_encoder_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.72

    eval_python wenet_encoder_non_streaming_fp32.bmodel ctc_prefix_beam_search . 2.02
    eval_python wenet_encoder_non_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.65
    eval_cpp soc wenet_encoder_non_streaming_fp32.bmodel ctc_prefix_beam_search . 2.02
    eval_cpp soc wenet_encoder_non_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.65
  elif test $TARGET = "BM1684X"
  then
    eval_python wenet_encoder_streaming_fp32.bmodel ctc_prefix_beam_search . 2.70
    eval_python wenet_encoder_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.80
    eval_cpp soc wenet_encoder_streaming_fp32.bmodel ctc_prefix_beam_search . 2.55
    eval_cpp soc wenet_encoder_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.72

    eval_python wenet_encoder_streaming_fp16.bmodel ctc_prefix_beam_search . 2.70
    eval_python wenet_encoder_streaming_fp16.bmodel attention_rescoring wenet_decoder_fp16.bmodel 3.30
    eval_cpp soc wenet_encoder_streaming_fp16.bmodel ctc_prefix_beam_search . 2.55
    eval_cpp soc wenet_encoder_streaming_fp16.bmodel attention_rescoring wenet_decoder_fp16.bmodel 3.15
    
    eval_python wenet_encoder_non_streaming_fp32.bmodel ctc_prefix_beam_search . 2.02
    eval_python wenet_encoder_non_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.65
    eval_cpp soc wenet_encoder_non_streaming_fp32.bmodel ctc_prefix_beam_search . 2.02
    eval_cpp soc wenet_encoder_non_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.65

    eval_python wenet_encoder_non_streaming_fp16.bmodel ctc_prefix_beam_search . 2.02
    eval_python wenet_encoder_non_streaming_fp16.bmodel attention_rescoring wenet_decoder_fp16.bmodel 2.62
    eval_cpp soc wenet_encoder_non_streaming_fp16.bmodel ctc_prefix_beam_search . 2.02
    eval_cpp soc wenet_encoder_non_streaming_fp16.bmodel attention_rescoring wenet_decoder_fp16.bmodel 2.62
  elif [ "$TARGET" = "BM1688" ] || [ "$TARGET" = "CV186X" ]
  then
    eval_python wenet_encoder_streaming_fp32.bmodel ctc_prefix_beam_search . 2.77
    eval_python wenet_encoder_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.87
    eval_cpp soc wenet_encoder_streaming_fp32.bmodel ctc_prefix_beam_search . 2.70
    eval_cpp soc wenet_encoder_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 2.10

    eval_python wenet_encoder_streaming_fp16.bmodel ctc_prefix_beam_search . 2.70
    eval_python wenet_encoder_streaming_fp16.bmodel attention_rescoring wenet_decoder_fp16.bmodel 3.75
    eval_cpp soc wenet_encoder_streaming_fp16.bmodel ctc_prefix_beam_search . 2.55
    eval_cpp soc wenet_encoder_streaming_fp16.bmodel attention_rescoring wenet_decoder_fp16.bmodel 3.82
    
    eval_python wenet_encoder_non_streaming_fp32.bmodel ctc_prefix_beam_search . 2.10
    eval_python wenet_encoder_non_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 2.10
    eval_cpp soc wenet_encoder_non_streaming_fp32.bmodel ctc_prefix_beam_search . 2.10
    eval_cpp soc wenet_encoder_non_streaming_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 2.10

    eval_python wenet_encoder_non_streaming_fp16.bmodel ctc_prefix_beam_search . 2.02
    eval_python wenet_encoder_non_streaming_fp16.bmodel attention_rescoring wenet_decoder_fp16.bmodel 2.77
    eval_cpp soc wenet_encoder_non_streaming_fp16.bmodel ctc_prefix_beam_search . 2.02
    eval_cpp soc wenet_encoder_non_streaming_fp16.bmodel attention_rescoring wenet_decoder_fp16.bmodel 2.77
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------wenet wer----------"
  cat scripts/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------wenet performance-----------"
  cat tools/benchmark.txt
fi

if [ $ALL_PASS -eq 0 ]
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