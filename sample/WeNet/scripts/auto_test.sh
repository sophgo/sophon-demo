#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir
 
#default config
TARGET="BM1684X"
MODE="pcie_test"
TPUID=0
ALL_PASS=1
 
usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|pcie_test|soc_test] [ -t TARGET BM1684|BM1684X] [ -d TPUID]" 1>&2 
}
 
while getopts ":m:t:s:a:d:" opt
do
  case $opt in 
    m)
      MODE=${OPTARG}
      echo "mode is $MODE";;
    t)
      TARGET=${OPTARG}
      echo "target is $TARGET";;
    d)
      TPUID=${OPTARG}
      echo "using tpu $TPUID";;
    ?)
      usage
      exit 1;;
  esac
done
 
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
 
function compile_nntc()
{
  ./scripts/gen_fp32bmodel_nntc.sh BM1684
  judge_ret $? "generate BM1684 fp32bmodel"
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
  cmake -DTARGET_ARCH=soc .. && make
  judge_ret $? "build wenet soc"
  popd
}
 
function compare_res(){
    ret=`awk -v x=$1 -v y=$2 'BEGIN{print(x-y<0.0001 && y-x<0.0001)?1:0}'`
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
 
function test_cpp()
{
  pushd cpp
  ./wenet.$1 --encoder_bmodel=../models/$TARGET/$2 --decoder_bmodel=../models/$TARGET/$4 --dict_file=../config/lang_char.txt --config_file=../config/train_u2++_conformer.yaml --result_file=./result.txt --input=../datasets/aishell_S0764/aishell_S0764.list --mode=$3 --dev_id=$TPUID > log/$1_$2_$3_$4_debug.log 2>&1
  judge_ret $? "./wenet.$1 --encoder_bmodel=../models/$TARGET/$2 --decoder_bmodel=../models/$TARGET/$4 --dict_file=../config/lang_char.txt --config_file=../config/train_u2++_conformer.yaml --result_file=./result.txt --input=../datasets/aishell_S0764/aishell_S0764.list --mode=$3 --dev_id=$TPUID > log/$1_$2_$3_$4_debug.log 2>&1"
  popd
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
  
  echo "Evaluating..."
  res=$(python3 ../tools/eval_aishell.py --char=1 --v=1 ../datasets/aishell_S0764/ground_truth.txt ./result.txt)
  regex='Overall -> ([0-9]+\.[0-9]+) %'

  if [[ $res =~ $regex ]]; then
    wer="${BASH_REMATCH[1]}"
  fi 
  compare_res $wer $5
  popd
  echo -e "########################\nCase End: eval cpp \n########################\n"
}
 
function test_python()
{
  pushd python
  python3 wenet.py --input ../datasets/aishell_S0764/aishell_S0764.list --encoder_bmodel ../models/$TARGET/$1 --decoder_bmodel ../models/$TARGET/$3 --dev_id $TPUID --result_file ./result.txt --mode $2
  judge_ret $? "python3 wenet.py --input ../datasets/aishell_S0764/aishell_S0764.list --encoder_bmodel ../models/$TARGET/$1 --decoder_bmodel ../models/$TARGET/$3 --dev_id $TPUID --result_file ./result.txt --mode $2"
  popd
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
  
  echo "Evaluating..."
  res=$(python3 ../tools/eval_aishell.py --char=1 --v=1 ../datasets/aishell_S0764/ground_truth.txt ./result.txt)
  regex='Overall -> ([0-9]+\.[0-9]+) %'

  if [[ $res =~ $regex ]]; then
    wer="${BASH_REMATCH[1]}"
  fi

  compare_res ${wer} $4
  echo -e "########################\nCase End: eval python \n########################\n"
  popd
}
 
if test $MODE = "compile_nntc"
then
  download
  compile_nntc
elif test $MODE = "pcie_test"
then
  build_pcie
  download pcie
  if test $TARGET = "BM1684"
  then
    test_python wenet_encoder_fp32.bmodel ctc_prefix_beam_search .
    test_cpp pcie wenet_encoder_fp32.bmodel ctc_prefix_beam_search .

    eval_python wenet_encoder_fp32.bmodel ctc_prefix_beam_search . 2.70
    eval_cpp pcie wenet_encoder_fp32.bmodel ctc_prefix_beam_search . 2.70
 
  elif test $TARGET = "BM1684X"
  then
    test_python wenet_encoder_fp32.bmodel ctc_prefix_beam_search .
    test_python wenet_encoder_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel
    test_cpp pcie wenet_encoder_fp32.bmodel ctc_prefix_beam_search .
    test_cpp pcie wenet_encoder_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel

    eval_python wenet_encoder_fp32.bmodel ctc_prefix_beam_search . 2.70
    eval_python wenet_encoder_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.80
    eval_cpp pcie wenet_encoder_fp32.bmodel ctc_prefix_beam_search . 2.70
    eval_cpp pcie wenet_encoder_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.80
  fi
elif test $MODE = "soc_test"
then
  download soc
  if test $TARGET = "BM1684"
  then
    test_python wenet_encoder_fp32.bmodel ctc_prefix_beam_search .
    test_cpp soc wenet_encoder_fp32.bmodel ctc_prefix_beam_search .

    eval_python wenet_encoder_fp32.bmodel ctc_prefix_beam_search . 2.70
    eval_cpp soc wenet_encoder_fp32.bmodel ctc_prefix_beam_search . 2.70
  elif test $TARGET = "BM1684X"
  then
    test_python wenet_encoder_fp32.bmodel ctc_prefix_beam_search .
    test_python wenet_encoder_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel
    test_cpp soc wenet_encoder_fp32.bmodel ctc_prefix_beam_search .
    test_cpp soc wenet_encoder_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel

    eval_python wenet_encoder_fp32.bmodel ctc_prefix_beam_search . 2.70
    eval_python wenet_encoder_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.80
    eval_cpp soc wenet_encoder_fp32.bmodel ctc_prefix_beam_search . 2.70
    eval_cpp soc wenet_encoder_fp32.bmodel attention_rescoring wenet_decoder_fp32.bmodel 1.80
  fi
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