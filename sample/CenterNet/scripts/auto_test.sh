#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
echo "-------------"
echo $scripts_dir
top_dir=$scripts_dir/../
pushd $top_dir

#soc_build需要传入sail路径(-a)和socsdk路径(-s)

#default config
TARGET="BM1684X"
MODE="pcie_test"
TPUID=0
ALL_PASS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi

usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [ -d TPUID] [ -a SAIL]" 1>&2 
}

while getopts ":m:t:s:d:a:" opt
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
    a)
      SAIL=${OPTARG}
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
      echo "sail is $SAIL";;  
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
  ./scripts/gen_int8bmodel_nntc.sh BM1684
  judge_ret $? "generate BM1684 int8bmodel"
}

function compile_mlir()
{
  ./scripts/gen_fp32bmodel_mlir.sh bm1684x
  judge_ret $? "generate BM1684X fp32bmodel"
  ./scripts/gen_fp16bmodel_mlir.sh bm1684x
  judge_ret $? "generate BM1684X fp16bmodel"
  ./scripts/gen_int8bmodel_mlir.sh bm1684x
  judge_ret $? "generate BM1684X int8bmodel"
}

function build_pcie()
{
  pushd cpp/centernet_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  if [ "$1" == "bmcv" ]; then
    cmake .. && make
  else
    cmake .. && make
  fi
  judge_ret $? "build centernet_$1"
  popd
}

function build_soc()
{
  pushd cpp/centernet_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  if [ "$1" == "bmcv" ]; then
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  else
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK -DSAIL_PATH=$SAIL && make
  fi
  judge_ret $? "build soc centernet_$1"
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
    else
        echo "***************************************"
        echo -e "\e[42m compare right! \e[0m" #green
        echo "***************************************"
    fi
}

function test_cpp()
{
  pushd cpp/centernet_$2
  ./centernet_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID
  judge_ret $? "./centernet_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id $TPUID"
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/centernet_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./centernet_$2.$1 --input=../../datasets/coco/val2017_1000 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.35 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./centernet_$2.$1 --input=../../datasets/coco/val2017_1000 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.35 --dev_id $TPUID > log/$1_$2_$3_debug.log 2>&1"
  tail -n 15 log/$1_$2_$3_debug.log

  echo "==================="
  echo "Comparing statis..."
  python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=centernet_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3
  judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=centernet_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../../tools/eval_coco.py --gt_path ../../datasets/coco/instances_val2017_1000.json --result_json results/$3_val2017_1000_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"

  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function test_python()
{
  python3 python/centernet_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID
  judge_ret $? "python3 python/centernet_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID"
}

function eval_python()
{  
  echo -e "\n########################\nCase Start: eval python\n########################"
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  python3 python/centernet_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.35  > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/centernet_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.35  > python/log/$1_$2_debug.log 2>&1"
  tail -n 15 python/log/$1_$2_debug.log

  echo "==================="
  echo "Comparing statis..."
  python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=centernet_$1.py --language=python --input=python/log/$1_$2_debug.log --bmodel=$2
  judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=centernet_$1.py --language=python --input=python/log/$1_$2_debug.log --bmodel=$2"
  echo "==================="
  
  echo "Evaluating..."
  res=$(python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/$2_val2017_1000_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
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
    test_python opencv centernet_fp32_1b.bmodel datasets/test
    test_python opencv centernet_int8_4b.bmodel datasets/test
    test_python bmcv centernet_fp32_1b.bmodel datasets/test
    test_python bmcv centernet_int8_4b.bmodel datasets/test
    test_cpp pcie bmcv centernet_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv centernet_int8_4b.bmodel ../../datasets/test
    test_cpp pcie sail centernet_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie sail centernet_int8_4b.bmodel ../../datasets/test


    eval_python opencv centernet_fp32_1b.bmodel 0.30170771852182776
    eval_python opencv centernet_int8_1b.bmodel 0.2954742860461241
    eval_python opencv centernet_int8_4b.bmodel 0.2954742860461241
    eval_python bmcv centernet_fp32_1b.bmodel 0.2579835082367173
    eval_python bmcv centernet_int8_1b.bmodel 0.24969053924051185
    eval_python bmcv centernet_int8_4b.bmodel 0.24969053924051185
    eval_cpp pcie bmcv centernet_fp32_1b.bmodel 0.26801259211506495
    eval_cpp pcie bmcv centernet_int8_1b.bmodel 0.2610760844102559
    eval_cpp pcie bmcv centernet_int8_4b.bmodel 0.2610760844102559
    eval_cpp pcie sail centernet_fp32_1b.bmodel 0.29672149546058746
    eval_cpp pcie sail centernet_int8_1b.bmodel 0.2897742204890654
    eval_cpp pcie sail centernet_int8_4b.bmodel 0.2897742204890654

  elif test $TARGET = "BM1684X"
  then
    test_python opencv centernet_fp32_1b.bmodel datasets/test
    test_python opencv centernet_int8_4b.bmodel datasets/test
    test_python bmcv centernet_fp32_1b.bmodel datasets/test
    test_python bmcv centernet_int8_4b.bmodel datasets/test
    test_cpp pcie bmcv centernet_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv centernet_int8_4b.bmodel ../../datasets/test
    test_cpp pcie sail centernet_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie sail centernet_int8_4b.bmodel ../../datasets/test

    eval_python opencv centernet_fp32_1b.bmodel 0.30170771852182776
    eval_python opencv centernet_fp16_1b.bmodel 0.30182218193493354
    eval_python opencv centernet_int8_1b.bmodel 0.29903951060724937
    eval_python opencv centernet_int8_4b.bmodel 0.29903951060724937
    eval_python bmcv centernet_fp32_1b.bmodel 0.2583137763887781
    eval_python bmcv centernet_fp16_1b.bmodel 0.25835675947956765
    eval_python bmcv centernet_int8_1b.bmodel 0.2569946819387488
    eval_python bmcv centernet_int8_4b.bmodel 0.2569946819387488
    eval_cpp pcie bmcv centernet_fp32_1b.bmodel 0.2676595517464278
    eval_cpp pcie bmcv centernet_fp16_1b.bmodel 0.2684261224257983
    eval_cpp pcie bmcv centernet_int8_1b.bmodel 0.26419835507400347
    eval_cpp pcie bmcv centernet_int8_4b.bmodel 0.26419835507400347
    eval_cpp pcie sail centernet_fp32_1b.bmodel 0.29605632005365545
    eval_cpp pcie sail centernet_fp16_1b.bmodel 0.29565463247983764
    eval_cpp pcie sail centernet_int8_1b.bmodel 0.2938854537962771
    eval_cpp pcie sail centernet_int8_4b.bmodel 0.2938854537962771
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
  build_soc sail

elif test $MODE = "soc_test"
then
  download
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
  if test $TARGET = "BM1684"
  then
    test_python opencv centernet_fp32_1b.bmodel datasets/test
    test_python opencv centernet_int8_4b.bmodel datasets/test
    test_python bmcv centernet_fp32_1b.bmodel datasets/test
    test_python bmcv centernet_int8_4b.bmodel datasets/test
    test_cpp soc bmcv centernet_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv centernet_int8_4b.bmodel ../../datasets/test
    test_cpp soc sail centernet_fp32_1b.bmodel ../../datasets/test
    test_cpp soc sail centernet_int8_4b.bmodel ../../datasets/test

    eval_python opencv centernet_fp32_1b.bmodel 0.30170771852182776
    eval_python opencv centernet_int8_1b.bmodel 0.2954742860461241
    eval_python opencv centernet_int8_4b.bmodel 0.2954742860461241
    eval_python bmcv centernet_fp32_1b.bmodel 0.2579835082367173
    eval_python bmcv centernet_int8_1b.bmodel 0.24969053924051185
    eval_python bmcv centernet_int8_4b.bmodel 0.24969053924051185
    eval_cpp soc bmcv centernet_fp32_1b.bmodel 0.26801259211506495 
    eval_cpp soc bmcv centernet_int8_1b.bmodel 0.2610760844102559
    eval_cpp soc bmcv centernet_int8_4b.bmodel 0.2610760844102559
    eval_cpp soc sail centernet_fp32_1b.bmodel 0.29672149546058746 
    eval_cpp soc sail centernet_int8_1b.bmodel 0.2897742204890654
    eval_cpp soc sail centernet_int8_4b.bmodel 0.2897742204890654
  elif test $TARGET = "BM1684X"
  then
    test_python opencv centernet_fp32_1b.bmodel datasets/test
    test_python opencv centernet_int8_4b.bmodel datasets/test
    test_python bmcv centernet_fp32_1b.bmodel datasets/test
    test_python bmcv centernet_int8_4b.bmodel datasets/test
    test_cpp soc bmcv centernet_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv centernet_int8_4b.bmodel ../../datasets/test
    test_cpp soc sail centernet_fp32_1b.bmodel ../../datasets/test
    test_cpp soc sail centernet_int8_4b.bmodel ../../datasets/test

    eval_python opencv centernet_fp32_1b.bmodel 0.30170771852182776
    eval_python opencv centernet_fp16_1b.bmodel 0.30182218193493354
    eval_python opencv centernet_int8_1b.bmodel 0.29903951060724937
    eval_python opencv centernet_int8_4b.bmodel 0.29903951060724937
    eval_python bmcv centernet_fp32_1b.bmodel 0.2583137763887781
    eval_python bmcv centernet_fp16_1b.bmodel 0.25835675947956765
    eval_python bmcv centernet_int8_1b.bmodel 0.2569946819387488
    eval_python bmcv centernet_int8_4b.bmodel 0.2569946819387488 
    eval_cpp soc bmcv centernet_fp32_1b.bmodel 0.2676595517464278
    eval_cpp soc bmcv centernet_fp16_1b.bmodel 0.2684261224257983
    eval_cpp soc bmcv centernet_int8_1b.bmodel 0.26419835507400347
    eval_cpp soc bmcv centernet_int8_4b.bmodel 0.26419835507400347
    eval_cpp soc sail centernet_fp32_1b.bmodel 0.29605632005365545
    eval_cpp soc sail centernet_fp16_1b.bmodel 0.29565463247983764
    eval_cpp soc sail centernet_int8_1b.bmodel 0.2938854537962771
    eval_cpp soc sail centernet_int8_4b.bmodel 0.2938854537962771
  elif test $TARGET = "BM1688"
  then
    test_python opencv centernet_fp32_1b.bmodel datasets/test
    test_python opencv centernet_int8_4b.bmodel datasets/test
    test_python bmcv centernet_fp32_1b.bmodel datasets/test
    test_python bmcv centernet_int8_4b.bmodel datasets/test
    test_cpp soc bmcv centernet_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv centernet_int8_4b.bmodel ../../datasets/test
    test_cpp soc sail centernet_fp32_1b.bmodel ../../datasets/test
    test_cpp soc sail centernet_int8_4b.bmodel ../../datasets/test

    eval_python opencv centernet_fp32_1b.bmodel 0.30170771852182776  
    eval_python opencv centernet_fp16_1b.bmodel 0.30182218193493354  
    eval_python opencv centernet_int8_1b.bmodel 0.2980216301853383    
    eval_python opencv centernet_int8_4b.bmodel 0.2980216301853383    
    eval_python bmcv centernet_fp32_1b.bmodel   0.2583137763887781    
    eval_python bmcv centernet_fp16_1b.bmodel   0.25835675947956765  
    eval_python bmcv centernet_int8_1b.bmodel   0.2545645328723769  
    eval_python bmcv centernet_int8_4b.bmodel   0.2545645328723769  
    eval_cpp soc bmcv centernet_fp32_1b.bmodel  0.2676595517464278  
    eval_cpp soc bmcv centernet_fp16_1b.bmodel  0.26694320131206006  
    eval_cpp soc bmcv centernet_int8_1b.bmodel  0.2652181489471362  
    eval_cpp soc bmcv centernet_int8_4b.bmodel  0.2652181489471362  
    eval_cpp soc sail centernet_fp32_1b.bmodel  0.29605632005365545  
    eval_cpp soc sail centernet_fp16_1b.bmodel  0.29565463247983764  
    eval_cpp soc sail centernet_int8_1b.bmodel  0.2888585987297452  
    eval_cpp soc sail centernet_int8_4b.bmodel  0.29233540424498117  

    eval_python opencv centernet_fp32_1b_2core.bmodel 0.30170771852182776
    eval_python opencv centernet_fp16_1b_2core.bmodel 0.30182218193493354
    eval_python opencv centernet_int8_1b_2core.bmodel 0.2980216301853383 
    eval_python opencv centernet_int8_4b_2core.bmodel 0.2980216301853383 
    eval_python bmcv centernet_fp32_1b_2core.bmodel   0.2583137763887781 
    eval_python bmcv centernet_fp16_1b_2core.bmodel   0.25835675947956765
    eval_python bmcv centernet_int8_1b_2core.bmodel   0.2545645328723769 
    eval_python bmcv centernet_int8_4b_2core.bmodel   0.2545645328723769  
    eval_cpp soc bmcv centernet_fp32_1b_2core.bmodel  0.2676595517464278 
    eval_cpp soc bmcv centernet_fp16_1b_2core.bmodel  0.26694320131206006
    eval_cpp soc bmcv centernet_int8_1b_2core.bmodel  0.2652181489471362 
    eval_cpp soc bmcv centernet_int8_4b_2core.bmodel  0.2652181489471362 
    eval_cpp soc sail centernet_fp32_1b_2core.bmodel  0.29605632005365545
    eval_cpp soc sail centernet_fp16_1b_2core.bmodel  0.29565463247983764
    eval_cpp soc sail centernet_int8_1b_2core.bmodel  0.2888585987297452 
    eval_cpp soc sail centernet_int8_4b_2core.bmodel  0.29233540424498117
  fi
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