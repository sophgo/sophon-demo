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
if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi
if [ -f "tools/acc.txt" ]; then
  rm tools/acc.txt
fi

usage() 
{
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET CV186X|BM1684X|BM1688] [ -s SOCSDK] [-a SAIL] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2 
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
  ./scripts/download.sh --$TARGET
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

PLATFORM=$TARGET
if test $MODE = "soc_test"; then
  if test $TARGET = "BM1684X"; then
    PLATFORM="SE7-32"
  elif test $TARGET = "CV186X"; then
    PLATFORM="SE9-8"
  elif test $TARGET = "BM1688"; then
    PLATFORM="SE9-16"
  else
    echo "Unknown TARGET type: $TARGET"
  fi
fi

function bmrt_test_case(){
   calculate_time_log=$(bmrt_test --bmodel $1 | grep "calculate" 2>&1)
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
      bmrt_test_case BM1684X/real_esrgan_fp32_1b.bmodel
      bmrt_test_case BM1684X/real_esrgan_fp16_1b.bmodel
      bmrt_test_case BM1684X/real_esrgan_int8_1b.bmodel
      bmrt_test_case BM1684X/real_esrgan_int8_4b.bmodel
    elif test $TARGET = "CV186X"; then
      bmrt_test_case BM1688/real_esrgan_fp32_1b.bmodel
      bmrt_test_case BM1688/real_esrgan_fp16_1b.bmodel
      bmrt_test_case BM1688/real_esrgan_int8_1b.bmodel
      bmrt_test_case BM1688/real_esrgan_int8_4b.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/real_esrgan_fp32_1b.bmodel
      bmrt_test_case BM1688/real_esrgan_fp16_1b.bmodel
      bmrt_test_case BM1688/real_esrgan_int8_1b.bmodel
      bmrt_test_case BM1688/real_esrgan_int8_4b.bmodel
      bmrt_test_case BM1688/real_esrgan_fp32_1b_2core.bmodel
      bmrt_test_case BM1688/real_esrgan_fp16_1b_2core.bmodel
      bmrt_test_case BM1688/real_esrgan_int8_1b_2core.bmodel
      bmrt_test_case BM1688/real_esrgan_int8_4b_2core.bmodel
    fi
    popd
}

function build_pcie()
{
  pushd cpp/real_esrgan_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build real_esrgan_$1" 0
  popd
}

function build_soc()
{
  pushd cpp/real_esrgan_$1
  if [ -d build ]; then
      rm -rf build
  fi
  if test $1 = "sail"; then
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK -DSAIL_PATH=$SAIL_PATH && make
    judge_ret $? "build soc real_esrgan_$1" 0
  else
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
    judge_ret $? "build soc real_esrgan_$1" 0
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

function eval_python()
{  
  echo -e "\n########################\nCase Start: eval python\n########################"
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  if [ ! -d results/images_onnx ];then
    python3 python/real_esrgan_onnx.py --input datasets/coco128 --onnx models/onnx/realesr-general-x4v3.onnx 
  fi

  python3 python/real_esrgan_$1.py --input datasets/coco128 --bmodel models/$TARGET/$2 --dev_id $TPUID > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/real_esrgan_$1.py --input datasets/coco128 --bmodel models/$TARGET/$2 --dev_id $TPUID > python/log/$1_$2_debug.log 2>&1" python/log/$1_$2_debug.log
  tail -n 20 python/log/$1_$2_debug.log

  echo "==================="
  echo "Comparing statis..."
  python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=real_esrgan_$1.py --language=python --input=python/log/$1_$2_debug.log --bmodel=$2
  judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=real_esrgan_$1.py --language=python --input=log/$1_$2_debug.log --bmodel=$2"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 tools/eval_psnr.py --left_results results/images_onnx --right_results results/images_$1 2>&1 | tee python/log/$1_$2_eval.log)
  echo "==================="
  echo "Comparing acc..."
  python3 tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=real_esrgan_$1.py --language=python --input=python/log/$1_$2_eval.log --bmodel=$2 2>&1
  judge_ret $? "python3 tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=real_esrgan_$1.py --language=python --input=python/log/$1_$2_eval.log --bmodel=$2"
  echo "==================="
  echo -e "########################\nCase End: eval python\n########################\n"
}
function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/real_esrgan_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./real_esrgan_$2.$1 --input=../../datasets/coco128 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./real_esrgan_$2.$1 --input=../../datasets/coco128 --bmodel=../../models/$TARGET/$3  --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1" log/$1_$2_$3_debug.log
  tail -n 15 log/$1_$2_$3_debug.log

  echo "==================="
  echo "Comparing statis..."
  python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=real_esrgan_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3
  judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=real_esrgan_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../../tools/eval_psnr.py --left_results ../../results/images_onnx --right_results results/images 2>&1 | tee log/$1_$2_$3_eval.log)
  echo "==================="
  echo "Comparing acc..."
  python3 ../../tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=real_esrgan_$2.$1 --language=cpp --input=log/$1_$2_$3_eval.log --bmodel=$3 2>&1
  judge_ret $? "python3 ../../tools/compare_acc.py --target=$TARGET --platform=${MODE%_*} --program=real_esrgan_$2.$1 --language=cpp --input=log/$1_$2_$3_eval.log --bmodel=$3"
  echo "==================="
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}
if test $MODE = "compile_mlir"
then
  download onnx
  compile_mlir
elif test $MODE = "pcie_build"
then
  build_pcie bmcv
elif test $MODE = "pcie_test"
then
  download $TARGET
  pip3 install onnxruntime==1.14.1 opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  for pre in fp32_1b fp16_1b int8_1b int8_4b; do
    eval_python opencv real_esrgan_${pre}.bmodel
  done
  if test $TARGET = "BM1688"; then
    eval_python opencv real_esrgan_int8_4b_2core.bmodel
  fi
  for pre in fp32_1b fp16_1b int8_1b int8_4b; do
    eval_python bmcv real_esrgan_${pre}.bmodel
  done
  if test $TARGET = "BM1688"; then
    eval_python bmcv real_esrgan_int8_4b_2core.bmodel
  fi
  for pre in fp32_1b fp16_1b int8_1b int8_4b; do
    eval_cpp pcie bmcv real_esrgan_${pre}.bmodel
  done
  if test $TARGET = "BM1688"; then
    eval_cpp pcie bmcv real_esrgan_int8_4b_2core.bmodel
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  download $TARGET
  if [ ! -d results ];then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Real-ESRGAN/onnx_results/images_onnx.tgz
    mkdir results/
    mv images_onnx.tgz results/
    cd results
    tar -zxvf images_onnx.tgz
    rm images_onnx.tgz
    cd ..
  fi
  pip3 install onnxruntime==1.14.1 opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  for pre in fp32_1b fp16_1b int8_1b int8_4b; do
    eval_python opencv real_esrgan_${pre}.bmodel
  done
  if test $TARGET = "BM1688"; then
    eval_python opencv real_esrgan_int8_4b_2core.bmodel
  fi
  for pre in fp32_1b fp16_1b int8_1b int8_4b; do
    eval_python bmcv real_esrgan_${pre}.bmodel
  done
  if test $TARGET = "BM1688"; then
    eval_python bmcv real_esrgan_int8_4b_2core.bmodel
  fi
  for pre in fp32_1b fp16_1b int8_1b int8_4b; do
    eval_cpp soc bmcv real_esrgan_${pre}.bmodel
  done
  if test $TARGET = "BM1688"; then
    eval_cpp soc bmcv real_esrgan_int8_4b_2core.bmodel
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ] 
then
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------real_esrgan performance-----------"
  cat tools/benchmark.txt
  echo "--------real_esrgan acc-----------"
  cat tools/acc.txt
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