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
export PYTHONPATH=/opt/sophon/sophon-opencv-latest/opencv-python/:$PYTHONPATH
if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
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
  echo -e "$res"
  psnr=$(echo "$res" | grep -oP 'average_psnr:\s+\K[0-9.]+')
  echo -e "$psnr"
  compare_res $psnr $3
  judge_ret $? "$2_$1_python_result: Precision compare!" log/$1_$2_eval.log

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
  echo -e "$res"
  psnr=$(echo "$res" | grep -oP 'average_psnr:\s+\K[0-9.]+')
  echo -e "$psnr"
  compare_res $psnr $4
  judge_ret $? "$2_$3_cpp_result: Precision compare!" log/$1_$2_$3_eval.log
  
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}
if test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_test"
then
  build_pcie bmcv
  download
  pip3 install onnxruntime==1.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684X"
  then
    #performence test
    eval_python opencv real_esrgan_fp32_1b.bmodel 79.39919963297913
    eval_python opencv real_esrgan_fp16_1b.bmodel 51.58411009845114
    eval_python opencv real_esrgan_int8_1b.bmodel 36.62011081880808
    eval_python opencv real_esrgan_int8_4b.bmodel 36.62011081880808
    eval_python bmcv real_esrgan_fp32_1b.bmodel 45.517305468646484
    eval_python bmcv real_esrgan_fp16_1b.bmodel 45.568800619928
    eval_python bmcv real_esrgan_int8_1b.bmodel 36.41790996020899
    eval_python bmcv real_esrgan_int8_4b.bmodel 36.41790996020899

    eval_cpp pcie bmcv real_esrgan_fp32_1b.bmodel 38.56451536398317
    eval_cpp pcie bmcv real_esrgan_fp16_1b.bmodel 38.56165460548448
    eval_cpp pcie bmcv real_esrgan_int8_1b.bmodel 34.98144302652623
    eval_cpp pcie bmcv real_esrgan_int8_4b.bmodel 34.98144626156269
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  download
  if [ ! -d results ];then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Real-ESRGAN/onnx_results/images_onnx.tgz
    mkdir results/
    mv images_onnx.tgz results/
    cd results
    tar -zxvf images_onnx.tgz
    rm images_onnx.tgz
    cd ..
  fi
  pip3 install onnxruntime==1.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684X"
  then
    eval_python opencv real_esrgan_fp32_1b.bmodel 69.39919963297913
    eval_python opencv real_esrgan_fp16_1b.bmodel 50.671151256486894
    eval_python opencv real_esrgan_int8_1b.bmodel 36.34181152305623
    eval_python opencv real_esrgan_int8_4b.bmodel 36.34178228546011
    eval_python bmcv real_esrgan_fp32_1b.bmodel 60.06186290140536
    eval_python bmcv real_esrgan_fp16_1b.bmodel 48.97027618435168
    eval_python bmcv real_esrgan_int8_1b.bmodel 36.28425859330598
    eval_python bmcv real_esrgan_int8_4b.bmodel 36.28404703197349

    eval_cpp soc bmcv real_esrgan_fp32_1b.bmodel 54.39043047276539
    eval_cpp soc bmcv real_esrgan_fp16_1b.bmodel 43.42556432129197
    eval_cpp soc bmcv real_esrgan_int8_1b.bmodel 34.97213125435497
    eval_cpp soc bmcv real_esrgan_int8_4b.bmodel 34.97247218881823


  elif test $TARGET = "CV186X"
  then
    eval_python opencv real_esrgan_fp32_1b.bmodel 38.21234165784764
    eval_python opencv real_esrgan_fp16_1b.bmodel 38.19603977835498
    eval_python opencv real_esrgan_int8_1b.bmodel 34.8606226865489
    eval_python opencv real_esrgan_int8_4b.bmodel 34.86062277835418
    eval_python bmcv real_esrgan_fp32_1b.bmodel 38.011214928712
    eval_python bmcv real_esrgan_fp16_1b.bmodel 38.016766744559426 
    eval_python bmcv real_esrgan_int8_1b.bmodel 35.27665138547572
    eval_python bmcv real_esrgan_int8_4b.bmodel 35.27665138547572
 
    eval_cpp soc bmcv real_esrgan_fp32_1b.bmodel 36.866299903695754
    eval_cpp soc bmcv real_esrgan_fp16_1b.bmodel 36.87021558528665
    eval_cpp soc bmcv real_esrgan_int8_1b.bmodel 34.204974878521384
    eval_cpp soc bmcv real_esrgan_int8_4b.bmodel 34.204992390796136

  elif test $TARGET = "BM1688"
  then
    eval_python opencv real_esrgan_fp32_1b.bmodel 38.69009343067908
    eval_python opencv real_esrgan_fp16_1b.bmodel 38.687814169107746
    eval_python opencv real_esrgan_int8_1b.bmodel 35.00328975700205
    eval_python opencv real_esrgan_int8_4b.bmodel 35.00329141383019
    eval_python bmcv real_esrgan_fp32_1b.bmodel 38.50490162494463
    eval_python bmcv real_esrgan_fp16_1b.bmodel 38.510411695441285
    eval_python bmcv real_esrgan_int8_1b.bmodel 34.947925792681346
    eval_python bmcv real_esrgan_int8_4b.bmodel 34.947907359621304
    eval_cpp soc bmcv real_esrgan_fp32_1b.bmodel 36.66300072451568
    eval_cpp soc bmcv real_esrgan_fp16_1b.bmodel 36.66722802560805
    eval_cpp soc bmcv real_esrgan_int8_1b.bmodel 34.15577607485862
    eval_cpp soc bmcv real_esrgan_int8_4b.bmodel 34.15580664236844

    eval_python opencv real_esrgan_fp32_1b_2core.bmodel 38.69009557946511
    eval_python opencv real_esrgan_fp16_1b_2core.bmodel 38.687814169107746
    eval_python opencv real_esrgan_int8_1b_2core.bmodel 35.00328975700205
    eval_python opencv real_esrgan_int8_4b_2core.bmodel 35.00329141383019
    eval_python bmcv real_esrgan_fp32_1b_2core.bmodel 38.50490162494463 
    eval_python bmcv real_esrgan_fp16_1b_2core.bmodel 38.510411695441285
    eval_python bmcv real_esrgan_int8_1b_2core.bmodel 34.947925792681346 
    eval_python bmcv real_esrgan_int8_4b_2core.bmodel 34.947907359621304 
    eval_cpp soc bmcv real_esrgan_fp32_1b_2core.bmodel 36.66300072451568
    eval_cpp soc bmcv real_esrgan_fp16_1b_2core.bmodel 36.66722802560805
    eval_cpp soc bmcv real_esrgan_int8_1b_2core.bmodel 34.15577438139404
    eval_cpp soc bmcv real_esrgan_int8_4b_2core.bmodel 34.155846699255996

  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ] 
then
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------real_esrgan performance-----------"
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