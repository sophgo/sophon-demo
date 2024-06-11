#!/bin/bash

# This script is used to test CLIP on BM1684/BM1684X/BM1688/CV186X platform

# use sophon-opencv
# 设置目录的基本路径和模式
base_path="/opt/sophon"
pattern="sophon-opencv_*"

# 使用find命令来定位正确的目录
# -maxdepth 1 保证不会搜索子目录
opencv_dir=$(find "$base_path" -maxdepth 1 -type d -name "$pattern" -print -quit)

# 检查是否找到了目录
if [[ -d "$opencv_dir" ]]; then
    export PYTHONPATH=$PYTHONPATH:$opencv_dir/opencv-python
    echo "Added $opencv_dir/opencv-python to PYTHONPATH"
else
    echo "Error: OpenCV directory not found."
fi

# install unzip first
res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    echo "To install, use the following command:"
    echo "sudo apt install unzip"
    exit
fi

scripts_dir=$(cd `dirname $BASH_SOURCE[0]`/ && pwd)
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684X"
MODE="pcie_test"
TPUID=0
ALL_PASS=1
PYTEST="auto_test"
ECHO_LINES=20


usage() 
{
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_test|soc_test] [ -t TARGET BM1684X|BM1688|CV186X] [ -s SOCSDK]  [ -d TPUID]  [ -p PYTEST auto_test|pytest]" 1>&2 
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
     printf "| %-35s| % 15s |\n" "$1" "$time"
   done
}

function bmrt_test_benchmark(){
    pushd models
    printf "| %-45s| % 15s |\n" "测试模型" "calculate time(ms)"
    printf "| %-45s| % 15s |\n" "-------------------" "--------------"
   
    if test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/clip_image_vitb32_bm1684x_f16_1b.bmodel
      bmrt_test_case BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/clip_image_vitb32_bm1688_f16_1b.bmodel
      bmrt_test_case BM1688/clip_text_vitb32_bm1688_f16_1b.bmodel
      bmrt_test_case BM1688/clip_image_vitb32_bm1688_f16_1b_2core.bmodel
      bmrt_test_case BM1688/clip_text_vitb32_bm1688_f16_1b_2core.bmodel
      if test "$PLATFORM" = "SE9-16"; then 
        bmrt_test_case BM1688/clip_image_vitb32_bm1688_f32_1b_2core.bmodel
        bmrt_test_case BM1688/clip_text_vitb32_bm1688_f32_1b_2core.bmodel
      fi
    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/clip_image_vitb32_cv186x_f16_1b.bmodel
      bmrt_test_case CV186X/clip_text_vitb32_cv186x_f16_1b.bmodel
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
  ./scripts/gen_fp16bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp16bmodel" 0
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

function test_python()
{
  if [ ! -d log ];then
    mkdir log
  fi
  # 传入的两个参数，1是image bmodel， 2是text bmodel
  python3 python/zeroshot_predict.py --image_model models/$TARGET/$1 --text_model models/$TARGET/$2 --dev_id=$TPUID  > log/$1_$2_python_test.log 2>&1
  judge_ret $? "python/zeroshot_predict.py --image_model models/$TARGET/$1 --text_model models/$TARGET/$2 --dev_id=$TPUID"
  tail -n 20 log/$1_$2_python_test.log

  # 对比速度
  echo "==================="
  echo "Comparing statis..."
  python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=zeroshot_predict.py --language=python --input=log/$1_$2_python_test.log --image_model $1 --text_model $2
  judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=zeroshot_predict.py --language=python --input=log/$1_$2_python_test.log --image_model $1 --text_model $2"
  echo "==================="

}


if test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_test"
then
  download
  # 安装python依赖
  pip3 install torch torchvision regex ftfy opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple 
  if test $TARGET = "BM1684X"
  then
    # python/zeroshot_predict.py运行测试，以及速度对比
    test_python clip_image_vitb32_bm1684x_f16_1b.bmodel clip_text_vitb32_bm1684x_f16_1b.bmodel
  fi
elif test $MODE = "soc_test"
then
  download
  # 安装python依赖
  pip3 install torch torchvision regex ftfy pycocotools opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684X"
  then
    test_python clip_image_vitb32_bm1684x_f16_1b.bmodel clip_text_vitb32_bm1684x_f16_1b.bmodel
  elif test $TARGET = "BM1688" 
  then
    # 测试这两个平台的单core模型
    test_python clip_image_vitb32_bm1688_f16_1b.bmodel clip_text_vitb32_bm1688_f16_1b.bmodel
  elif test $TARGET = "CV186X"
  then
    test_python clip_image_vitb32_cv186x_f16_1b.bmodel clip_text_vitb32_cv186x_f16_1b.bmodel
  fi
fi


if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
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