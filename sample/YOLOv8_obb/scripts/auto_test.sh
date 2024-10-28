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
echo "|   测试平台    |      测试程序     |              测试模型               |  map  |" >> scripts/acc.txt

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
    printf "| %-35s| % 15s |\n" "测试模型" "calculate time(ms)"
    printf "| %-35s| % 15s |\n" "-------------------" "--------------"
   
    if test $TARGET = "BM1684"; then
      bmrt_test_case BM1684/yolov8s-obb_fp32_1b.bmodel
    elif test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/yolov8s-obb_fp32_1b.bmodel
      bmrt_test_case BM1684X/yolov8s-obb_fp16_1b.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/yolov8s-obb_fp32_1b.bmodel
      bmrt_test_case BM1688/yolov8s-obb_fp16_1b.bmodel
      if test "$PLATFORM" = "SE9-16"; then 
        bmrt_test_case BM1688/yolov8s-obb_fp32_1b_2core.bmodel
        bmrt_test_case BM1688/yolov8s-obb_fp16_1b_2core.bmodel
      fi
    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/yolov8s-obb_fp32_1b.bmodel
      bmrt_test_case CV186X/yolov8s-obb_fp16_1b.bmodel
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
    pushd cpp/yolov8_$1
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. && make
    judge_ret $? "build yolov8_$1" 0
    popd
}

function build_soc()
{
    pushd cpp/yolov8_$1
    if [ -d build ]; then
        rm -rf build
    fi
    if test $1 = "sail"; then
      mkdir build && cd build
      cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK -DSAIL_PATH=$SAIL_PATH && make
      judge_ret $? "build soc yolov8_$1" 0
    else
      mkdir build && cd build
      cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
      judge_ret $? "build soc yolov8_$1" 0
    fi
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
        return 1
    else
        echo "***************************************"
        echo -e "\e[42m compare right! \e[0m" #green
        echo "***************************************"
        return 0
    fi
}

function test_cpp()
{
    pushd cpp/yolov8_$2
    if [ ! -d log ];then
      mkdir log
    fi
    ./yolov8_$2.$1 --input=../../datasets/DOTAv1/images/val --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID --conf_thresh=0.25 --nms_thresh=0.7 > log/$1_$2_$3_cpp_test.log 2>&1
    judge_ret $? "./yolov8_$2.$1 --input=../../datasets/DOTAv1/images/val --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID" log/$1_$2_$3_cpp_test.log
    tail -n 15 log/$1_$2_$3_cpp_test.log
    echo "==================="
    echo "Comparing statis..."
    python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$2.$1 --language=cpp --input=log/$1_$2_$3_cpp_test.log --bmodel=$3
    judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$2.$1 --language=cpp --input=log/$1_$2_$3_cpp_test.log --bmodel=$3"
    echo "==================="
    popd

    if test $MODE = "soc_test"; then
        echo "Evaluating..."
        pushd tools/
        python3 eval_DOTA.py --result_json ../cpp/yolov8_$2/results/$3_val_$2_cpp_result.json
        pushd DOTA_devkit_soc/
        res=$(python3 dota_evaluation_task1.py 2>&1 | tee ../../cpp/yolov8_$2/log/$1_$2_$3_eval.log)
        echo -e "$res" | tail -n10
        map=$(echo "$res" |grep "map"| awk '{print $2}')
        compare_res $map $4
        judge_ret $? "Precision compare!" ../../cpp/yolov8_$2/log/$1_$2_$3_eval.log
        printf "| %-12s | %-18s | %-40s | %8.3f |\n" "$PLATFORM" "yolov8_$2.$1" "$3" "$(printf "%.3f" $map)">> ../../scripts/acc.txt
        popd
        popd
    fi
}

function test_python()
{
    if [ ! -d log ];then
      mkdir log
    fi
    python3 python/yolov8_$1.py --input datasets/DOTAv1/images/val --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.25 --nms_thresh 0.7 > log/$1_$2_python_test.log 2>&1
    judge_ret $? "python3 python/yolov8_$1.py --input datasets/DOTAv1/images/val --bmodel models/$TARGET/$2 --dev_id $TPUID" log/$1_$2_python_test.log
    tail -n 20 log/$1_$2_python_test.log
    echo "==================="
    echo "Comparing statis..."
    python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$1.py --language=python --input=log/$1_$2_python_test.log --bmodel=$2
    judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolov8_$1.py --language=python --input=log/$1_$2_python_test.log --bmodel=$2"
    echo "==================="

    if test $MODE = "soc_test"; then
        echo "Evaluating..."
        pushd tools/
        python3 eval_DOTA.py --result_json ../results/$2_val_$1_python_result.json
        pushd DOTA_devkit_soc/
        res=$(python3 dota_evaluation_task1.py 2>&1 | tee ../../log/$1_$2_eval.log)
        echo -e "$res" | tail -n10
        map=$(echo "$res" |grep "map"| awk '{print $2}')
        compare_res $map $3
        judge_ret $? "Precision compare!" ../../log/$1_$2_eval.log
        printf "| %-12s | %-18s | %-40s | %8.3f |\n" "$PLATFORM" "yolov8_$1.py" "$2" "$(printf "%.3f" $map)">> ../../scripts/acc.txt
        popd
        popd
    fi
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
    pip3 install opencv-python-headless torch==2.4.1 torchvision==0.19.1 matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
    if test $TARGET = "BM1684"
    then
        echo "Not support BM1684 yet."
    elif test $TARGET = "BM1684X"
    then
        test_python opencv yolov8s-obb_fp32_1b.bmodel 0.562
        test_python opencv yolov8s-obb_fp16_1b.bmodel 0.562
        test_cpp pcie bmcv yolov8s-obb_fp32_1b.bmodel 0.550
        test_cpp pcie bmcv yolov8s-obb_fp16_1b.bmodel 0.550
    fi
elif test $MODE = "soc_build"
then
    build_soc bmcv
elif test $MODE = "soc_test"
then
    download
    pip3 install opencv-python-headless torch==2.4.1 torchvision==0.19.1 matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
    if test $TARGET = "BM1684"
    then
        echo "Not support BM1684 yet."
    elif test $TARGET = "BM1684X"
    then
        test_python opencv yolov8s-obb_fp32_1b.bmodel 0.562
        test_python opencv yolov8s-obb_fp16_1b.bmodel 0.562
        test_cpp soc bmcv yolov8s-obb_fp32_1b.bmodel  0.550
        test_cpp soc bmcv yolov8s-obb_fp16_1b.bmodel  0.550
    elif [ "$TARGET" = "BM1688" ] || [ "$TARGET" = "CV186X" ]
    then
        test_python opencv yolov8s-obb_fp32_1b.bmodel 0.562
        test_python opencv yolov8s-obb_fp16_1b.bmodel 0.562
        test_cpp soc bmcv yolov8s-obb_fp32_1b.bmodel  0.551
        test_cpp soc bmcv yolov8s-obb_fp16_1b.bmodel  0.551
        if test "$PLATFORM" = "SE9-16"; then
            test_python opencv yolov8s-obb_fp32_1b_2core.bmodel 0.562
            test_python opencv yolov8s-obb_fp16_1b_2core.bmodel 0.562
            test_cpp soc bmcv yolov8s-obb_fp32_1b_2core.bmodel  0.551
            test_cpp soc bmcv yolov8s-obb_fp16_1b_2core.bmodel  0.551
        fi
    fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------yolov8-obb mAP----------"
  cat scripts/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------yolov8-obb performance-----------"
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