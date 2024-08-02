#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684"
MODE="pcie_test"
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
echo "|   测试平台    |      测试程序     |              测试模型               |AP@IoU=0.5:0.95|AP@IoU=0.5|" >> scripts/acc.txt

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
      bmrt_test_case BM1684/ppyoloe_fp32_1b.bmodel
      # bmrt_test_case BM1684/ppyoloe_int8_1b.bmodel
      # bmrt_test_case BM1684/ppyoloe_int8_4b.bmodel
    elif test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/ppyoloe_fp32_1b.bmodel
      bmrt_test_case BM1684X/ppyoloe_fp16_1b.bmodel
      # bmrt_test_case BM1684X/ppyoloe_int8_1b.bmodel
      # bmrt_test_case BM1684X/ppyoloe_int8_4b.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/ppyoloe_fp32_1b.bmodel
      bmrt_test_case BM1688/ppyoloe_fp16_1b.bmodel
      # bmrt_test_case BM1688/ppyoloe_int8_1b.bmodel
      # bmrt_test_case BM1688/ppyoloe_int8_4b.bmodel
      if test "$PLATFORM" = "SE9-16"; then 
        bmrt_test_case BM1688/ppyoloe_fp32_1b_2core.bmodel
        bmrt_test_case BM1688/ppyoloe_fp16_1b_2core.bmodel
        # bmrt_test_case BM1688/ppyoloe_int8_1b_2core.bmodel
        # bmrt_test_case BM1688/ppyoloe_int8_4b_2core.bmodel
      fi
    elif test $TARGET = "CV186X"; then
      bmrt_test_case CV186X/ppyoloe_fp32_1b.bmodel
      bmrt_test_case CV186X/ppyoloe_fp16_1b.bmodel
      # bmrt_test_case CV186X/ppyoloe_int8_1b.bmodel
      # bmrt_test_case CV186X/ppyoloe_int8_4b.bmodel
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
  chmod -R +x scripts/
  ./scripts/download.sh
  judge_ret $? "download" 0
}

function compile_mlir()
{
  ./scripts/gen_fp32bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  # ./scripts/gen_int8bmodel_mlir.sh $TARGET
  # judge_ret $? "generate $TARGET int8bmodel" 0
  ./scripts/gen_fp16bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp16bmodel" 0
}

function build_pcie()
{
  pushd cpp/ppyoloe_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build ppyoloe_$1" 0
  popd
}

function build_soc()
{
  pushd cpp/ppyoloe_$1
  if [ -d build ]; then
      rm -rf build
  fi
  if test $1 = "sail"; then
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK -DSAIL_PATH=$SAIL_PATH && make
    judge_ret $? "build soc ppyoloe_$1" 0
  else
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
    judge_ret $? "build soc ppyoloe_$1" 0
  fi
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
  pushd cpp/ppyoloe_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./ppyoloe_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_cpp_test.log
  judge_ret $? "./ppyoloe_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID" log/$1_$2_$3_cpp_test.log
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/ppyoloe_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./ppyoloe_$2.$1 --input=../../datasets/coco/val2017_1000 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.4 --nms_thresh=0.6 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./ppyoloe_$2.$1 --input=../../datasets/coco/val2017_1000 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.4 --nms_thresh=0.6 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1" log/$1_$2_$3_debug.log
  tail -n 15 log/$1_$2_$3_debug.log
    
  echo "==================="
  echo "Comparing statis..."
  python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=ppyoloe_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3
  judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=ppyoloe_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 ../../tools/eval_coco.py --gt_path ../../datasets/coco/instances_val2017_1000.json --result_json results/$3_val2017_1000_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"

  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  judge_ret $? "$3_val2017_1000_$2_cpp_result: Precision compare!" log/$1_$2_$3_eval.log

  ap0=$(echo -e "$res"| grep "Average Precision  (AP) @\[ IoU\=0.50:0.95 | area\=   all | maxDets\=100 \]" | grep -oP ' = \K\d+\.\d+' | awk '{printf "%.3f \n", $1}')
  ap1=$(echo -e "$res"| grep "Average Precision  (AP) @\[ IoU\=0.50      | area\=   all | maxDets\=100 \]" | grep -oP ' = \K\d+\.\d+' | awk '{printf "%.3f \n", $1}')
  printf "| %-12s | %-18s | %-30s | %8.3f | %8.3f |\n" "$PLATFORM" "ppyoloe_$2.$1" "$3" "$(printf "%.3f" $ap0)" "$(printf "%.3f" $ap1)">> ../../scripts/acc.txt

  echo -e "########################\nCase End: eval cpp\n########################\n"
  popd
}

function test_python()
{
  if [ ! -d log ];then
    mkdir log
  fi
  python3 python/ppyoloe_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_python_test.log
  judge_ret $? "python3 python/ppyoloe_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID" log/$1_$2_python_test.log
}

function eval_python()
{
  echo -e "\n########################\nCase Start: eval python\n########################"
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  python3 python/ppyoloe_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.4 --nms_thresh 0.6 > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/ppyoloe_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.4 --nms_thresh 0.6 > python/log/$1_$2_debug.log 2>&1" python/log/$1_$2_debug.log
  tail -n 20 python/log/$1_$2_debug.log

  echo "==================="
  echo "Comparing statis..."
  python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=ppyoloe_$1.py --language=python --input=python/log/$1_$2_debug.log --bmodel=$2
  judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=ppyoloe_$1.py --language=python --input=python/log/$1_$2_debug.log --bmodel=$2"
  echo "==================="

  echo "Evaluating..."
  res=$(python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/$2_val2017_1000_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
  judge_ret $? "$2_val2017_1000_$1_python_result: Precision compare!" python/log/$1_$2_eval.log

  ap0=$(echo -e "$res"| grep "Average Precision  (AP) @\[ IoU\=0.50:0.95 | area\=   all | maxDets\=100 \]" | grep -oP ' = \K\d+\.\d+' | awk '{printf "%.3f \n", $1}')
  ap1=$(echo -e "$res"| grep "Average Precision  (AP) @\[ IoU\=0.50      | area\=   all | maxDets\=100 \]" | grep -oP ' = \K\d+\.\d+' | awk '{printf "%.3f \n", $1}')
  printf "| %-12s | %-18s | %-30s | %8.3f | %8.3f |\n" "$PLATFORM" "ppyoloe_$1.py" "$2" "$(printf "%.3f" $ap0)" "$(printf "%.3f" $ap1)">> scripts/acc.txt
  
  echo -e "########################\nCase End: eval python\n########################\n"
}

if test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_test"
then
  build_pcie bmcv
  build_pcie sail
  pip3 install pycocotools opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  download
  if test $TARGET = "BM1684"
  then
    test_python opencv ppyoloe_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv ppyoloe_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv ppyoloe_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail ppyoloe_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv ppyoloe_fp32_1b.bmodel 0.3774908679349741
    eval_python bmcv ppyoloe_fp32_1b.bmodel 0.38058603692128434
    eval_cpp pcie bmcv ppyoloe_fp32_1b.bmodel 0.37836229565532076
    eval_cpp pcie sail ppyoloe_fp32_1b.bmodel 0.37836229565532076

  elif test $TARGET = "BM1684X"
  then
    test_python opencv ppyoloe_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv ppyoloe_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv ppyoloe_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail ppyoloe_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv ppyoloe_fp32_1b.bmodel 0.3774907636324529
    eval_python opencv ppyoloe_fp16_1b.bmodel 0.37712423871409745
    eval_python bmcv ppyoloe_fp32_1b.bmodel 0.38047662857892256
    eval_python bmcv ppyoloe_fp16_1b.bmodel 0.3799782017592526
    eval_cpp pcie bmcv ppyoloe_fp32_1b.bmodel 0.3785768470377686
    eval_cpp pcie bmcv ppyoloe_fp16_1b.bmodel 0.3779939801816987
    eval_cpp pcie sail ppyoloe_fp32_1b.bmodel 0.3785768470377686
    eval_cpp pcie sail ppyoloe_fp16_1b.bmodel 0.3779939801816987
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
  build_soc sail
elif test $MODE = "soc_test"
then
  download
  pip3 install pycocotools opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684"
  then
    test_python opencv ppyoloe_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv ppyoloe_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv ppyoloe_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail ppyoloe_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv ppyoloe_fp32_1b.bmodel 0.3774908679349741
    eval_python bmcv ppyoloe_fp32_1b.bmodel 0.38058603692128434
    eval_cpp soc bmcv ppyoloe_fp32_1b.bmodel 0.37836229565532076
    eval_cpp soc sail ppyoloe_fp32_1b.bmodel 0.37836229565532076
  elif test $TARGET = "BM1684X"
  then
    test_python opencv ppyoloe_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv ppyoloe_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv ppyoloe_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail ppyoloe_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv ppyoloe_fp32_1b.bmodel 0.3774907636324529
    eval_python opencv ppyoloe_fp16_1b.bmodel 0.37712423871409745
    eval_python bmcv ppyoloe_fp32_1b.bmodel 0.38047662857892256
    eval_python bmcv ppyoloe_fp16_1b.bmodel 0.3799782017592526
    eval_cpp soc bmcv ppyoloe_fp32_1b.bmodel 0.3785768470377686
    eval_cpp soc bmcv ppyoloe_fp16_1b.bmodel 0.3779939801816987
    eval_cpp soc sail ppyoloe_fp32_1b.bmodel 0.3785768470377686
    eval_cpp soc sail ppyoloe_fp16_1b.bmodel 0.3779939801816987
  elif [ "$TARGET" = "BM1688" ] || [ "$TARGET" = "CV186X" ]
  then
    test_python opencv ppyoloe_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv ppyoloe_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv ppyoloe_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail ppyoloe_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4

    eval_python opencv ppyoloe_fp32_1b.bmodel 0.377
    eval_python opencv ppyoloe_fp16_1b.bmodel 0.376
    eval_python bmcv   ppyoloe_fp32_1b.bmodel 0.381
    eval_python bmcv   ppyoloe_fp16_1b.bmodel 0.380
    eval_cpp soc bmcv  ppyoloe_fp32_1b.bmodel 0.379
    eval_cpp soc bmcv  ppyoloe_fp16_1b.bmodel 0.378
    eval_cpp soc sail  ppyoloe_fp32_1b.bmodel 0.379
    eval_cpp soc sail  ppyoloe_fp16_1b.bmodel 0.378
    if test "$PLATFORM" = "SE9-16"; then 
      eval_python opencv ppyoloe_fp32_1b_2core.bmodel 0.377
      eval_python opencv ppyoloe_fp16_1b_2core.bmodel 0.376
      eval_python bmcv   ppyoloe_fp32_1b_2core.bmodel 0.381
      eval_python bmcv   ppyoloe_fp16_1b_2core.bmodel 0.380
      eval_cpp soc bmcv  ppyoloe_fp32_1b_2core.bmodel 0.379
      eval_cpp soc bmcv  ppyoloe_fp16_1b_2core.bmodel 0.378
      eval_cpp soc sail  ppyoloe_fp32_1b_2core.bmodel 0.379
      eval_cpp soc sail  ppyoloe_fp16_1b_2core.bmodel 0.378
    fi
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------ppyoloe mAP----------"
  cat scripts/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------ppyoloe performance-----------"
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