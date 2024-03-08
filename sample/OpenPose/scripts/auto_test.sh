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

usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X] [ -s SOCSDK] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2 
}

while getopts ":m:t:s:d:p:" opt
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


if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi
if [ -f "tools/benchmark_tpu_kernel_opt.txt" ]; then
  rm tools/benchmark_tpu_kernel_opt.txt
fi
if [ -f "tools/benchmark_tpu_kernel_half_img_size_opt.txt" ]; then
  rm tools/benchmark_tpu_kernel_half_img_size_opt.txt
fi
if [ -f "tools/benchmark_cpu_opt.txt" ]; then
  rm tools/benchmark_cpu_opt.txt
fi
if [ -f "scripts/acc.txt" ]; then
  rm scripts/acc.txt
fi
echo "|   测试平台    |      测试程序     |    测试模型        |AP@IoU=0.5:0.95|AP@IoU=0.5|" >> scripts/acc.txt

if [ -f "scripts/acc_tpu_kernel_opt.txt" ]; then
  rm scripts/acc_tpu_kernel_opt.txt
fi
echo "|   测试平台    |      测试程序     |    测试模型        |AP@IoU=0.5:0.95|AP@IoU=0.5|" >> scripts/acc_tpu_kernel_opt.txt

if [ -f "scripts/acc_tpu_kernel_half_img_size_opt.txt" ]; then
  rm scripts/acc_tpu_kernel_half_img_size_opt.txt
fi
echo "|   测试平台    |      测试程序     |    测试模型        |AP@IoU=0.5:0.95|AP@IoU=0.5|" >> scripts/acc_tpu_kernel_half_img_size_opt.txt

if [ -f "scripts/acc_cpu_opt.txt" ]; then
  rm scripts/acc_cpu_opt.txt
fi
echo "|   测试平台    |      测试程序     |    测试模型        |AP@IoU=0.5:0.95|AP@IoU=0.5|" >> scripts/acc_cpu_opt.txt

PLATFORM=$TARGET
if test $MODE = "soc_test"; then
  if test $TARGET = "BM1684X"; then
    PLATFORM="SE7-32"
  elif test $TARGET = "BM1684"; then
    PLATFORM="SE5-16"
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
   
    if test $TARGET = "BM1684"; then
      bmrt_test_case BM1684/pose_coco_fp32_1b.bmodel
      bmrt_test_case BM1684/pose_coco_int8_1b.bmodel
      bmrt_test_case BM1684/pose_coco_int8_4b.bmodel
    elif test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/pose_coco_fp32_1b.bmodel
      bmrt_test_case BM1684X/pose_coco_fp16_1b.bmodel
      bmrt_test_case BM1684X/pose_coco_int8_1b.bmodel
      bmrt_test_case BM1684X/pose_coco_int8_4b.bmodel
    elif test $TARGET = "BM1688"; then
      bmrt_test_case BM1688/pose_coco_fp32_1b.bmodel
      bmrt_test_case BM1688/pose_coco_fp16_1b.bmodel
      bmrt_test_case BM1688/pose_coco_int8_1b.bmodel
      bmrt_test_case BM1688/pose_coco_int8_4b.bmodel
      bmrt_test_case BM1688/pose_coco_fp32_1b_2core.bmodel
      bmrt_test_case BM1688/pose_coco_fp16_1b_2core.bmodel
      bmrt_test_case BM1688/pose_coco_int8_1b_2core.bmodel
      bmrt_test_case BM1688/pose_coco_int8_4b_2core.bmodel
    fi
    popd
}

if test $PYTEST = "pytest"
then
  >${top_dir}auto_test_result.txt
fi

function judge_ret()
{
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

function build_pcie()
{
  pushd cpp/openpose_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build openpose_$1" 0
  popd
}

function build_soc()
{
  pushd cpp/openpose_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
  judge_ret $? "build soc openpose_$1" 0
  popd
}

function test_python()
{
  if [ ! -d log ];then
    mkdir log
  fi
  echo "testing python $1 $2:"
  python3 python/openpose_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_python_test.log 2>&1
  judge_ret $? "python3 python/openpose_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_python_test.log 2>&1" log/$1_$2_python_test.log
  tail -n 22 log/$1_$2_python_test.log

  if test $3 = "datasets/coco/val2017_1000"; then
    echo "==================="
    echo "Comparing statis..."
    python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=openpose_$1.py --language=python --input=log/$1_$2_python_test.log --bmodel=$2
    judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=openpose_$1.py --language=python --input=log/$1_$2_python_test.log --bmodel=$2"
    echo "==================="

    echo "Evaluating..."
    res=$(python3 tools/eval_coco.py --result_json results/$2_val2017_1000_$1_python_result.json --gt_path datasets/coco/person_keypoints_val2017_1000.json 2>&1 | tee log/$1_$2_python_eval.log)
    echo -e "$res"
    array=(${res//=/ })
    acc=${array[1]}
    compare_res $acc $4
    judge_ret $? "$2 $1 python: $acc vs $4 Precision compare!" log/$1_$2_python_eval.log
    printf "| %-12s | %-14s | %-22s | %8.3f |\n" "$PLATFORM" "openpose_$1.py" "$2" "$(printf "%.3f" $acc)" >> scripts/acc.txt
  fi
}

function test_cpp()
{
  pushd cpp/openpose_$2
  if [ ! -d log ];then
    mkdir log
  fi
  echo "testing cpp $2 $3:"
  chmod +x ./openpose_$2.$1
  ./openpose_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./openpose_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1" log/$1_$2_$3_debug.log
  tail -n 12 log/$1_$2_$3_debug.log
  if test $4 = "../../datasets/coco/val2017_1000"; then
    echo "==================="
    echo "Comparing statis..."
    python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=openpose_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3
    judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=openpose_$2.$1 --language=cpp --input=log/$1_$2_$3_debug.log --bmodel=$3"
    echo "==================="

    echo "Evaluating..."
    res=$(python3 ../../tools/eval_coco.py --result_json results/$3_val2017_1000_$2_cpp_result.json --gt_path ../../datasets/coco/person_keypoints_val2017_1000.json 2>&1 | tee log/$2_$1_$3_eval.log)
    echo -e "$res"
    array=(${res//=/ })
    acc=${array[1]}
    compare_res $acc $5
    judge_ret $? "$3_openpose_$2.$1_cpp: $acc vs $5 Precision compare!" log/$2_$1_$3_eval.log

    printf "| %-12s | %-14s | %-22s | %8.3f |\n" "$PLATFORM" "openpose_$2.$1" "$3" "$(printf "%.3f" $acc)" >> ../../scripts/acc.txt
  fi
  popd
}

function test_cpp_opt()
{
  pushd cpp/openpose_$2
  if [ ! -d log ];then
    mkdir log
  fi
  echo "testing cpp $2 $3 $6:"
  chmod +x ./openpose_$2.$1
  ./openpose_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID --performance_opt=$6 > log/$1_$2_$3_$6_debug.log 2>&1
  judge_ret $? "./openpose_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID --performance_opt=$6 > log/$1_$2_$3_$6_debug.log 2>&1" log/$1_$2_$3_$6_debug.log
  tail -n 12 log/$1_$2_$3_$6_debug.log
  if test $4 = "../../datasets/coco/val2017_1000"; then
    echo "==================="
    echo "Comparing statis..."
    python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=openpose_$2.$1 --language=cpp --input=log/$1_$2_$3_$6_debug.log --bmodel=$3 --performance_opt=$6
    judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=openpose_$2.$1 --language=cpp --input=log/$1_$2_$3_$6_debug.log --bmodel=$3 --performance_opt=$6"
    echo "==================="

    echo "Evaluating..."
    res=$(python3 ../../tools/eval_coco.py --result_json results/$3_val2017_1000_$2_cpp_result.json --gt_path ../../datasets/coco/person_keypoints_val2017_1000.json 2>&1 | tee log/$2_$1_$3_$6_eval.log)
    echo -e "$res"
    array=(${res//=/ })
    acc=${array[1]}
    compare_res $acc $5
    judge_ret $? "$3 openpose_$2.$1 cpp $6: $acc vs $5 Precision compare!" log/$2_$1_$3_$6_eval.log

    printf "| %-12s | %-14s | %-22s | %8.3f |\n" "$PLATFORM" "openpose_$2.$1" "$3" "$(printf "%.3f" $acc)" >> ../../scripts/acc_$6.txt
  fi
  popd
}

function compile_nntc()
{
  ./scripts/gen_fp32bmodel_nntc.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  ./scripts/gen_int8bmodel_nntc.sh $TARGET
  judge_ret $? "generate $TARGET int8bmodel" 0
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

pushd $top_dir

if test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "compile_nntc"
then
  download
  compile_nntc  
elif test $MODE = "pcie_test"
then
  build_pcie bmcv
  download
  pip3 install -r python/requirements.txt
  if test $TARGET = "BM1684"
  then
    test_python opencv pose_coco_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_int8_1b.bmodel datasets/test
    test_python opencv pose_body_25_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_fp32_1b.bmodel datasets/coco/val2017_1000 0.4389763121962809
    test_python opencv pose_coco_int8_1b.bmodel datasets/coco/val2017_1000 0.4292836389703862
    test_python opencv pose_coco_int8_4b.bmodel datasets/coco/val2017_1000 0.4292836389703862
    test_cpp pcie bmcv pose_coco_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv pose_coco_int8_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv pose_body_25_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv pose_coco_fp32_1b.bmodel ../../datasets/coco/val2017_1000 0.422054793081309
    test_cpp pcie bmcv pose_coco_int8_1b.bmodel ../../datasets/coco/val2017_1000 0.4072217325729422
    test_cpp pcie bmcv pose_coco_int8_4b.bmodel ../../datasets/coco/val2017_1000 0.4072217325729422
    test_cpp_opt pcie bmcv pose_coco_fp32_1b.bmodel ../../datasets/coco/val2017_1000 0.393 cpu_opt
    test_cpp_opt pcie bmcv pose_coco_int8_1b.bmodel ../../datasets/coco/val2017_1000 0.384 cpu_opt
    test_python opencv pose_coco_fp32_1b.bmodel datasets/dance_1080P.mp4
    test_python opencv pose_coco_int8_1b.bmodel datasets/dance_1080P.mp4
    test_cpp pcie bmcv pose_coco_fp32_1b.bmodel ../../datasets/dance_1080P.mp4
    test_cpp pcie bmcv pose_coco_int8_1b.bmodel ../../datasets/dance_1080P.mp4

  elif test $TARGET = "BM1684X"
  then
    test_python opencv pose_coco_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_fp16_1b.bmodel datasets/test
    test_python opencv pose_coco_int8_4b.bmodel datasets/test
    test_python opencv pose_body_25_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_fp32_1b.bmodel datasets/coco/val2017_1000 0.43869402774722493
    test_python opencv pose_coco_fp16_1b.bmodel datasets/coco/val2017_1000 0.4395831670593007
    test_python opencv pose_coco_int8_1b.bmodel datasets/coco/val2017_1000 0.4373644673570596
    test_python opencv pose_coco_int8_4b.bmodel datasets/coco/val2017_1000 0.4373644673570596
    test_cpp pcie bmcv pose_coco_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv pose_coco_fp16_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv pose_coco_int8_4b.bmodel ../../datasets/test
    test_cpp pcie bmcv pose_body_25_fp32_1b.bmodel ../../datasets/test
    test_cpp pcie bmcv pose_coco_fp32_1b.bmodel ../../datasets/coco/val2017_1000 0.4195446474200322
    test_cpp pcie bmcv pose_coco_fp16_1b.bmodel ../../datasets/coco/val2017_1000 0.41953060024558564	
    test_cpp pcie bmcv pose_coco_int8_1b.bmodel ../../datasets/coco/val2017_1000 0.4182554848276549
    test_cpp pcie bmcv pose_coco_int8_4b.bmodel ../../datasets/coco/val2017_1000 0.4182554848276549
    test_cpp_opt pcie bmcv pose_coco_int8_4b.bmodel ../../datasets/coco/val2017_1000 0.3939004061513843 cpu_opt
    test_cpp_opt pcie bmcv pose_coco_int8_4b.bmodel ../../datasets/coco/val2017_1000 0.3896601083346971 tpu_kernel_half_img_size_opt
    test_cpp_opt pcie bmcv pose_coco_int8_4b.bmodel ../../datasets/coco/val2017_1000 0.4183194980091748 tpu_kernel_opt
    test_python opencv pose_coco_fp32_1b.bmodel datasets/dance_1080P.mp4
    test_python opencv pose_coco_fp16_1b.bmodel datasets/dance_1080P.mp4
    test_python opencv pose_coco_int8_4b.bmodel datasets/dance_1080P.mp4
    test_cpp pcie bmcv pose_coco_fp32_1b.bmodel ../../datasets/dance_1080P.mp4
    test_cpp pcie bmcv pose_coco_fp16_1b.bmodel ../../datasets/dance_1080P.mp4
    test_cpp pcie bmcv pose_coco_int8_4b.bmodel ../../datasets/dance_1080P.mp4
    
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
elif test $MODE = "soc_test"
then
  download
  pip3 install -r python/requirements.txt
  if test $TARGET = "BM1684"
  then
    test_python opencv pose_coco_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_int8_4b.bmodel datasets/test
    test_python opencv pose_body_25_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_fp32_1b.bmodel datasets/coco/val2017_1000 0.4389763121962809
    test_python opencv pose_coco_int8_1b.bmodel datasets/coco/val2017_1000 0.43116749971507407
    test_python opencv pose_coco_int8_4b.bmodel datasets/coco/val2017_1000 0.43116749971507407
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_body_25_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/coco/val2017_1000 0.42203606157860385
    test_cpp soc bmcv pose_coco_int8_1b.bmodel ../../datasets/coco/val2017_1000 0.40742588929917295
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/coco/val2017_1000 0.40742588929917295
    test_cpp_opt soc bmcv pose_coco_int8_4b.bmodel ../../datasets/coco/val2017_1000 0.38402293761174966 cpu_opt
    test_python opencv pose_coco_fp32_1b.bmodel datasets/dance_1080P.mp4
    test_python opencv pose_coco_int8_4b.bmodel datasets/dance_1080P.mp4
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/dance_1080P.mp4
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/dance_1080P.mp4

  elif test $TARGET = "BM1684X"
  then
    test_python opencv pose_coco_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_fp16_1b.bmodel datasets/test
    test_python opencv pose_coco_int8_4b.bmodel datasets/test
    test_python opencv pose_body_25_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_fp32_1b.bmodel datasets/coco/val2017_1000 0.43897649080775064
    test_python opencv pose_coco_fp16_1b.bmodel datasets/coco/val2017_1000 0.43944845994426185
    test_python opencv pose_coco_int8_1b.bmodel datasets/coco/val2017_1000 0.4362598550565294
    test_python opencv pose_coco_int8_4b.bmodel datasets/coco/val2017_1000 0.4362598550565294
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_coco_fp16_1b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_body_25_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/coco/val2017_1000 0.4195446474200322
    test_cpp soc bmcv pose_coco_fp16_1b.bmodel ../../datasets/coco/val2017_1000 0.41953060024558564
    test_cpp soc bmcv pose_coco_int8_1b.bmodel ../../datasets/coco/val2017_1000 0.4182554848276549
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/coco/val2017_1000 0.4182554848276549
    test_cpp_opt soc bmcv pose_coco_int8_4b.bmodel ../../datasets/coco/val2017_1000 0.3939004061513843 cpu_opt
    test_cpp_opt soc bmcv pose_coco_int8_4b.bmodel ../../datasets/coco/val2017_1000 0.3896601083346971 tpu_kernel_half_img_size_opt
    test_cpp_opt soc bmcv pose_coco_int8_4b.bmodel ../../datasets/coco/val2017_1000 0.4183194980091748 tpu_kernel_opt
    test_python opencv pose_coco_fp32_1b.bmodel datasets/dance_1080P.mp4
    test_python opencv pose_coco_fp16_1b.bmodel datasets/dance_1080P.mp4
    test_python opencv pose_coco_int8_4b.bmodel datasets/dance_1080P.mp4
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/dance_1080P.mp4
    test_cpp soc bmcv pose_coco_fp16_1b.bmodel ../../datasets/dance_1080P.mp4
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/dance_1080P.mp4

  elif test $TARGET = "BM1688"
  then
    test_python opencv pose_coco_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_fp16_1b.bmodel datasets/test
    test_python opencv pose_coco_int8_4b.bmodel datasets/test
    test_python opencv pose_body_25_fp32_1b.bmodel datasets/test
    test_python opencv pose_coco_fp32_1b.bmodel datasets/coco/val2017_1000 0.43897649080775064
    test_python opencv pose_coco_fp16_1b.bmodel datasets/coco/val2017_1000 0.4399855779871844
    test_python opencv pose_coco_int8_1b.bmodel datasets/coco/val2017_1000 0.4372115779812914
    test_python opencv pose_coco_int8_4b.bmodel datasets/coco/val2017_1000 0.4372115779812914
    test_python opencv pose_coco_fp32_1b_2core.bmodel datasets/coco/val2017_1000 0.43897649080775064
    test_python opencv pose_coco_fp16_1b_2core.bmodel datasets/coco/val2017_1000 0.4399855779871844
    test_python opencv pose_coco_int8_1b_2core.bmodel datasets/coco/val2017_1000 0.4372115779812914
    test_python opencv pose_coco_int8_4b_2core.bmodel datasets/coco/val2017_1000 0.4372115779812914
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_coco_fp16_1b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_body_25_fp32_1b.bmodel ../../datasets/test
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/coco/val2017_1000 0.4189068800960253
    test_cpp soc bmcv pose_coco_fp16_1b.bmodel ../../datasets/coco/val2017_1000 0.41890082470653983
    test_cpp soc bmcv pose_coco_int8_1b.bmodel ../../datasets/coco/val2017_1000 0.41785336055872097
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/coco/val2017_1000 0.41785336055872097
    test_cpp soc bmcv pose_coco_fp32_1b_2core.bmodel ../../datasets/coco/val2017_1000 0.4189068800960253
    test_cpp soc bmcv pose_coco_fp16_1b_2core.bmodel ../../datasets/coco/val2017_1000 0.41890082470653983
    test_cpp soc bmcv pose_coco_int8_1b_2core.bmodel ../../datasets/coco/val2017_1000 0.41785336055872097
    test_cpp soc bmcv pose_coco_int8_4b_2core.bmodel ../../datasets/coco/val2017_1000 0.41785336055872097
    test_cpp_opt soc bmcv pose_coco_int8_4b_2core.bmodel ../../datasets/coco/val2017_1000 0.3877315447463998 cpu_opt
    test_python opencv pose_coco_fp32_1b.bmodel datasets/dance_1080P.mp4
    test_python opencv pose_coco_fp16_1b.bmodel datasets/dance_1080P.mp4
    test_python opencv pose_coco_int8_4b.bmodel datasets/dance_1080P.mp4
    test_cpp soc bmcv pose_coco_fp32_1b.bmodel ../../datasets/dance_1080P.mp4
    test_cpp soc bmcv pose_coco_fp16_1b.bmodel ../../datasets/dance_1080P.mp4
    test_cpp soc bmcv pose_coco_int8_4b.bmodel ../../datasets/dance_1080P.mp4

  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------openpose mAP----------"
  cat scripts/acc.txt
  echo "--------openpose cpu_opt mAP----------"
  cat scripts/acc_cpu_opt.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------openpose performance-----------"
  cat tools/benchmark.txt
  echo "--------openpose cpu_opt performance-----------"
  cat tools/benchmark_cpu_opt.txt
fi

popd

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