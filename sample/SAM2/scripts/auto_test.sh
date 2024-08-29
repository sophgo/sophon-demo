#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1688"
MODE="soc_test"
TPUID=0
ALL_PASS=1
PYTEST="auto_test"
ECHO_LINES=20
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
CASE_MODE="fully"

usage()
{
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_test|soc_test] [ -t TARGET BM1684X|BM1688|CV186X] [ -s SOCSDK] [-a SAIL] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2
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
echo "|   测试平台    |            测试程序            |    测试模型        | mIoU |" >> scripts/acc.txt
PLATFORM=$TARGET
if test $MODE = "soc_test"; then
  if test $TARGET = "BM1684X"; then
    PLATFORM="SE7-32"
  elif test $TARGET = "BM1684"; then
    PLATFORM="SE5-16"
  elif test $TARGET = "BM1688"; then
    PLATFORM="SE9-16"
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
     printf "| %-60s| % 15s |\n" "$1" "$time"
   done
}

function bmrt_test_benchmark(){
    pushd models
    printf "| %-60s| % 15s |\n" "测试模型" "calculate time(ms)"
    printf "| %-60s| % 15s |\n" "------------------------------------------------------------" "--------------"
   
    bmrt_test_case BM1688/image_encoder/sam2_encoder_f16_1b_1core.bmodel
    bmrt_test_case BM1688/image_encoder/sam2_encoder_f16_1b_2core.bmodel
    bmrt_test_case BM1688/image_encoder/sam2_encoder_f32_1b_1core.bmodel
    bmrt_test_case BM1688/image_encoder/sam2_encoder_f32_1b_2core.bmodel

    bmrt_test_case BM1688/image_decoder/sam2_decoder_f16_1b_1core.bmodel
    bmrt_test_case BM1688/image_decoder/sam2_decoder_f16_1b_2core.bmodel
    bmrt_test_case BM1688/image_decoder/sam2_decoder_f32_1b_1core.bmodel
    bmrt_test_case BM1688/image_decoder/sam2_decoder_f32_1b_2core.bmodel

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
    if [[ -n "$3" ]];then
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

function test_python() {
	if [[ ! -d log ]]; then
		mkdir log
	fi
	python3 python/sam2_"$1".py --img_path datasets/truck.jpg \
		--points '[[500, 375]]' --label 1 \
		--encoder_bmodel models/"${TARGET}"/image_encoder/"$2".bmodel \
		--decoder_bmodel models/"${TARGET}"/image_decoder/"$3".bmodel \
		>log/"$1"_"$2"_python_test.log 2>&1
	judge_ret $? "python3 python/sam2_$1.py --input_image datasets/truck.jpg --input_point '[[500, 375]]' --input_label 1 \
                            --encoder_bmodel models/${TARGET}/image_encoder/$2.bmodel   --decoder_bmodel models/${TARGET}/image_decoder/$3.bmodel " log/"$1"_"$2"_python_test.log
	tail -n 20 log/"$1"_"$2"_python_test.log
	if test "$4" = "img"; then
		echo "==================="
		echo "Comparing statis..."
		python3 tools/compare_statis.py --target="${TARGET}" --platform="${MODE%_*}" --program=sam2_"$1".py --language=python --input=log/"$1"_"$2"_python_test.log --encoder_bmodel models/${TARGET}/image_encoder/$2.bmodel   --decoder_bmodel models/${TARGET}/image_decoder/$3.bmodel
		judge_ret $? "python3 tools/compare_statis.py --target=${TARGET} --platform=${MODE%_*} --program=sam2_$1.py --language=python --input=log/$1_$2_python_test.log --encoder_bmodel models/${TARGET}/image_encoder/$2.bmodel   --decoder_bmodel models/${TARGET}/image_decoder/$3.bmodel"
		echo "==================="
	fi
}

function eval_python()
{
  echo -e "\n########################\nCase Start: eval python\n########################"
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  python3 python/sam2_$1.py --img_path datasets/val2017 \
                            --encoder_bmodel models/$TARGET/image_encoder/$2.bmodel  \
                            --decoder_bmodel models/$TARGET/image_decoder/$3.bmodel  \
                            --gt_path $6  --mode $4 --dataset_type $5 \
                            --detect_num $7 > python/log/$1_$2_debug.log 2>&1

  judge_ret $? "python3 python/sam2_$1.py --img_path datasets/val2017  \
                                          --encoder_bmodel models/$TARGET/image_encoder/$2.bmodel  \
                                          --decoder_bmodel models/$TARGET/image_decoder/$3.bmodel  \
                                          --gt_path $6 --mode $4 --dataset_type $5 \
                                          --detect_num $7 > python/log/$1_$2_debug.log 2>&1" python/log/$1_$2_debug.log
  tail -n 20 python/log/$1_$2_debug.log

  echo "Evaluating..."
  res=$(python3 tools/eval.py --gt_path datasets/instances_val2017.json --res_path results/$2_$5_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  echo -e "$res"
  miou_value=$(echo "$res" | grep "mIoU" | awk -F'=' '{print $2}')
  compare_res $miou_value $8
  judge_ret $? "$2_$5_$1_python_result: Precision compare!" python/log/$1_$2_eval.log
  printf "| %-12s | %-18s | %-30s | %8.3f|\n" "$PLATFORM" "sam2_$1.py" "$2.bmodel" "$(printf "%.3f" $miou_value)" >> scripts/acc.txt
  echo -e "########################\nCase End: eval python\n########################\n"
}

function compile_mlir()
{
  ./scripts/gen_fp32bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  ./scripts/gen_fp16bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp16bmodel" 0
}

function compare_res(){
    ret=`awk -v x=$2 -v y=$1 'BEGIN{print(x-y<0.2 && y-x<0.2)?1:0}'`
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

if test $MODE = "compile_nntc"
then
  download
  compile_nntc
elif test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "soc_build"
then
  build_soc bmcv
  build_soc sail
elif test $MODE = "soc_test"
then
  download
  if test $TARGET = "BM1688"
  then
    echo ""

    test_python opencv sam2_encoder_f32_1b_1core sam2_decoder_f32_1b_1core img
    test_python opencv sam2_encoder_f32_1b_2core sam2_decoder_f32_1b_2core img
    test_python opencv sam2_encoder_f16_1b_1core sam2_decoder_f16_1b_1core img
    test_python opencv sam2_encoder_f16_1b_2core sam2_decoder_f16_1b_2core img

    #performence test
    eval_python opencv sam2_encoder_f32_1b_1core sam2_decoder_f32_1b_1core dataset COCODataset datasets/instances_val2017.json 200 0.4
    eval_python opencv sam2_encoder_f32_1b_2core sam2_decoder_f32_1b_2core dataset COCODataset datasets/instances_val2017.json 200 0.4
    eval_python opencv sam2_encoder_f16_1b_1core sam2_decoder_f16_1b_1core dataset COCODataset datasets/instances_val2017.json 200 0.4
    eval_python opencv sam2_encoder_f16_1b_2core sam2_decoder_f16_1b_2core dataset COCODataset datasets/instances_val2017.json 200 0.4
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------sam2 mIoU----------"
  cat scripts/acc.txt
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------sam2 performance-----------"
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
