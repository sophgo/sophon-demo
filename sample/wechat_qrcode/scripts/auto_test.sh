#!/bin/bash
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# Unified auto test for sample/wechat_qrcode: download -> (optionally compile)
# -> build -> run every variant over the BoofCV QR V4 set for accuracy and over
# qr_small.png for FPS, then compare against fixed baselines.
#
# All four variants are accuracy-checked over the BoofCV QR V4 set (C++ bmcv +
# C++ sail + Python opencv + Python bmcv) and FPS-benchmarked on qr_small.png.
#
# Usage (run at sample/wechat_qrcode):
#   bash scripts/auto_test.sh -m soc_test     -t BM1688
#   bash scripts/auto_test.sh -m compile_mlir -t BM1684X
#   bash scripts/auto_test.sh -m pcie_test    -t BM1684X
# Options: -m MODE, -t TARGET, -s SOCSDK, -a SAIL_PATH, -d TPUID,
#          -p PYTEST(only for review), -c CASE_MODE(fully|partly)
set -uo pipefail

scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

# default config
TARGET="BM1684X"
MODE="pcie_test"
SOCSDK=""
SAIL_PATH=""
TPUID=0
ALL_PASS=1
PYTEST="auto_test"
ECHO_LINES=20
CASE_MODE="fully"

# fixed accuracy tolerances (BoofCV QR V4, IoU>=0.5 detection; see README 5.2).
# Recall/precision are per program (the image-loading path differs, and bmcv is
# a pure port); the exact per-(platform, program) baselines are set below once
# TARGET is finalised. ACC_TOL only absorbs float-formatting noise.
ACC_PRECISION=1.0000
ACC_DECODE=1.0000
ACC_TOL=0.002

usage()
{
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_build|pcie_test|soc_build|soc_test] [ -t TARGET BM1684X|BM1688|CV186X] [ -s SOCSDK] [-a SAIL_PATH] [ -d TPUID] [ -c CASE_MODE fully|partly]" 1>&2
}

while getopts ":m:t:s:a:d:p:c:" opt
do
  case $opt in
    m) MODE=${OPTARG}; echo "mode is $MODE";;
    t) TARGET=${OPTARG}; echo "target is $TARGET";;
    s) SOCSDK=${OPTARG}; echo "soc-sdk is $SOCSDK";;
    a) SAIL_PATH=${OPTARG}; echo "sail_path is $SAIL_PATH";;
    d) TPUID=${OPTARG}; echo "using tpu $TPUID";;
    p) PYTEST=${OPTARG}; echo "generate logs for $PYTEST";;
    c) CASE_MODE=${OPTARG}; echo "case mode is $CASE_MODE";;
    ?) usage; exit 1;;
  esac
done

# TARGET -> lowercase chip / benchmark platform / ld target board
case "$TARGET" in
  BM1684X) CHIP=bm1684x; PLATFORM=SE7-32;;
  BM1688)  CHIP=bm1688;  PLATFORM=SE9-16;;
  CV186X)  CHIP=cv186x;  PLATFORM=SE9-8;;
  *) echo "unknown TARGET: $TARGET" >&2; exit 1;;
esac

# per-(platform, program) accuracy baselines (deterministic; see README 5.2).
# bmcv.soc / sail.soc / opencv.py precision is 1.0000; bmcv.py's sail.Decoder
# path emits a few IoU<0.5 boxes on some platforms, so its precision is lower.
case "$TARGET" in
  BM1684X) # SE7-32
    ACC_RECALL_BMCV=0.4754; ACC_RECALL_SAIL=0.4768; ACC_RECALL_OPENCV=0.4802
    ACC_RECALL_BMCVPY=0.4774; ACC_PRECISION_BMCVPY=0.9843;;
  BM1688|CV186X) # SE9-16 / SE9-8
    ACC_RECALL_BMCV=0.4740; ACC_RECALL_SAIL=0.4768; ACC_RECALL_OPENCV=0.4837
    ACC_RECALL_BMCVPY=0.4879; ACC_PRECISION_BMCVPY=0.9986;;
esac

# compiled C++ binary paths: pcie builds emit <name>.pcie, soc cross-builds emit
# <name>.soc, both under cpp/<variant>/ (CMake EXECUTABLE_OUTPUT_PATH).
case "$MODE" in
  pcie_build|pcie_test)
    BMCV_BIN="cpp/wechat_qrcode_bmcv/wechat_qrcode_bmcv.pcie"
    SAIL_BIN="cpp/wechat_qrcode_sail/wechat_qrcode_sail.pcie";;
  *)
    BMCV_BIN="cpp/wechat_qrcode_bmcv/wechat_qrcode_bmcv.soc"
    SAIL_BIN="cpp/wechat_qrcode_sail/wechat_qrcode_sail.soc";;
esac

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/libsophon-current/lib:/opt/sophon/sophon-opencv-latest/lib

if [ -f "tools/benchmark.txt" ]; then rm tools/benchmark.txt; fi
if [ -f "scripts/acc.txt" ]; then rm scripts/acc.txt; fi
printf "| %-11s | %-24s | %-10s | %-10s | %-12s |\n" "platform" "program" "recall" "precision" "decode_acc" > scripts/acc.txt

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
  else
    echo "Failed: $2"
    ALL_PASS=0
  fi
  if [[ $3 != 0 ]] && [[ $3 != "" ]]; then
    tail -n ${ECHO_LINES} $3
  fi
  sleep 3
}

function compare_res() {
  ret=`awk -v x=$1 -v y=$2 -v t=$ACC_TOL 'BEGIN{print((x-y<t && y-x<t)?1:0)}'`
  if [ $ret -eq 0 ]; then
    ALL_PASS=0
    echo "***************************************"
    echo "Ground truth is $2, your result is: $1"
    echo -e "\e[41m compare wrong! \e[0m"
    echo "***************************************"
    return 1
  else
    echo -e "\e[42m compare right! \e[0m"
    return 0
  fi
}

function download()
{
  chmod -R +x scripts/
  ./scripts/download.sh
  judge_ret $? "download models"
  ./scripts/download_datasets.sh
  judge_ret $? "download datasets"
}

function compile_mlir()
{
  ./scripts/gen_bmodel.sh $CHIP
  judge_ret $? "generate $TARGET bmodel"
}

function build_pcie()
{
  for v in bmcv sail; do
    pushd cpp/wechat_qrcode_$v
    if [ -d build ]; then rm -rf build; fi
    mkdir build && cd build
    # sail's pcie CMake pins SAIL_PATH to /opt/sophon/sophon-sail
    cmake .. && make
    judge_ret $? "build pcie wechat_qrcode_$v"
    popd
  done
}

function build_soc()
{
  # cross-compile via CMake (same convention as PP-OCR / YOLOv5): the sail
  # variant additionally needs -DSAIL_PATH pointing at the full libsail.so.
  for v in bmcv sail; do
    pushd cpp/wechat_qrcode_$v
    if [ -d build ]; then rm -rf build; fi
    mkdir build && cd build
    if [ "$v" = "sail" ]; then
      cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK -DSAIL_PATH=$SAIL_PATH && make
    else
      cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
    fi
    judge_ret $? "build soc wechat_qrcode_$v"
    popd
  done
}

# run one variant over the dataset and evaluate detection + decoding accuracy
function run_accuracy()
{
  # $1 = command line to produce a result JSON; $2 = result json path
  local cmd="$1" json="$2"
  rm -f "$json"
  eval "$cmd" > log.txt 2>&1
  judge_ret $? "run accuracy ($2)" log.txt
  python3 tools/eval_qrcode.py --gt_path datasets/BoofCV_qrcode_v4/qrcodes \
      --result_json "$json" > eval.txt 2>&1
  judge_ret $? "eval_qrcode.py ($2)" eval.txt
}

# accuracy for one C++ variant ($1 = binary, $2 = label, $3 = recall baseline);
# both binaries dump to the same fixed results/wechat_qrcode_results.json, which
# run_accuracy removes before each run, so sequential invocation is safe.
function eval_cpp_bin()
{
  local bin="$1" label="$2" brec="$3"
  echo -e "\n########################\nCase Start: eval cpp $label\n########################"
  local json=results/wechat_qrcode_results.json
  run_accuracy "$bin models/$TARGET/detect_f32_fused.bmodel models/$TARGET/sr_f16_fused.bmodel datasets/BoofCV_qrcode_v4/qrcodes $TPUID 0" "$json"
  local recall=$(grep -oE "recall    : [0-9.]+" eval.txt | grep -oE "[0-9.]+")
  local precision=$(grep -oE "precision : [0-9.]+" eval.txt | grep -oE "[0-9.]+")
  local dacc=$(grep -oE "accuracy: [0-9.]+" eval.txt | grep -oE "[0-9.]+")
  compare_res "$recall" "$brec"; judge_ret $? "wechat_qrcode_$label detection recall"
  compare_res "$precision" "$ACC_PRECISION"; judge_ret $? "wechat_qrcode_$label detection precision"
  compare_res "$dacc" "$ACC_DECODE"; judge_ret $? "wechat_qrcode_$label decode accuracy"
  printf "| %-11s | %-24s | %-10s | %-10s | %-12s |\n" "$PLATFORM" "wechat_qrcode_$label" "$recall" "$precision" "$dacc" >> scripts/acc.txt
  echo -e "########################\nCase End: eval cpp $label\n########################"
}

function eval_cpp()
{
  eval_cpp_bin "$BMCV_BIN" "bmcv" "$ACC_RECALL_BMCV"
  eval_cpp_bin "$SAIL_BIN" "sail" "$ACC_RECALL_SAIL"
}

function eval_python_acc()
{
  local tag=$1   # opencv | bmcv
  echo -e "\n########################\nCase Start: eval python $tag\n########################"
  pushd python
  python3 wechat_qrcode_$tag.py --detect ../models/$TARGET/detect_f32_fused.bmodel \
      --sr ../models/$TARGET/sr_f16_fused.bmodel \
      --input ../datasets/BoofCV_qrcode_v4/qrcodes --dev_id $TPUID > log.txt 2>&1
  judge_ret $? "python $tag accuracy" log.txt
  local json=$(ls -t results/*_${tag}_python_result.json 2>/dev/null | head -1)
  python3 ../tools/eval_qrcode.py --gt_path ../datasets/BoofCV_qrcode_v4/qrcodes \
      --result_json "$json" > eval.txt 2>&1
  judge_ret $? "python $tag eval_qrcode.py" eval.txt
  local recall=$(grep -oE "recall    : [0-9.]+" eval.txt | grep -oE "[0-9.]+")
  local precision=$(grep -oE "precision : [0-9.]+" eval.txt | grep -oE "[0-9.]+")
  local dacc=$(grep -oE "accuracy: [0-9.]+" eval.txt | grep -oE "[0-9.]+")
  local brec="$ACC_RECALL_OPENCV" bprec="$ACC_PRECISION"
  if [ "$tag" = "bmcv" ]; then brec="$ACC_RECALL_BMCVPY"; bprec="$ACC_PRECISION_BMCVPY"; fi
  compare_res "$recall" "$brec"; judge_ret $? "python $tag detection recall"
  compare_res "$precision" "$bprec"; judge_ret $? "python $tag detection precision"
  compare_res "$dacc" "$ACC_DECODE"; judge_ret $? "python $tag decode accuracy"
  printf "| %-11s | %-24s | %-10s | %-10s | %-12s |\n" "$PLATFORM" "wechat_qrcode_$tag.py" "$recall" "$precision" "$dacc" >> ../scripts/acc.txt
  popd
  echo -e "########################\nCase End: eval python $tag\n########################"
}

# bench one variant (C++ or Python) and compare e2e FPS against the baseline
function bench_one()
{
  # $1 = program label, $2 = command producing the bench block, $3 = log file,
  # $4 = language (cpp|python) for compare_statis.py
  local label="$1" cmd="$2" log="$3" lang="$4"
  eval "$cmd" > "$log" 2>&1
  judge_ret $? "bench $label" "$log"
  python3 tools/compare_statis.py --target=$TARGET --platform=$PLATFORM \
      --program="$label" --language="$lang" --input="$log"
  judge_ret $? "compare_statis.py $label"
}

function bench_all()
{
  echo -e "\n########## Bench all four variants ##########"
  local detect=models/$TARGET/detect_f32_fused.bmodel
  local sr=models/$TARGET/sr_f16_fused.bmodel
  local img=images/qr_small.png
  bench_one "wechat_qrcode_bmcv" "$BMCV_BIN $detect $sr $img $TPUID 50" "bmcv_bench.log" cpp
  bench_one "wechat_qrcode_sail" "$SAIL_BIN $detect $sr $img $TPUID 50" "sail_bench.log" cpp
  bench_one "wechat_qrcode_opencv.py" "python3 python/wechat_qrcode_opencv.py --detect $detect --sr $sr --input $img --dev_id $TPUID --iters 50" "opencv_bench.log" python
  bench_one "wechat_qrcode_bmcv.py" "python3 python/wechat_qrcode_bmcv.py --detect $detect --sr $sr --input $img --dev_id $TPUID --iters 50" "bmcvpy_bench.log" python
}

function bmrt_test_case()
{
  # $1 = model path, $2 = bmodel label
  local log=$(/opt/sophon/libsophon-current/bin/bmrt_test --bmodel "$1" --devid $TPUID 2>&1 | grep "calculate")
  local t_flt=$(echo "$log" | grep -oE "calculate[ ]+time\(s\):[ ]+[0-9.]+" | head -1 | grep -oE "[0-9.]+$")
  local t_ms=$(awk -v x=$t_flt 'BEGIN{printf "%.2f", x*1000}')
  printf "| %-15s | %-30s| % 7s | % 8s |\n" "$PLATFORM" "$2" "stage0" "$t_ms" >> tools/benchmark.txt
}

function bmrt_test_benchmark()
{
  printf "| %-15s | %-30s| % 7s | % 8s |\n" "$PLATFORM" "测试模型" "stage" "calculate time(ms)" > tools/benchmark.txt
  bmrt_test_case models/$TARGET/detect_f32_fused.bmodel "detect_f32_fused.bmodel"
  bmrt_test_case models/$TARGET/sr_f16_fused.bmodel "sr_f16_fused.bmodel"
}

if test $MODE = "compile_mlir"; then
  download
  compile_mlir
elif test $MODE = "pcie_build"; then
  build_pcie
elif test $MODE = "soc_build"; then
  build_soc
elif test $MODE = "pcie_test"; then
  pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  download
  build_pcie
  eval_cpp
  if test $CASE_MODE = "fully"; then
    eval_python_acc opencv
    eval_python_acc bmcv
  fi
  bench_all
elif test $MODE = "soc_test"; then
  pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  download
  build_soc
  eval_cpp
  if test $CASE_MODE = "fully"; then
    eval_python_acc opencv
    eval_python_acc bmcv
  fi
  bench_all
fi

if [ x$MODE = x"pcie_test" ] || [ x$MODE = x"soc_test" ]; then
  echo "-------- wechat_qrcode accuracy ----------"
  cat scripts/acc.txt
  echo "-------- bmrt_test performance -----------"
  bmrt_test_benchmark
  cat tools/benchmark.txt
fi

if [[ $ALL_PASS -eq 0 ]]; then
  echo "===================================================================="
  echo "Some process produced unexpected results, please look out their logs!"
  echo "===================================================================="
else
  echo "===================="
  echo "Test cases all pass!"
  echo "===================="
fi

popd