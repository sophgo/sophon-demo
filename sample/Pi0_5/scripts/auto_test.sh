#!/bin/bash
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# auto_test.sh — build and exercise the Pi0_5 example on a BM1684X SoC device.
#
# Prerequisites (this sample cannot run without them):
#   1. The six bmodels under models/BM1684X/   (scripts/download.sh)
#   2. The dataset under datasets/pi05_libero_sample/  (scripts/download.sh)
#   3. Run on the device, or with the SoC SDK available for cross compilation.
#
# Usage: ./auto_test.sh [ -m soc_build|soc_test ] [ -t BM1684X ] [ -s SOCSDK ] [ -d TPUID ]

scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/..
pushd $top_dir

TARGET="BM1684X"
MODE="soc_test"
SOCSDK=""
TPUID=0
ALL_PASS=1
NUM_STEPS=2
TASK_ID=0
DATA_DIR="datasets/pi05_libero_sample"
OBS="$DATA_DIR/obs/t00_init0"

usage() {
  echo "Usage: $0 [ -m soc_build|soc_test ] [ -t BM1684X ] [ -s SOCSDK ] [ -d TPUID ]" 1>&2
  echo "  -m  mode, default soc_test" 1>&2
  echo "  -t  target chip, only BM1684X is supported" 1>&2
  echo "  -s  SoC SDK path, required by soc_build" 1>&2
  echo "  -d  TPU device id, default 0" 1>&2
}

while getopts ":m:t:s:d:" opt; do
  case $opt in
    m) MODE=${OPTARG}; echo "mode is $MODE";;
    t) TARGET=${OPTARG}; echo "target is $TARGET";;
    s) SOCSDK=${OPTARG}; echo "soc-sdk is $SOCSDK";;
    d) TPUID=${OPTARG}; echo "using tpu $TPUID";;
    ?) usage; exit 1;;
  esac
done

if [ "$TARGET" != "BM1684X" ]; then
  echo "ERROR: this sample only supports BM1684X (SE7 series), got $TARGET"
  exit 1
fi

PLATFORM="SE7-32"
MODEL_DIR="models/BM1684X"
BMODELS="pi05_siglip_w8bf16_2b pi05_dkv0_9_w8bf16_1b pi05_dkv9_18_w8bf16_1b \
         pi05_ddn0_6_bf16_1b pi05_ddn6_12_bf16_1b pi05_ddn12_18_bf16_1b"

# bmrt_test ships with libsophon, but the device only puts its bin/ on PATH from a login
# shell -- a non-interactive run (ssh host '<cmd>', CI) would not find it. Resolve it
# explicitly so the timing stage does not silently degrade to "not found".
BMRT_TEST="$(command -v bmrt_test 2>/dev/null || true)"
if [ -z "$BMRT_TEST" ] && [ -n "$SOCSDK" ] && [ -x "$SOCSDK/bin/bmrt_test" ]; then
  BMRT_TEST="$SOCSDK/bin/bmrt_test"
fi
if [ -z "$BMRT_TEST" ] && [ -x /opt/sophon/libsophon-current/bin/bmrt_test ]; then
  BMRT_TEST=/opt/sophon/libsophon-current/bin/bmrt_test
fi

# Verifies that every bmodel and the dataset are present before doing any work.
check_assets() {
  local missing=0
  for m in $BMODELS; do
    if [ ! -f "$MODEL_DIR/$m.bmodel" ]; then
      echo "MISSING: $MODEL_DIR/$m.bmodel"
      missing=1
    fi
  done
  if [ ! -d "$OBS" ]; then
    echo "MISSING: $OBS"
    missing=1
  fi
  if [ $missing -ne 0 ]; then
    echo "Run scripts/download.sh bm1684x first."
    return 1
  fi
  return 0
}

# Prints the theoretical latency of every bmodel, without pre/post processing.
# The value of interest is bmrt_test's "calculate time".
bmrt_test_case() {
  echo "$1"
  if [ -z "$BMRT_TEST" ]; then
    echo "  (bmrt_test not found -- pass -s <libsophon sdk> or put its bin/ on PATH)"
    return
  fi
  "$BMRT_TEST" --bmodel "$MODEL_DIR/$1.bmodel" --devid "$TPUID" 2>&1 \
    | grep -E "calculate time|calculate  time" || echo "  (bmrt_test produced no timing line)"
}

do_soc_build() {
  if [ -z "$SOCSDK" ]; then
    echo "ERROR: -s <soc-sdk path> is required for soc_build"
    exit 1
  fi
  bash "$scripts_dir/build.sh" "$SOCSDK" || exit 1
}

do_soc_test() {
  check_assets || exit 1

  if [ ! -x "cpp/pi05_bmcv/pi05_bmcv.soc" ]; then
    echo "ERROR: cpp/pi05_bmcv/pi05_bmcv.soc not found; run -m soc_build first"
    exit 1
  fi

  # 1) theoretical latency
  echo "=== bmrt_test ===" | tee scripts/perf.txt
  for m in $BMODELS; do
    bmrt_test_case "$m" | tee -a scripts/perf.txt
  done

  # 2) end-to-end accuracy over the sample dataset
  echo "=== accuracy ==="
  mkdir -p results
  for d in datasets/pi05_libero_sample/obs/*/; do
    name=$(basename "$d")
    # Task index is encoded in the observation directory name, e.g. t03_init1.
    task_id=$(echo "$name" | sed -n 's/^t\([0-9]\{2\}\).*/\1/p')
    task_id=${task_id:-$TASK_ID}
    ./cpp/pi05_bmcv/pi05_bmcv.soc \
      --bmodel_dir "$MODEL_DIR" \
      --data_dir "$DATA_DIR" \
      --input "$d" \
      --task_id "$((10#$task_id))" \
      --num_steps "$NUM_STEPS" \
      --noise "$DATA_DIR/noise/$name.npy" \
      --dev_id "$TPUID" \
      --output "results/$name.npy" || ALL_PASS=0
  done

  # 3) compare against the official reference
  python3 tools/compare_acc.py \
    --pred_dir results \
    --gt_dir "$DATA_DIR/actions_gt" | tee scripts/acc.txt
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    ALL_PASS=0
  fi

  if [ $ALL_PASS -eq 1 ]; then
    echo "ALL TESTS PASSED"
  else
    echo "SOME TESTS FAILED"
  fi
}

case $MODE in
  soc_build) do_soc_build;;
  soc_test)  do_soc_test;;
  *) usage; exit 1;;
esac

popd
exit $((1 - ALL_PASS))
