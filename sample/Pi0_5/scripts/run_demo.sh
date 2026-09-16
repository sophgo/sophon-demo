#!/bin/bash
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# run_demo.sh — build and run one pi0.5 inference, end to end.
#
# Intended for a first-time user: it checks every prerequisite up front and prints
# exactly what is missing, rather than failing somewhere deep in the run.
#
# Run it ON the target device (SE7 / BM1684X). The device ships g++ and libsophon,
# so no cross toolchain is needed:
#
#     cd sample/Pi0_5
#     ./scripts/run_demo.sh
#
# Options:
#     -t TASK_ID   task index 0-9 (default 0)
#     -n STEPS     denoise steps (default 2)
#     -d DEV_ID    TPU device id (default 0)
#     -l LOOPS     repeat inference N times for timing (default 1)
#     -S SDK       libsophon path (default /opt/sophon/libsophon-current)

set -u

scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$(readlink -f "$scripts_dir/..")
cd "$top_dir"

TASK_ID=0
STEPS=2
DEV_ID=0
LOOPS=1
SDK="${SOC_SDK:-/opt/sophon/libsophon-current}"

usage() {
    sed -n '10,27p' "$0" | sed 's/^# \{0,1\}//'
}

while getopts ":t:n:d:l:S:h" opt; do
    case $opt in
        t) TASK_ID=${OPTARG};;
        n) STEPS=${OPTARG};;
        d) DEV_ID=${OPTARG};;
        l) LOOPS=${OPTARG};;
        S) SDK=${OPTARG};;
        h) usage; exit 0;;
        ?) usage; exit 1;;
    esac
done

MODEL_DIR="models/BM1684X"
DATA_DIR="datasets/pi05_libero_sample"
OBS="${DATA_DIR}/obs/t$(printf '%02d' "$TASK_ID")_init0"

MODELS="pi05_siglip_w8bf16_2b pi05_dkv0_9_w8bf16_1b pi05_dkv9_18_w8bf16_1b \
        pi05_ddn0_6_bf16_1b pi05_ddn6_12_bf16_1b pi05_ddn12_18_bf16_1b"

fail() {
    echo
    echo "ERROR: $1"
    [ -n "${2:-}" ] && echo "  -> $2"
    exit 1
}

echo "=== pi0.5 demo ==="
echo "task=$TASK_ID  steps=$STEPS  dev=$DEV_ID  loops=$LOOPS"
echo

# ---- 1. toolchain ----
command -v cmake >/dev/null || fail "cmake not found" "install cmake, or build on the host and copy the binary over"
command -v g++   >/dev/null || fail "g++ not found"   "install a C++ compiler"
[ -d "$SDK/include" ] || fail "libsophon not found at $SDK" "pass -S <libsophon path>"
echo "[ok] toolchain: $(g++ --version | head -1)"

# ---- 2. bmodels ----
missing=""
for m in $MODELS; do
    [ -f "$MODEL_DIR/$m.bmodel" ] || missing="$missing $m.bmodel"
done
if [ -n "$missing" ]; then
    echo "[--] missing bmodels:$missing"
    fail "bmodels not found in $MODEL_DIR" "run: cd scripts && ./download.sh bm1684x"
fi
echo "[ok] 6 bmodels present"

# ---- 3. dataset ----
[ -f "$DATA_DIR/action_unnorm.npz" ] || fail "missing $DATA_DIR/action_unnorm.npz" "run: cd scripts && ./download.sh bm1684x"
[ -d "$DATA_DIR/prefix_assets" ]     || fail "missing $DATA_DIR/prefix_assets/"     "run: cd scripts && ./download.sh bm1684x"
[ -f "$OBS/agentview.npy" ]          || fail "no observation at $OBS" "check the task id (-t) is 0-9"
[ -f "$OBS/wrist.npy" ]              || fail "no wrist view at $OBS"
echo "[ok] dataset and observation present"

# ---- 4. build (only if needed) ----
BIN="cpp/pi05_bmcv/pi05_bmcv.soc"
if [ ! -x "$BIN" ] || [ -n "$(find cpp/pi05_bmcv \( -name '*.cpp' -o -name '*.h' \) -newer "$BIN" 2>/dev/null)" ]; then
    echo "[..] building"
    mkdir -p cpp/pi05_bmcv/build
    ( cd cpp/pi05_bmcv/build \
      && cmake .. -DTARGET_ARCH=soc -DSDK="$SDK" > cmake.log 2>&1 \
      && make -j"$(nproc)" > make.log 2>&1 ) \
      || fail "build failed" "see cpp/pi05_bmcv/build/{cmake,make}.log"
    [ -x "$BIN" ] || fail "build produced no $BIN" "see cpp/pi05_bmcv/build/make.log"
fi
echo "[ok] binary: $BIN"

# ---- 5. run ----
echo
CASE="t$(printf '%02d' "$TASK_ID")_init0"
OUT="results/$CASE.npy"
mkdir -p results

# The dataset ships the initial noise each reference was computed from. Feeding it in
# makes the output directly comparable with the shipped ground truth; generating fresh
# noise from --seed would compare two different draws and inflate the measured error.
NOISE="$DATA_DIR/noise/$CASE.npy"
if [ -f "$NOISE" ]; then
    NOISE_ARGS="--noise $NOISE"
    echo "[ok] initial noise: $NOISE"
else
    NOISE_ARGS="--seed 0"
    echo "[--] no $NOISE shipped; falling back to --seed 0 (not comparable with actions_gt)"
fi
# shellcheck disable=SC2086
"$BIN" --bmodel_dir "$MODEL_DIR" \
       --data_dir "$DATA_DIR" \
       --input "$OBS" \
       --task_id "$TASK_ID" \
       --num_steps "$STEPS" \
       --loops "$LOOPS" \
       $NOISE_ARGS \
       --dev_id "$DEV_ID" \
       --output "$OUT"
rc=$?

echo
if [ $rc -ne 0 ]; then
    echo "FAILED (exit $rc)"
    exit $rc
fi
echo "OK — action chunk written to $OUT"
echo "Inspect it with:  python3 -c \"import numpy as np; print(np.load('$OUT'))\""
echo "Check accuracy with: python3 tools/compare_acc.py --pred $OUT --gt $DATA_DIR/actions_gt/$CASE.npy"
