#!/usr/bin/env bash
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# WeChatQRCode bmodel generator (tpu-mlir, caffe frontend).
#
# Produces, for one target chip, the delivered model pair:
#   detect_f32_fused.bmodel   detect: SSD-MobileNet, 19 static shapes merged
#                             into one multi-network bmodel (F32): the caffe
#                             frontend does not support dynamic shape, so a
#                             "shape table" simulates OpenCV's proportional
#                             rescale s=min(1,sqrt(160000/(w*h))).
#   sr_f16_fused.bmodel       sr: super-resolution 224x224 -> 447x447 (F16).
#
# Both models use --fuse_preprocess + --pixel_format gray + scale 0.0039216,
# so the network input is a raw uint8 single-channel image (the demo feeds
# data by input dtype, no explicit normalization on the host).
#
# Prerequisite: run inside the tpu-mlir environment (e.g. the lcx_mlir
# container, with the host /home mounted, so relative paths below resolve to
# the host). The caffe source models (OpenCV 4.8.0 native detect/sr) live in
# CAFFE_SRC; override it if your checkout differs.
#
# Usage:
#   scripts/gen_bmodel.sh [bm1684x|bm1688|cv186x]    # default: bm1684x

set -euo pipefail

CHIP=${1:-bm1684x}
# OpenCV 4.8.0 native caffe models (detect/sr prototxt + caffemodel), fetched by
# scripts/download.sh into models/opencv_3rdparty_wechat_qrcode/ (Apache-2.0,
# from WeChatCV/opencv_3rdparty; the same tarball sophon-sail ships). Override
# CAFFE_SRC if your checkout differs.
CAFFE_SRC=${CAFFE_SRC:-$(dirname $(readlink -f "$0"))/../models/opencv_3rdparty_wechat_qrcode}
OUT=$(dirname $(readlink -f "$0"))/../models

case "$CHIP" in
  bm1684x) DIR=BM1684X;;
  bm1688)  DIR=BM1688;;
  cv186x)  DIR=CV186X;;
  *) echo "usage: $0 [bm1684x|bm1688|cv186x]"; exit 1;;
esac

WORK="$OUT/tmp/$CHIP"
mkdir -p "$WORK" "$OUT/$DIR"

# The 19-shape table: (width, height), area ~160000, aspect ratios 1:1..8:1
# plus transposes. W/H sampled at ~1.25 geometric intervals (~5.7% max error).
SHAPES=(
  "400 400" "448 356" "504 316" "564 284" "632 252" "716 224"
  "800 200" "896 180" "1012 160" "1132 140" "356 448" "316 504"
  "284 564" "252 632" "224 716" "200 800" "180 896" "160 1012" "140 1132"
)

detect_transform() {  # <W> <H> <name>
  # name the network "detect_<W>_<H>" (not a fixed "detect") so the runtime can
  # parse the static shape back out of the graph name (see ssd_detector.cpp
  # parseGraphName); 19 sub-networks must keep distinct names through combine.
  model_transform.py \
    --model_name "$(basename "$3")" \
    --model_def "$CAFFE_SRC/detect.prototxt" \
    --model_data "$CAFFE_SRC/detect.caffemodel" \
    --input_shapes "[[1,1,$2,$1]]" \
    --mean 0 --scale 0.0039216 --pixel_format gray \
    --mlir "$3.mlir"
}

detect_deploy() {  # <name> <out.bmodel>
  model_deploy.py \
    --mlir "$1.mlir" --chip "$CHIP" --quantize F32 \
    --fuse_preprocess --model "$2"
}

echo "===== detect multi-shape ($CHIP, 19 shapes, F32 fused) ====="
DETECT_ACC=()
for entry in "${SHAPES[@]}"; do
  read -r W H <<< "$entry"
  NAME="detect_${W}_${H}"
  MLIR="$WORK/${NAME}.mlir"
  BMODEL="$WORK/${NAME}_${CHIP}_f32_fused.bmodel"
  if [ ! -f "$BMODEL" ]; then
    detect_transform "$W" "$H" "$WORK/$NAME" 2>&1 | tail -1
    detect_deploy "$WORK/$NAME" "$BMODEL" 2>&1 | tail -1
    echo "[ok] $NAME"
  else
    echo "[skip] $NAME"
  fi
  DETECT_ACC+=("$BMODEL")
done

DETECT_OUT="$OUT/$DIR/detect_f32_fused.bmodel"
echo "===== combine ${#DETECT_ACC[@]} networks -> $DETECT_OUT ====="
model_tool --combine "${DETECT_ACC[@]}" -o "$DETECT_OUT" 2>&1 | tail -3
model_tool --info "$DETECT_OUT" 2>&1 | grep -E "net |input:|output:" | head -80

echo "===== sr ($CHIP, F16 fused) ====="
model_transform.py \
  --model_name sr \
  --model_def "$CAFFE_SRC/sr.prototxt" \
  --model_data "$CAFFE_SRC/sr.caffemodel" \
  --input_shapes "[[1,1,224,224]]" \
  --mean 0 --scale 0.0039216 --pixel_format gray \
  --mlir "$WORK/sr.mlir"
SR_OUT="$OUT/$DIR/sr_f16_fused.bmodel"
model_deploy.py \
  --mlir "$WORK/sr.mlir" --chip "$CHIP" --quantize F16 \
  --fuse_preprocess --model "$SR_OUT" 2>&1 | tail -1

echo "===== done ====="
echo "detect: $DETECT_OUT"
echo "sr    : $SR_OUT"