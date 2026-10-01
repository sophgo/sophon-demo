#!/bin/bash
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# Download the BoofCV QR Code V4 evaluation dataset (used by tools/eval_qrcode.py
# to measure detection + decoding accuracy) from dfss.
#
# The dfss tarball is a repackaged copy of the official qrcodes_v4.zip; it
# unpacks to a top-level BoofCV_qrcode_v4/ dir:
#   BoofCV_qrcode_v4/qrcodes/detection/<16 categories>/imageNNN.{jpg,png,txt}
#     (718 images / 1441 QR codes; .txt ground truth = corner sets)
#   BoofCV_qrcode_v4/qrcodes/decoding/<name>.{png,txt}
#     (26 clean synthetic codes; <name>.txt = expected text)
#
# Original source: https://boofcv.org/notwiki/regression/fiducial/qrcodes_v4.zip
# License: main library Apache-2.0, data repository CC-BY-4.0 (Peter Abeles);
# see the README for attribution.

set -euo pipefail

scripts_dir=$(dirname $(readlink -f "$0"))
DATASETS="$scripts_dir/../datasets"

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

mkdir -p "$DATASETS"
pushd "$DATASETS"

if [ -d "BoofCV_qrcode_v4" ]; then
  echo "dataset already extracted at $DATASETS/BoofCV_qrcode_v4; remove it if you need to update."
else
  python3 -m dfss --url=open@sophgo.com:sophon-demo/wechat_qrcode/datasets/BoofCV_qrcode_v4.tar.gz
  tar xvf BoofCV_qrcode_v4.tar.gz && rm BoofCV_qrcode_v4.tar.gz
  echo "dataset download done: $DATASETS/BoofCV_qrcode_v4"
fi

popd