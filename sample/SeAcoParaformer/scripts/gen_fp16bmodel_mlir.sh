#!/usr/bin/env bash
# Compile SeACoParaformer bmodels for CV84X6 (SE13-64) from tools/export_onnx.py onnx.
# Usage: ./gen_fp16bmodel_mlir.sh [cv84x6]   (run inside tpu-mlir docker, from scripts/)
#
# Notes:
# - BF16 for all three parts (fp16 overflows in encoder block 31).
# - predictor is compiled STATIC (--dynamic LSTM codegen hangs the cv84x6
#   TPU driver at large seq; see tools/export_onnx.py which drops the pad
#   mask so the static compile succeeds). python/seaco_paraformer.py pads
#   enc_out to 1100 frames and slices us_alphas back to 3T.
set -e
target=${1:-cv84x6}
target_dir=${target^^}
scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

onnx_dir=../models/onnx
outdir=../models/$target_dir
mkdir -p $outdir

# npz inputs for --test_input (also reused for later npz compare)
python3 - <<'EOF'
import numpy as np
np.random.seed(0)
np.savez("encoder_input.npz",
         speech=np.random.randn(1, 1100, 560).astype(np.float32),
         speech_lengths=np.array([1100], dtype=np.int32))
np.savez("decoder_input.npz",
         enc=np.random.randn(1, 1100, 512).astype(np.float32),
         enc_len=np.array([1100], dtype=np.int32),
         pre_acoustic_embeds=np.random.randn(1, 600, 512).astype(np.float32),
         pre_token_length=np.array([600], dtype=np.int32))
np.savez("predictor_input.npz",
         enc=np.random.randn(1, 1100, 512).astype(np.float32))
EOF

# dynamic compile, batch-agnostic; --disable_layer_group needed (LSTM/SANM precedent)
# encoder
model_transform.py \
    --model_name encoder \
    --model_def $onnx_dir/encoder.onnx \
    --input_shapes [[1,1100,560],[1]] \
    --test_input encoder_input.npz \
    --test_result encoder_top_results.npz \
    --dynamic \
    --mlir encoder.mlir
model_deploy.py \
    --mlir encoder.mlir \
    --quantize BF16 \
    --chip $target \
    --dynamic \
    --disable_layer_group \
    --model encoder_bf16_1b.bmodel

# decoder (4 inputs; enc_len/pre_token_length influence shapes)
model_transform.py \
    --model_name decoder \
    --model_def $onnx_dir/decoder.onnx \
    --input_shapes [[1,1100,512],[1],[1,600,512],[1]] \
    --test_input decoder_input.npz \
    --test_result decoder_top_results.npz \
    --dynamic \
    --shape_influencing_input_names enc_len,pre_token_length \
    --mlir decoder.mlir
model_deploy.py \
    --mlir decoder.mlir \
    --quantize BF16 \
    --chip $target \
    --dynamic \
    --disable_layer_group \
    --model decoder_bf16_1b.bmodel

# predictor (CifPredictorV3 upsample head) -- STATIC compile:
# --dynamic + LSTM at large seq hangs the cv84x6 TPU driver, and the
# pad-mask (Range op) that blocked static compile was dropped at export.
model_transform.py \
    --model_name predictor_static \
    --model_def $onnx_dir/predictor.onnx \
    --input_shapes [[1,1100,512]] \
    --test_input predictor_input.npz \
    --test_result predictor_top_results.npz \
    --mlir predictor.mlir
model_deploy.py \
    --mlir predictor.mlir \
    --quantize BF16 \
    --chip $target \
    --disable_layer_group \
    --model predictor_bf16_1b.bmodel

mv encoder_bf16_1b.bmodel decoder_bf16_1b.bmodel predictor_bf16_1b.bmodel $outdir/
echo "done -> $outdir"
popd
