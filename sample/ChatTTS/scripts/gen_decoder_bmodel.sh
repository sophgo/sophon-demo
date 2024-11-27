script_dir=$(dirname $(readlink -f "$0"))
if [ ! $1 ]; then
    target=bm1684x
else
    target=${1,,}
fi

pushd $script_dir

mkdir -p decoder_mlir_files
model_transform.py --model_name chattts_decoder \
--model_def ../models/torch/decoder_jit.pt \
--input_shapes [[1,768,1024]] \
--mlir decoder_mlir_files/chattts_decoder.mlir

model_deploy.py --mlir decoder_mlir_files/chattts_decoder.mlir \
--model decoder_1-768-1024_$target.bmodel \
--quantize BF16 \
--chip $target

popd
