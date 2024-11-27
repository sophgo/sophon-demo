script_dir=$(dirname $(readlink -f "$0"))
if [ ! $1 ]; then
    target=bm1684x
else
    target=${1,,}
fi

pushd $script_dir

mkdir -p vocos_mlir_files
model_transform.py --model_name chattts_vocos \
--model_def ../models/onnx/vocos_1-100-2048.onnx \
--input_shapes [[1,100,2048]] \
--mlir vocos_mlir_files/chattts_vocos.mlir

model_deploy.py --mlir vocos_mlir_files/chattts_vocos.mlir \
--model vocos_1-100-2048_$target.bmodel \
--quantize BF16 \
--chip $target

popd
