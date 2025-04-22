#!/bin/bash
#./compile.sh --name qwen2.5-vl-3b --seq_length 2048 --chip bm1684x
set -ex
models=
quantize_args=""
name="qwen2.5-vl-3b"
num_layers=
hidden_size=
mode="w4bf16"
chip="bm1684x"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --name)
        name="$2"
        shift 2
        ;;
    --seq_length)
        seq_length="$2"
        shift 2
        ;;
    --chip)
        chip="\$2"
        shift 2
        ;;
    *)
        echo "Invalid option: $key" >&2
        exit 1
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1
        ;;
    esac
done

if [[ -z "$seq_length" ]]; then
    echo "Error: --seq_length is required." >&2
    exit 1
fi

if [ "$name" = "qwen2.5-vl-7b" ]; then
  num_layers=28
  hidden_size=3584
  echo "Compile Qwen2.5-VL-7B"
elif [ "$name" = "qwen2.5-vl-3b" ]; then
  num_layers=36
  hidden_size=2048
  echo "Compile Qwen2.5-VL-3B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mqwen2.5-vl-3b|qwen2.5-vl-7b\033[0m"
  exit 1
fi

quantize_args="--quantize W4BF16 --q_group_size 128"
half_quantize_args="--quantize BF16"

out_model=$name'_'$chip'_'$mode'_seq'$seq_length'.bmodel'


MODEL_DIR=${DIR}/tmp/onnx
COMPILE_DIR=${DIR}/tmp/$mode"_1dev"
TASK_FILE=${COMPILE_DIR}/task.txt


embedding() {
    echo \
    model_convert.py \
        --model_name embedding \
        --model_def ${MODEL_DIR}/embedding.pt \
        --input_shapes [[1,${seq_length}]] \
        --input_types "int32" \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip ${chip} \
        --debug \
        --model embedding.bmodel \
        >> ${TASK_FILE}

    models=${models}${COMPILE_DIR}'/embedding.bmodel '

    echo \
    model_convert.py \
        --model_name embedding_cache \
        --model_def ${MODEL_DIR}/embedding.pt \
        --input_shapes [[1,1]] \
        --input_types "int32" \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip ${chip} \
        --debug \
        --model embedding_cache.bmodel \
        >> ${TASK_FILE}

    models=${models}${COMPILE_DIR}'/embedding_cache.bmodel '

}

lm_head() {
    echo \
    model_convert.py \
        --model_name lm_head \
        --model_def ${MODEL_DIR}/lm_head.pt \
        --input_shapes [[1,${hidden_size}]] \
        ${half_quantize_args} \
        --high_precision \
        --quant_input \
        --chip ${chip} \
        --debug \
        --model lm_head.bmodel \
        >> ${TASK_FILE}

    models=${models}${COMPILE_DIR}'/lm_head.bmodel '
}

block() {
    for ((i=0; i<$num_layers; i++)); do
        echo \
        model_convert.py \
            --model_name block_$i \
            --model_def ${MODEL_DIR}/block_$i.onnx \
            ${quantize_args} \
            --high_precision \
            --quant_input \
            --quant_output \
            --chip ${chip} \
            --debug \
            --model block_$i.bmodel \
            >> ${TASK_FILE}

        echo \
        model_convert.py \
            --model_name block_cache_$i \
            --model_def ${MODEL_DIR}/block_cache_$i.onnx \
            ${quantize_args} \
            --high_precision \
            --quant_input \
            --quant_output \
            --chip ${chip} \
            --addr_mode io_alone \
            --debug \
            --model block_cache_$i.bmodel \
            >> ${TASK_FILE}
        
        models=${models}${COMPILE_DIR}'/block_'$i'.bmodel '${COMPILE_DIR}'/block_cache_'$i'.bmodel '
    done
}

vision_transformer() {
    echo \
    model_convert.py \
        --model_name vit \
        --model_def ${MODEL_DIR}/vit/vision_transformer.onnx \
        --do_onnx_sim True \
        ${half_quantize_args} \
        --quant_output \
        --high_precision \
        --chip ${chip} \
        --debug \
        --model vit.bmodel \
        >> ${TASK_FILE}

    models=${models}${COMPILE_DIR}'/vit.bmodel '
}

mkdir -p $COMPILE_DIR
echo $COMPILE_DIR
: > ${TASK_FILE}
pushd $COMPILE_DIR
vision_transformer
block
embedding
lm_head

parallel -j $(nproc) --progress --joblog ${TASK_FILE}.log < ${TASK_FILE}
[[ $? -ne 0 ]] && { echo "Error: model convert failed"; exit 1; }

rm -f *.npz *.onnx
popd

model_tool --combine $models -o $out_model
echo "Success: gen model $out_model"
