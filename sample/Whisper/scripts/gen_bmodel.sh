#!/bin/bash
script_dir=$(dirname $(readlink -f "$0"))
# Default values for parameters
model="small"
beam_size=5
padding_size=448
quant=true
process=""


work_dir="$script_dir/../models"
onnx_dir="$work_dir/onnx"
if [ ! -d "$work_dir/onnx" ]; then
    echo "[Err] "$work_dir" directory not found, please download onnx model first"
    exit 1
fi

bmodel_dir="./models/BM1684X"
if [ ! -d "$bmodel_dir" ]; then
    mkdir "$bmodel_dir"
    echo "[Cmd] mkdir $bmodel_dir"
fi


pushd "$work_dir"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            model="$2"
            shift 2
            ;;
        --process)
            process="$2"
            shift 2
            ;;
        *)
            # Unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# padding_size_1=$((padding_size - 1))
padding_size_1=$((padding_size))
n_mels=80
n_audio_ctx=1500


if [ "$model" == "base" ]; then
	n_text_state=512
	n_text_head=8
	n_text_layer=6
    process_list=("encoder" "logits_decoder" "decoder_main_with_kvcache" "decoder_post" "decoder_loop_with_kvcache" "kvcache_rearrange" )
elif [ "$model" == "small" ]; then
	n_text_state=768
	n_text_head=12
	n_text_layer=12
    process_list=("encoder" "logits_decoder" "decoder_main_with_kvcache" "decoder_post" "decoder_loop_with_kvcache" "kvcache_rearrange" )
elif [ "$model" == "medium" ]; then
	n_text_state=1024
	n_text_head=16
	n_text_layer=24
    process_list=("encoder" "logits_decoder" "decoder_main_with_kvcache" "decoder_post" "decoder_loop_with_kvcache" "kvcache_rearrange" )
elif [ "$model" == "small.en" ]; then
	n_text_state=768
	n_text_head=12
	n_text_layer=12
    process_list=("encoder" "decoder_main_with_kvcache" "decoder_post" "decoder_loop_with_kvcache" "kvcache_rearrange" )
elif [ "$model" == "distil.small.en" ]; then
	n_text_state=768
	n_text_head=12
	n_text_layer=4
    process_list=("encoder" "decoder_main_with_kvcache" "decoder_post" "decoder_loop_with_kvcache" "kvcache_rearrange" )
else
	echo "model must be one of tiny, base, small, medium, large, small.en, distil.small.en"
	exit 1
fi

function gen_bmodel() {
    echo "Transforming $process_name ..."
    case $process_name in
        encoder)
            input_shapes="[[1,80,3000]]"
            ;;
        logits_decoder)
            input_shapes="[[1,1],[1,${n_audio_ctx},${n_text_state}]]"
            ;;
        decoder_main_with_kvcache)
            input_shapes="[[$beam_size,$padding_size],[1,$n_audio_ctx,$n_text_state],[$padding_size,$n_text_state],[$beam_size,$padding_size,$n_text_head,$padding_size]]"
            ;;
        decoder_post)
            input_shapes="[[$beam_size,1,$n_text_state],[$beam_size,1,$n_text_state]]"
            ;;
        decoder_loop_with_kvcache)
            input_shapes="[[$beam_size,1],[1,$n_text_state],[$beam_size,1,$n_text_head,$padding_size]"
            for ((i=0; i<$((n_text_layer * 2)); i++)); do
                input_shapes="${input_shapes},[$beam_size,$padding_size_1,$n_text_state]"
            done
            for ((i=0; i<$((n_text_layer * 2)); i++)); do
                input_shapes="${input_shapes},[1,$n_audio_ctx,$n_text_state]"
            done
            input_shapes="${input_shapes}]"
            ;;
        kvcache_rearrange)
            input_shapes="[[$beam_size,$padding_size,$n_text_state],[$beam_size]]"
            ;;
        *)
            # Unknown option
            echo "Unknown option: $process_name"
            exit 1
            ;;
    esac

    onnx_file="${onnx_dir}/${model_name}.onnx"
    model_transform_cmd="model_transform.py --model_name $model_name \
        --model_def ${onnx_file} \
        --input_shapes $input_shapes \
        --mlir transformed.mlir"

    model_deploy_cmd="model_deploy.py --mlir transformed.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model $bmodel_file"

    if [ "$quant" = true ]; then
        model_deploy_cmd="$model_deploy_cmd --quant_input --quant_output"
    fi

    echo "[Cmd] Running command: $model_transform_cmd"
    eval "$model_transform_cmd"

    echo "[Cmd] Running command: $model_deploy_cmd"
    eval "$model_deploy_cmd"

    echo "[Msg] Bmodel generate done!"
}

echo "Generating process list ..."

echo "process list: ${process_list[@]}"

models=""
for process_name in "${process_list[@]}"; do
    model_name="${process_name}_${model}_${beam_size}beam_${padding_size}pad"
    bmodel_file="all_quant_${model_name}_1684x_f16.bmodel"
    models="${models} ${bmodel_file}"
    if [ -e "$bmodel_dir/$bmodel_file" ]; then
        echo "[Msg] $bmodel_dir/$bmodel_file already exists, skip this process"
        continue
    fi

    if [ ! -d "$process_name" ]; then
        mkdir "$process_name"
        echo "[Cmd] mkdir $process_name"
    fi
    pushd "$process_name"
    if [ ! -d "$model_name" ]; then
        mkdir "$model_name"
        echo "[Cmd] mkdir $model_name"
    fi
    pushd "$model_name"
    gen_bmodel
    echo "[Cmd] mv $bmodel_file ../../BM1684X/"
    mv $bmodel_file ../../BM1684X/
    popd
    rm -rf $bmodel_file
    popd
done
popd
chmod -R 777 $bmodel_dir/
cd $bmodel_dir/

model_tool --combine $models \
                     -o bmwhisper_${model}_1684x_f16.bmodel && rm $models
chown 1000:1000 bmwhisper_${model}_1684x_f16.bmodel