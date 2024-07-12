#!/bin/bash

work_dir="../models"
if [ ! -d "$work_dir/onnx" ]; then
    echo "[Err] "$work_dir/onnx" directory not found, please download onnx model first"
    exit 1
fi

bmodel_dir="../models/BM1684X"
if [ ! -d "$bmodel_dir" ]; then
    mkdir "$bmodel_dir"
    echo "[Cmd] mkdir $bmodel_dir"
fi


pushd "$work_dir"

function gen_dynamic_fp16bmodel()
{
    model_transform.py \
        --model_name $1 \
        --model_def ${2} \
        --input_shapes $3 \
        --mlir transformed.mlir \
        --dynamic_inputs input_seqs
    
    model_deploy.py \
        --mlir transformed.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model $4 \
        --dynamic
    mv $4 $bmodel_dir/
}

function gen_static_fp16bmodel()
{
    model_transform.py \
        --model_name $1 \
        --model_def ${2} \
        --input_shapes $3 \
        --mlir transformed.mlir
    
    model_deploy.py \
        --mlir transformed.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model $4
    mv $4 $bmodel_dir/
}

function gen_static_fp32bmodel()
{
    model_transform.py \
        --model_name $1 \
        --model_def ${2} \
        --input_shapes $3 \
        --mlir transformed.mlir
    
    model_deploy.py \
        --mlir transformed.mlir \
        --quantize F32 \
        --chip bm1684x \
        --model $4
    mv $4 $bmodel_dir/
}

encoder_frontend_model_name=seamless_streaming_unity_speech_encoder_frontend
encoder_frontend_onnx_file=onnx/streaming_s2t_onnx/streaming_s2t_encoder_frontend/seamless_streaming_unity_speech_encoder_frontend.onnx
encoder_frontend_input_shapes=[[1,160,80]]
encoder_frontend_bmodel_file=seamless_streaming_encoder_frontend_fp16_s2t.bmodel 
gen_dynamic_fp16bmodel $encoder_frontend_model_name $encoder_frontend_onnx_file $encoder_frontend_input_shapes $encoder_frontend_bmodel_file

encoder_model_name=seamless_streaming_unity_speech_encoder
encoder_onnx_file=onnx/streaming_s2t_onnx/streaming_s2t_encoder/seamless_streaming_unity_speech_encoder.onnx
encoder_input_shapes=[[1,80,1024],[1]]
encoder_bmodel_file=seamless_streaming_encoder_fp16_s2t.bmodel
gen_static_fp16bmodel $encoder_model_name $encoder_onnx_file $encoder_input_shapes $encoder_bmodel_file

decoder_frontend_model_name=seamless_streaming_monotonic_text_decoder_frontend
decoder_frontend_onnx_file=onnx/streaming_s2t_onnx/streaming_s2t_decoder_frontend/seamless_streaming_monotonic_decoder_text_decoder_frontend.onnx
decoder_frontend_input_shapes=[[1,64],[1]]
decoder_frontend_bmodel_file=seamless_streaming_decoder_frontend_fp16_s2t.bmodel
gen_static_fp16bmodel $decoder_frontend_model_name $decoder_frontend_onnx_file $decoder_frontend_input_shapes $decoder_frontend_bmodel_file

decoder_step_equal_1_model_name=seamless_streaming_monotonic_text_decoder_step_equal_1
decoder_step_equal_1_onnx_file=onnx/streaming_s2t_onnx/streaming_s2t_decoder_step_equal_1/seamless_streaming_monotonic_decoder_text_decoder.onnx
decoder_step_equal_1_input_shapes=[[1,64,1024],[64,64],[1,11,1024],[64,11]]
decoder_step_equal_1_bmodel_file=seamless_streaming_decoder_step_equal_1_fp16_s2t.bmodel
gen_static_fp16bmodel $decoder_step_equal_1_model_name $decoder_step_equal_1_onnx_file $decoder_step_equal_1_input_shapes $decoder_step_equal_1_bmodel_file

decoder_step_bigger_than_1_64kvcache_model_name=seamless_streaming_monotonic_text_decoder_step_bigger_than_1_64kvcache
decoder_step_bigger_than_1_64kvcache_onnx_file=onnx/streaming_s2t_onnx/streaming_s2t_decoder_step_bigger_than_1/seamless_streaming_monotonic_decoder_text_decoder.onnx
decoder_step_bigger_than_1_64kvcache_input_shapes=[[1,1,1024],[1,64],[1,11,1024],[1,11],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64],[1,16,64,64]]
decoder_step_bigger_than_1_64kvcache_bmodel_file=seamless_streaming_decoder_step_bigger_1_fp16_s2t.bmodel
gen_static_fp16bmodel $decoder_step_bigger_than_1_64kvcache_model_name $decoder_step_bigger_than_1_64kvcache_onnx_file $decoder_step_bigger_than_1_64kvcache_input_shapes $decoder_step_bigger_than_1_64kvcache_bmodel_file

decoder_final_proj_model_name=seamless_streaming_monotonic_decoder_final_proj
decoder_final_proj_onnx_file=onnx/streaming_s2t_onnx/streaming_s2t_final_proj/seamless_streaming_monotonic_decoder_final_proj.onnx
decoder_final_proj_input_shapes=[[1,1,1024]]
decoder_final_proj_bmodel_file=seamless_streaming_decoder_final_proj_fp16_s2t.bmodel
gen_static_fp16bmodel $decoder_final_proj_model_name $decoder_final_proj_onnx_file $decoder_final_proj_input_shapes $decoder_final_proj_bmodel_file
popd
