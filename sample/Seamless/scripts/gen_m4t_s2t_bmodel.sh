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

function gen_dynamic_frontend_fp16bmodel()
{
    model_transform.py \
        --model_name $1 \
        --model_def $2 \
        --input_shapes $3 \
        --mlir transformed.mlir \
        --test_input ../scripts/decoder_frontend.npz \
        --test_result ./decoder_frontend_out.npz \
        --dynamic_inputs start_step \
        --inputs_is_shape start_step

    model_deploy.py --mlir transformed.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model $4 \
        --tolerance 0.99,0.85
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

encoder_frontend_model_name=m4t_unity_speech_encoder_frontend
encoder_frontend_onnx_file=onnx/m4t_s2t_onnx/m4t_s2t_encoder_frontend/m4t_unity_speech_encoder_frontend.onnx
encoder_frontend_input_shapes=[[1,1152,80]]
encoder_frontend_bmodel_file=m4t_encoder_frontend_fp16_s2t.bmodel 
gen_dynamic_fp16bmodel $encoder_frontend_model_name $encoder_frontend_onnx_file $encoder_frontend_input_shapes $encoder_frontend_bmodel_file

encoder_model_name=m4t_unity_speech_encoder
encoder_onnx_file=onnx/m4t_s2t_onnx/m4t_s2t_encoder/m4t_unity_speech_encoder.onnx
encoder_input_shapes=[[1,576,1024],[1]]
encoder_bmodel_file=m4t_encoder_fp16_s2t.bmodel
gen_static_fp16bmodel $encoder_model_name $encoder_onnx_file $encoder_input_shapes $encoder_bmodel_file

decoder_frontend_model_name=m4t_decoder_frontend_beam_size_s2t
decoder_frontend_onnx_file=onnx/m4t_s2t_onnx/m4t_s2t_decoder_frontend/m4t_decoder_frontend_beam_size_s2t.onnx
decoder_frontend_input_shapes=[[5,1],[1]]
decoder_frontend_bmodel_file=m4t_decoder_frontend_beam_size_fp16_s2t.bmodel
gen_dynamic_frontend_fp16bmodel $decoder_frontend_model_name $decoder_frontend_onnx_file $decoder_frontend_input_shapes $decoder_frontend_bmodel_file

decoder_model_name=m4t_decoder_beam_size_s2t
decoder_onnx_file=onnx/m4t_s2t_onnx/m4t_s2t_decoder/m4t_decoder_beam_size_s2t.onnx
decoder_input_shapes=[[5,1,1024],[1,32],[5,73,1024],[1,73],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64],[5,16,32,64]]
decoder_bmodel_file=m4t_decoder_beam_size_fp16_s2t.bmodel
gen_static_fp16bmodel $decoder_model_name $decoder_onnx_file $decoder_input_shapes $decoder_bmodel_file

decoder_final_proj_model_name=m4t_decoder_final_proj_beam_size_s2t
decoder_final_proj_onnx_file=onnx/m4t_s2t_onnx/m4t_s2t_final_proj/m4t_decoder_final_proj_beam_size_s2t.onnx
decoder_final_proj_input_shapes=[[5,1,1024]]
decoder_final_proj_bmodel_file=m4t_decoder_final_proj_beam_size_fp16_s2t.bmodel
gen_static_fp16bmodel $decoder_final_proj_model_name $decoder_final_proj_onnx_file $decoder_final_proj_input_shapes $decoder_final_proj_bmodel_file
popd
