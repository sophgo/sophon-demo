#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

outdir=../models/$target

function gen_fp32_encoder_bmodel()
{
    python3 -m bmneto \
            --model=../models/onnx/wenet_encoder.onnx \
            --target=$target \
            --input_names="chunk_xs,chunk_lens,offset,att_cache,cnn_cache,cache_mask" \
            --shapes=[[1,67,80],[1],[1,1],[1,12,4,80,128],[1,12,256,7],[1,1,80]] \
            --net_name=wenet_encoder \
            --dyn=False \
            --cmp=False \
            --descs="[1,int32,0,1000],[2,int64,0,1000]"
    mv compilation/compilation.bmodel $outdir/wenet_encoder_fp32.bmodel
}

function gen_fp32_decoder_bmodel()
{
    python3 -m bmneto \
            --model=../models/onnx/wenet_decoder.onnx \
            --target=$target \
            --input_names="encoder_out,encoder_out_lens,hyps_pad_sos_eos,hyps_lens_sos,r_hyps_pad_sos_eos,ctc_score" \
            --shapes=[[1,350,256],[1],[1,10,350],[1,10],[1,10,350],[1,10]] \
            --net_name=wenet_decoder \
            --dyn=False \
            --cmp=False \
            --descs="[1,int32,0,1000],[2,int64,0,1000],[3,int32,0,1000],[4,int64,0,1000]"
    mv compilation/compilation.bmodel $outdir/wenet_decoder_fp32.bmodel
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

# batch_size=1
gen_fp32_encoder_bmodel
gen_fp32_decoder_bmodel

popd
