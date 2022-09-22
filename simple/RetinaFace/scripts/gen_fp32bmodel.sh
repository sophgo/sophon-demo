#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
target=$1
outdir=../data/models/$target


function gen_fp32bmodel()
{
    python3 -m bmneto --net_name=retinaface_mobilenet0.25 \
                      --target=$target \
                      --opt=1 \
                      --cmp=true \
                      --shapes=[$1,3,640,640] \
                      --model=../data/models/onnx/retinaface_mobilenet0.25.onnx \
                      --outdir=$outdir \
                      --dyn=false
    mv $outdir/compilation.bmodel $outdir/retinaface_mobilenet0.25_fp32_$1b.bmodel

}

pushd $model_dir
#batch_size=1
gen_fp32bmodel 1
#batch_size=4
gen_fp32bmodel 4
# tpu_model --combine $outdir/retinaface_mobilenet0.25_fp32_1b.bmodel $outdir/retinaface_mobilenet0.25_fp32_4b.bmodel -o $outdir/retinaface_mobilenet0.25_fp32_1b4b.bmodel 
popd
