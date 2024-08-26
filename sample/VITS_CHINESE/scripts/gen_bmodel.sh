#!/bin/bash
set -ex
model_dir=$(dirname $(readlink -f "$0"))
pushd $model_dir

if [ ! $1 ]; then
	target=bm1684x
	target_dir=BM1684X
else
	target=${1,,}
	target_dir=${target^^}
fi

outdir=../models/$target_dir
if [ ! -d $outdir ]; then
	mkdir -p $outdir
fi

function gen_mlir()
{
	model_transform.py \
		--model_name vits_chinese \
		--model_def ../vits_chinese_128.onnx \
		--input_shapes [[1,128],[1,128,256]] \
		--input_types [int32,float32] \
		--mlir vits_chinese.mlir
  
	model_transform.py \
		--model_name bert \
		--model_def ../bert.onnx \
		--mlir bert.mlir \
		--input_shapes [[1,64],[1,64],[1,64]] 

}

function gen_bmodel()
{
	model_deploy.py \
		--mlir vits_chinese.mlir \
		--quantize F16 \
		--chip $target \
		--model vits_chinese_f16.bmodel \
		--compare_all \
		--debug
	mv vits_chinese_f16.bmodel $outdir/

	model_deploy.py \
		--mlir bert.mlir \
		--quantize F16 \
		--chip $target \
		--num_core 1 \
		--model bert_f16_1core.bmodel \
		--compare_all \
		--debug 

	mv bert_f16_1core.bmodel $outdir/
	rm -rf obj/
	mkdir -p obj/
	find . -maxdepth 1 -type f ! -name '*.sh' -exec mv {} obj/ \;
	find . -maxdepth 1 -type d ! -name 'obj' ! -name '.' -exec mv {} obj/ \;
}

gen_mlir 
gen_bmodel 
popd