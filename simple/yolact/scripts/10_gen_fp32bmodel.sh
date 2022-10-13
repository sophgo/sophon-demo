#!/bin/bash
root_dir=$(dirname $(readlink -f "$0"))/..

if [ ! $1 ]; then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

data_dir=$root_dir/data
model_dir=$data_dir/models
trace_model_path=$model_dir/torch/yolact_base_54_800000.trace.pt
outdir=$root_dir/data/models/$target
if [ ! -d "$data_dir" ]; then
  echo "create data dir: $data_dir"
  mkdir -p $data_dir
fi

if [ ! -d "$model_dir" ]; then
  echo "create model dir: $model_dir"
  mkdir -p $model_dir
fi

if [ ! -f "$trace_model_path" ]; then
  echo "$trace_model_path is not existed. Please get yolact_base_54_800000.trace.pt"
  exit
fi

if [ ! -d "$outdir" ]; then
  echo "create output dir: $outdir"
  mkdir -p $outdir
fi

function gen_fp32bmodel()
{
	python3 -m bmnetp --net_name=yolact_base \
			  --target=$target \
			  --opt=1 \
			  --cmp=true \
			  --shapes="[$1,3,550,550]" \
			  --model=$trace_model_path \
			  --outdir=$outdir \
			  --dyn=false
	mv $outdir/compilation.bmodel $outdir/yolact_base_54_800000_fp32_$1b.bmodel
}

pushd $model_dir
# bs=1
gen_fp32bmodel 1
# bs=4
gen_fp32bmodel 4

popd
