#!/bin/bash

pushd $(dirname $(readlink -f "$0"))
source model_info.sh
popd

if [ ! -d $build_dir ]; then
    mkdir -p $build_dir
fi

pushd $build_dir

check_file $src_model_file

if [ ! -d "$fp32model_dir" ]; then
    echo "create data dir: $fp32model_dir"
    mkdir -p $fp32model_dir
fi

function gen_fp32bmodel()
{
	python3 -m bmnetp \
		--mode="compile" \
		--model="${src_model_file}" \
		--outdir="${fp32model_dir}" \
		--target=${platform} \
		--shapes=[[$1, 3, ${img_height}, ${img_width}]] \
		--net_name=$dst_model_prefix \
		--opt=1 \
		--dyn=False \
		--cmp=True \
		--enable_profile=True 
}

gen_fp32bmodel 1

if [ $? -eq 0 ]; then
    cp "${fp32model_dir}/compilation.bmodel" "${fp32model_dir}/../${dst_model_prefix}_${platform}_fp32_${batch_size}b.bmodel" -rf
    echo "Congratulation! fp32bmodel is done!"
else
    echo "Something is wrong, pleae have a check!"
    popd
    exit -1
fi

popd
