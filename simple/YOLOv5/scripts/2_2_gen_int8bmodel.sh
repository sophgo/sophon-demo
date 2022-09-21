#!/bin/bash

pushd $(dirname $(readlink -f "$0"))
source model_info.sh
popd

check_file $src_model_file
check_dir $lmdb_src_dir

auto_cali_dir=$build_dir/auto_cali

if [ ! -d "${auto_cali_dir}" ]; then
    echo "create data dir: ${auto_cali_dir}"
    mkdir -p ${auto_cali_dir}
fi

pushd ${auto_cali_dir}

if [ ! -d "$int8model_dir" ]; then
    echo "create data dir: $int8model_dir"
    mkdir -p $int8model_dir
fi

python3 -m ufw.cali.cali_model  \
    --net_name=$dst_model_prefix  \
    --model=${src_model_file}  \
    --cali_image_path=${image_src_dir}  \
    --cali_iterations=200   \
    --cali_image_preprocess='resize_h=640,resize_w=640;scale=0.003921569,bgr2rgb=True'   \
    --input_shapes="[${batch_size},3,${img_height},${img_width}]"  \
    --outdir=${int8model_dir}   \
    --target=${platform}   \
    --convert_bmodel_cmd_opt="-opt=1"   \
    --try_cali_accuracy_opt="-fpfwd_outputs=< 24 >86,< 24 >55,< 24 >18;-th_method=ADMM"

    

if [ $? -eq 0 ]; then
    cp ${int8model_dir}/${dst_model_prefix}_batch${batch_size}/compilation.bmodel ${int8model_dir}/../${dst_model_prefix}_${img_size}_${dst_model_postfix}_int8_${batch_size}b.bmodel
    echo "Congratulation! in8bmode is done!"
else
    echo "Something is wrong, pleae have a check!"
    popd
    exit -1
fi

popd
exit 0
