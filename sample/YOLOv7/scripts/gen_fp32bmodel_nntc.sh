##############################################################################
# 将yolov7 torchscript模型转换为fp32bmodel
# 注意：
#     1.下载的为官方原生模型，必须要先转成torchscript模型后继续后面转换
##############################################################################
#!/bin/bash

root_dir=$(dirname $(readlink -f "$0"))/..
if [ $# -lt 1 ];then
    echo "please input platform($#), eg:BM1684"
    exit -1
fi

platform=$1

output_dir="${root_dir}/models/${platform}"
if [ ! -d "$output_dir" ]; then
    echo "create data dir: $output_dir"
    mkdir -p $output_dir
fi


function gen_fp32bmodel()
{
    python3 -m bmnetp \
       --net_name=yolov7 \
       --target=${platform} \
       --opt=1 \
       --cmp=true \
       --shapes=[[$1,3,640,640]] \
       --model="${root_dir}/models/torch/yolov7_v0.1_3outputs.torchscript.pt" \
       --outdir=${output_dir} \
       --dyn=false
    mv ${output_dir}/compilation.bmodel ${output_dir}/yolov7_v0.1_3output_fp32_$1b.bmodel
}

echo "start fp32bmodel transform, platform: ${platform}......"
pushd ${root_dir}
# bacth_size = 1
gen_fp32bmodel 1
# batch_size = 4
gen_fp32bmodel 4
if [ $? -eq 0 ]; then
    echo "Congratulation! fp32bmode is done!"
    popd
    exit 0
else
    echo "Something is wrong, pleae have a check!"
    popd
    exit -1
fi
