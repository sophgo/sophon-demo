#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir/../python/

python3 -m dfss --url=open@sophgo.com:sophon-demo/VITA1_5/models/codec_bm1684x_fp16_1core.bmodel
python3 -m dfss --url=open@sophgo.com:sophon-demo/VITA1_5/models/vita-Qwen2_bm1684x_int4_1core.bmodel
python3 -m dfss --url=open@sophgo.com:sophon-demo/VITA1_5/models/vqvae_fp16_1b.bmodel
python3 -m dfss --url=open@sophgo.com:sophon-demo/VITA1_5/datasets.tar.gz
tar xvf datasets.tar.gz && rm datasets.tar.gz
python3 -m dfss --url=open@sophgo.com:sophon-demo/VITA1_5/vita_processor.tar.gz
tar xvf vita_processor.tar.gz && rm vita_processor.tar.gz

popd