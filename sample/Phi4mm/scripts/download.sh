#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir/../python/

python3 -m dfss --url=open@sophgo.com:sophon-demo/Phi4mm/phi4mm_bm1684x_int4_1core.bmodel
python3 -m dfss --url=open@sophgo.com:sophon-demo/Phi4mm/processor.tar.gz
tar xvf processor.tar.gz && rm processor.tar.gz

popd