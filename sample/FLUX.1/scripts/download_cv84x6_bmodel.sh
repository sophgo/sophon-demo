#!/bin/bash
# 下载 CV84X6 平台的 bmodel (clip F16 + w4bf16_t5 + schnell_w4bf16_transformer + tiny_vae BF16)
pip3 install dfss --upgrade

scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

# models
if [ ! -d "../models/CV84X6/" ];
then
    mkdir -p ../models/CV84X6/
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/CV84X6_bmodels.zip
    unzip CV84X6_bmodels.zip -d ../models/CV84X6/
    rm CV84X6_bmodels.zip

    echo "CV84X6 bmodels download!"
else
    echo "models exists!"
fi

# ids_emb (与 BM1684X 共用 1024 版本)
if [ ! -f "../models/ids_emb_1024.pt" ];
then
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/ids_emb_1024.pt
    mv ids_emb_1024.pt ../models/
    echo "ids_emb_1024 download!"
else
    echo "ids_emb exists!"
fi

# tokenizer (与 BM1684X 共用)
if [ ! -d "../models/tokenizer" ] || [ ! -d "../models/tokenizer_2" ];
then
    mkdir -p ../models/tokenizer
    mkdir -p ../models/tokenizer_2
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/tokenizer.zip
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/tokenizer_2.zip
    unzip tokenizer.zip -d ../models/
    rm tokenizer.zip
    unzip tokenizer_2.zip -d ../models/
    rm tokenizer_2.zip

    echo "tokenizer download!"
else
    echo "tokenizer exists!"
fi
popd
