#!/bin/bash
set -ex
pip3 install dfss --upgrade

res=$(which unzip)

if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi

scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

chip="bm1684x"

if [ "$1" == "bm1688" ]; then
    chip="bm1688"
    sed -i 's/BM1684X/BM1688/g' ../python/config.ini
else
    chip="bm1684x"
    sed -i 's/BM1688/BM1684X/g' ../python/config.ini
fi

# nltk_data & embedding model & reranker model are required
if [ ! -d "../nltk_data" ]; then
    echo "../nltk_data does not exist, download..."
    python3 -m dfss --url=open@sophgo.com:ezoo/chatdoc/nltk_data.zip
    unzip nltk_data.zip
    mv nltk_data ..
    rm nltk_data.zip
    echo "nltk_data download!"
else
    echo "../nltk_data already exist..."
fi

# download qwen1.5-7b
if [ ! -d "../models/BM1684X" ]; then
    mkdir -p ../models/BM1684X
    echo "download qwen1.5-7b as an example"
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen1.5/qwen1.5-7b_int4_seq2048_1dev.bmodel
    mv qwen1.5-7b_int4_seq2048_1dev.bmodel ../models/BM1684X
fi

# download qwen tokenizer
if [ ! -d "../models/qwen/token_config" ]; then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen1_5/token_config.zip
    unzip token_config.zip -d ../models/qwen/
    rm token_config.zip
fi

# download embedding model
if [[ "$chip" == "bm1684x" && ! -d "../models/BM1684X/bce_embedding" ]]; then
    echo "bce_embedding model does not exist, download..."
    python3 -m dfss --url=open@sophgo.com:ezoo/chatdoc/bce_embedding.zip
    unzip bce_embedding.zip -d ../models/BM1684X
    rm bce_embedding.zip
    echo "bce_embedding download!"
elif [[ "$chip" == "bm1688" && ! -d "../models/BM1688/bce_embedding" ]]; then
    echo "bce_embedding model does not exist, download..."
    python3 -m dfss --url=open@sophgo.com:sophon-demo/application/ChatDoc/bce_embedding_bm1688.zip
    unzip bce_embedding_bm1688.zip -d ../models/BM1688
    rm bce_embedding_bm1688.zip
    echo "bce_embedding download!"
else
    echo "embedding model already exist..."
fi

# download reranker model
if [[ "$chip" == "bm1684x" && ! -d "../models/BM1684X/bce_reranker" ]]; then
    echo "bce_reranker model does not exist, download..."
    python3 -m dfss --url=open@sophgo.com:ezoo/chatdoc/bce_reranker.zip
    unzip bce_reranker.zip -d ../models/BM1684X
    rm bce_reranker.zip
    echo "bce_reranker download!"
elif [[ "$chip" == "bm1688" && ! -d "../models/BM1688/bce_reranker" ]]; then
    echo "bce_reranker model does not exist, download..."
    python3 -m dfss --url=open@sophgo.com:sophon-demo/application/ChatDoc/bce_reranker_bm1688.zip
    unzip bce_reranker_bm1688.zip -d ../models/BM1688
    rm bce_reranker_bm1688.zip
    echo "bce_reranker download!"
else
    echo "bce reranker model already exist..."
fi

popd