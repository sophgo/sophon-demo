#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

function download_bm1684x {
    if [ ! -d "../models/BM1684X" ]; then
        mkdir -p ../models/BM1684X
    fi
    pushd ../models/BM1684X

    if [ x"$1" == x"qwen" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen/qwen-7b_int4_seq512_1dev.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen/qwen-7b_int4_seq2048_1dev.bmodel 
    elif [ x"$1" == x"qwen1.5" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen1.5/qwen1.5-7b_int4_seq512_1dev.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen1.5/qwen1.5-7b_int4_seq2048_1dev.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen1.5/qwen1.5-7b_int4_seq4096_2dev_dyn.bmodel 
    elif [ x"$1" == x"qwen2" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen2/qwen2-7b_int4_seq512_1dev.bmodel
    elif [ x"$1" == x"qwen2.5" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen2.5/qwen2.5-1.5b_int4_seq512_1dev.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen2.5/qwen2.5-1.5b_int4_seq1024_1dev.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen2.5/qwen2.5-0.5b_int4_seq1024_1dev.bmodel 
    elif [ x"$1" == x"qwen3" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen3/qwen3-4b_int4_seq512_1dev.bmodel
    elif [ x"$1" == x"deepseek-r1-distill-qwen2" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/deepseek-r1-distill-qwen2-1.5b.tar.gz
        tar xvf deepseek-r1-distill-qwen2-1.5b.tar.gz && rm deepseek-r1-distill-qwen2-1.5b.tar.gz
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/deepseek-r1-distill-qwen2-7b.tar.gz
        tar xvf deepseek-r1-distill-qwen2-7b.tar.gz && rm deepseek-r1-distill-qwen2-7b.tar.gz
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/deepseek-r1-distill-qwen2-14b.tar.gz
        tar xvf deepseek-r1-distill-qwen2-14b.tar.gz && rm deepseek-r1-distill-qwen2-14b.tar.gz
    elif [ x"$1" == x"qwq-32b" ]; then
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwq/qwq-32b_int4_seq2048_2dev.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwq/qwq-32b_int4_seq2048_4dev.bmodel
        python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwq/qwq-32b_int4_seq2048_8dev.bmodel
    else
        echo "invalid model name"
    fi
    popd
}

function download_bm1688 {
    if [ ! -d "../models/BM1688" ]; then
        mkdir -p ../models/BM1688
    fi
    pushd ../models/BM1688
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen1.5/qwen1.5-1.8b_int4_seq512_bm1688_1dev_2core.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/1688/qwen2.5-1.5b_int4_seq512_1688_2core.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/1688/qwen2.5-1.5b_int4_seq1024_1688_2core.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/1688/qwen2.5-1.5b_int4_seq2048_1688_2core.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/1688/deepseek-r1-distill-qwen-1.5b_int4_seq1024_1688_2core.bmodel
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/1688/deepseek-r1-distill-qwen-7b_int4_seq1024_1688_2core.bmodel    
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/tokenizer_deepseek_r1_distill_qwen2.tgz
    tar -xzvf tokenizer_deepseek_r1_distill_qwen2.tgz
    rm -f tokenizer_deepseek_r1_distill_qwen2.tgz
    popd
}

function download_cv186x {
    if [ ! -d "../models/CV186X" ]; then
        mkdir -p ../models/CV186X
    fi
    pushd ../models/CV186X
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/qwen1.5/qwen1.5-1.8b_int4_seq512_cv186x_1dev.bmodel
    popd
}

if [ "$1" == "bm1688" ]; then
    download_bm1688
elif [ "$1" == "cv186x" ]; then
    download_cv186x
else
    download_bm1684x $1
fi
