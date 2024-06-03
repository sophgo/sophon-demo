#!/bin/bash
res=$(which unzip)
if [ $? != 0 ]; then
    echo "Please install unzip on your system!"
    echo "Please run the following command: sudo apt-get install unzip"
    exit
fi
echo "unzip is installed in your system!"

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

# 检查 MiniCPM-2B-sft-bf16.zip 是否存在
if [ ! -d "../tools/MiniCPM-2B-sft-bf16" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/MiniCPM-2B-sft-bf16.zip
    unzip MiniCPM-2B-sft-bf16.zip -d ../tools/
    rm MiniCPM-2B-sft-bf16.zip
    echo "MiniCPM-2B-sft-bf16 download!"
else
    echo "tools/MiniCPM-2B-sft-bf16 folder exist! Remove it if you need to update."
fi

if [ ! -d "../models" ]; then
    mkdir -p ../models
fi

# 检查 BM1688文件夹 是否存在
if [ ! -d "../models/BM1688" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/BM1688.zip
    unzip BM1688.zip -d ../models/
    rm BM1688.zip
    echo "BM1688 download!"
else
    echo "BM1688 folder exist! Remove it if you need to update."
fi

# 检查 BM1684X文件夹 是否存在
if [ ! -d "../models/BM1684X" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/BM1684X.zip
    unzip BM1684X.zip -d ../models/
    rm BM1684X.zip
    echo "BM1684X download!"
else
    echo "BM1684X folder exist! Remove it if you need to update."
fi

# 检查 lib_pcie文件夹 是否存在
if [ ! -d "../cpp/lib_pcie" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/lib_pcie.zip
    unzip lib_pcie.zip -d ../cpp/
    rm lib_pcie.zip
    echo "lib_pcie download!"
else
    echo "lib_pcie folder exist! Remove it if you need to update."
fi

# 检查 lib_soc_bm1684x文件夹 是否存在
if [ ! -d "../cpp/lib_soc_bm1684x" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/lib_soc_bm1684x.zip
    unzip lib_soc_bm1684x.zip -d ../cpp/
    rm lib_soc_bm1684x.zip
    echo "lib_soc_bm1684x download!"
else
    echo "lib_soc_bm1684x folder exist! Remove it if you need to update."
fi

# 检查 lib_soc_bm1688文件夹 是否存在
if [ ! -d "../cpp/lib_soc_bm1688" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/lib_soc_bm1688.zip
    unzip lib_soc_bm1688.zip -d ../cpp/
    rm lib_soc_bm1688.zip
    echo "lib_soc_bm1688 download!"
else
    echo "lib_soc_bm1688 folder exist! Remove it if you need to update."
fi

# 检查 token_config文件夹 是否存在
if [ ! -d "../cpp/token_config" ]; 
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPM/token_config.zip
    unzip token_config.zip -d ../cpp/
    rm token_config.zip
    echo "token_config download!"
else
    echo "token_config folder exist! Remove it if you need to update."
fi

popd

