#!/bin/bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

# datasets
if [ ! -d "../datasets" ]; 
then
    pushd ../
    python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/datasets.tar.gz
    tar xvf datasets.tar.gz && rm datasets.tar.gz
    popd
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
mkdir -p ../models
pushd ../models
python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v5/BM1688.tar.gz
tar xvf BM1688.tar.gz && rm BM1688.tar.gz
python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v5/CV186X.tar.gz
tar xvf CV186X.tar.gz && rm CV186X.tar.gz
python3 -m dfss --url=open@sophgo.com:sophon-demo/PP-OCR/models_v5/sophon-demo/PP-OCR/models_v5/ppocrv5_dict.txt
mv ppocrv5_dict.txt ../datasets/
popd

popd