#!/bin/bash
pip3 install dfss --upgrade

res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    echo "To install, use the following command:"
    echo "sudo apt install unzip"
    exit
fi

scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

# 创建存放模型的目录
mkdir models
mkdir -p models/BM1690
# 下载tokenizer
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_3/tokenizer.zip
unzip tokenizer.zip -d ./models/
rm tokenizer.zip
# 下载所需的模型，已有的不用再下载，可注释掉
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_3/BM1690/clip_l.bmodel
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_3/BM1690/clip_g.bmodel
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_3/BM1690/t5.bmodel
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_3/BM1690/mmdit.bmodel
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_3/BM1690/vae_decoder.bmodel
mv *bmodel ./models/BM1690/

popd