#!/bin/bash
pip3 install dfn

res=$(dpkg -l|grep unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir

mkdir -p ../data/images
mkdir -p ../data/videos
mkdir -p ../data/images/face

# 测试集WIDER FACE
python3 -m dfn --url http://219.142.246.77:65000/sharing/4ryx2MOgm
tar -xf WIDERVAL.tar -C ../data/images/
rm WIDERVAL.tar

## 单张图片与视频
python3 -m dfn --url http://219.142.246.77:65000/sharing/BaQm9vdpp
tar -xf test.tar -C ../data
mv ../data/images/face0*.jpg ../data/images/face
rm test.tar

# models
# 包含原始pytorch模型、traced pytorch模型、fp32bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/BmC4MEfEl
unzip models.zip -d ../data/
rm models.zip


popd
