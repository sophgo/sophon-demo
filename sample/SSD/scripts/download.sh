#!/bin/bash
res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi
scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir
# sudo apt-get install zip
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
pushd $scripts_dir
python3 -m dfss --url=open@sophgo.com:sophon-demo/SSD/data.zip
unzip data.zip -d ../
rm data.zip
echo "All done!"
popd
