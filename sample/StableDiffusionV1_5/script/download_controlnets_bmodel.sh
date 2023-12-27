#!/bin/bash
pip3 install dfss

res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    echo "To install, use the following command:"
    echo "sudo apt install unzip"
    exit
fi

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir

# controlnets
if [ ! -d "../models/BM1684X/controlnets/" ]; 
then
    mkdir -p ../models/BM1684X/controlnets/
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_v1_5/controlnets.zip
    unzip controlnets.zip -d ../models/BM1684X/controlnets/
    rm controlnets.zip

    echo "controlnets download!"
else
    echo "controlnets exist!"
fi

if [ ! -d "../models/BM1684X/processors/" ]; 
then
    mkdir -p ../models/BM1684X/processors/
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_v1_5/processors.zip
    unzip processors.zip -d ../models/BM1684X/processors/
    rm processors.zip

    echo "processors download!"
else
    echo "processors exists!"
fi

popd