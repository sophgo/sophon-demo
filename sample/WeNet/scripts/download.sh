#!/bin/bash
pip3 install dfn
# sudo apt install unzip

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir
 
pushd $scripts_dir
# datasets
if [ ! -d "../datasets" ]; 
then
    mkdir -p ../datasets
    # aishell data
    python3 -m dfn --url http://219.142.246.77:65000/sharing/JigSE5PuB
    unzip aishell_S0764.zip -d ../datasets/
    rm aishell_S0764.zip
 
    echo "datasets download!"
else
    echo "datasets exist!"
fi
 
# models
if [ ! -d "../models" ]; 
then
    python3 -m dfn --url http://219.142.246.77:65000/sharing/COB3NbyUj
    unzip models.zip -d ../
    rm models.zip
    echo "models download!"
else
    echo "models exist!"
fi

# swig decoders module
if [ ! -d "../python/swig_decoders" ]; 
then
    if [ "$1" = "soc" ];
    then
        python3 -m dfn --url http://219.142.246.77:65000/sharing/FBii7gQRZ
        unzip swig_decoders_aarch64.zip -d ../python/
        mv ../python/swig_decoders_aarch64 ../python/swig_decoders
        rm swig_decoders_aarch64.zip
    elif [ "$1" = "pcie" ];
    then
        python3 -m dfn --url http://219.142.246.77:65000/sharing/N6GDGWyOg
        unzip swig_decoders_x86_64.zip -d ../python/
        mv ../python/swig_decoders_x86_64 ../python/swig_decoders
        rm swig_decoders_x86_64.zip
    else
        echo "Please set the platform. Option: soc and pcie"
        exit
    fi

    echo "swig decoders module download!"
else
    echo "swig decoders module exist!"
fi
popd