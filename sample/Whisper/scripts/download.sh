#!/bin/bash
res=$(which unzip)

if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

TARGET="BM1684X"
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            model="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        *)
            # Unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ ! -d "../models/$TARGET" ];
then
    mkdir -p ../models/$TARGET
fi
python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/models/${TARGET}/bmwhisper_${model}_1684x_f16.bmodel
mv bmwhisper_${model}_1684x_f16.bmodel ../models/$TARGET

# datasets
if [ ! -d "../datasets" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/datasets_240327/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip

    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# assets
if [ ! -d "../python/bmwhisper/assets" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/model_240408/assets.zip
    unzip assets.zip -d ../python/bmwhisper
    rm assets.zip
    echo "assets download!"
else
    echo "Assets folder exist! Remove it if you need to update."
fi
popd