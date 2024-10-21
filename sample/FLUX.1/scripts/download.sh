#!/bin/bash
pip3 install dfss --upgrade

res=$(which 7z)
if [ $? != 0 ];
then
    echo "Please install 7z on your system!"
    echo "To install, use the following command:"
    echo "sudo apt install p7zip-full"
    exit
fi

scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

flux_type="dev"
quantize="W4BF16"
chip_type=BM1684X
use_taef1=1

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in 
    --flux_type)
        flux_type="$2"
        shift 2
        ;;
    --quantize)
        quantize="$2"
        shift 2
        ;;
    --chip_type)
        chip_type="$2"
        shift 2
        ;;
    --use_taef1)
        use_taef1="$2"
        shift 2
        ;;
    *)
        shift
        ;;
    esac
done

# models
if [ ! -d "../models/${chip_type}/" ];
then
    mkdir -p ../models/${chip_type}/
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/clip.bmodel
    mv clip.bmodel ../model/${chip_type}
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/w4bf16_t5.bmodel
    mv w4bf16_t5.bmodel../model/${chip_type}
    echo "clip/t5 bmodels download!"
else
    echo "models exists!"
fi

if [ "$flux_type" == "schnell" ] && [ "$quantize" == "W4BF16" ]; then
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/schnell_w4bf16_transformer.bmodel
    mv schnell_w4bf16_transformer.bmodel "../models/${chip_type}/"
elif [ "$flux_type" == "schnell" ] && [ "$quantize" == "BF16" ]; then
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/schnell_bf16.7z
    7z x schnell_bf16.7z -o ../models/${chip_type}
    rm schnell_bf16.7z
elif [ "$flux_type" == "dev" ] && [ "$quantize" == "W4BF16" ]; then
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/dev_w4bf16_transformer.bmodel
    mv dev_w4bf16_transformer.bmodel "../models/${chip_type}/"
elif [ "$flux_type" == "dev" ] && [ "$quantize" == "BF16" ]; then
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/dev_bf16.7z
    7z x dev_bf16.7z -o ../models/${chip_type}
    rm dev_bf16.7z
fi

if [ "$use_taef1" == 1 ]; then
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/tiny_vae_decoder_bf16.bmodel
    mv tiny_vae_decoder_bf16.bmodel ../model/${chip_type}
else
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/vae_decoder_bf16.bmodel
    mv vae_decoder_bf16.bmodel ../model/${chip_type}
fi
echo "flux/vae bmodels download!"

# tokenizer
if [ ! -d "../models/tokenizer" ] || [ ! -d "../models/tokenizer_2" ] ; 
then
    mkdir -p ../models/tokenizer
    mkdir -p ../models/tokenizer_2
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/tokenizer.zip
    python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/tokenizer_2.zip

    7z x tokenizer.zip -o../models/
    rm tokenizer.zip

    7z x tokenizer_2.zip -o../models/
    rm tokenizer_2.zip

    echo "tokenizer download!"
else
    echo "tokenizer exists!"
fi

# ids_emb.pt
python3 -m dfss --url=open@sophgo.com:/sophon-demo/FLUX_1/ids_emb.pt
mv ids_emb.pt ../models

popd