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

# models
if [ ! -d "../models" ];
then
    mkdir ../models/BM1684X
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/streaming_s2t_BM1684X.zip
    unzip streaming_s2t_BM1684X.zip -d ../models/
    echo "bmodel download!"
    
    mkdir ../models/onnx
    python3 -m dfss --url=open@sophgo.com:test/seamless_static/seamless_streaming_unity_speech_encoder_frontend.zip
    unzip seamless_streaming_unity_speech_encoder_frontend.zip -d ../models/onnx
    rm seamless_streaming_unity_speech_encoder_frontend.zip
    python3 -m dfss --url=open@sophgo.com:test/seamless_static/seamless_streaming_unity_speech_encoder.zip
    unzip seamless_streaming_unity_speech_encoder.zip -d ../models/onnx
    rm seamless_streaming_unity_speech_encoder.zip
    python3 -m dfss --url=open@sophgo.com:test/seamless_static/seamless_communication_monotonic_text_decoder_frontend_step_equal_1.zip
    unzip seamless_communication_monotonic_text_decoder_frontend_step_equal_1.zip -d ../models/onnx
    rm seamless_communication_monotonic_text_decoder_frontend_step_equal_1.zip
    python3 -m dfss --url=open@sophgo.com:test/seamless_static/seamless_communication_monotonic_text_decoder_frontend_step_bigger_than_1.zip
    unzip seamless_communication_monotonic_text_decoder_frontend_step_bigger_than_1.zip -d ../models/onnx
    rm seamless_communication_monotonic_text_decoder_frontend_step_bigger_than_1.zip
    python3 -m dfss --url=open@sophgo.com:test/seamless_static/seamless_communication_monotonic_text_decoder_step_equal_1.zip
    unzip seamless_communication_monotonic_text_decoder_step_equal_1.zip -d ../models/onnx
    rm seamless_communication_monotonic_text_decoder_step_equal_1.zip
    python3 -m dfss --url=open@sophgo.com:test/seamless_static/seamless_communication_monotonic_text_decoder_step_bigger_than_1_64kvcache.zip
    unzip seamless_communication_monotonic_text_decoder_step_bigger_than_1_64kvcache.zip -d ../models/onnx
    rm seamless_communication_monotonic_text_decoder_step_bigger_than_1_64kvcache.zip
    python3 -m dfss --url=open@sophgo.com:test/seamless_static/seamless_streaming_monotonic_decoder_final_proj.zip
    unzip seamless_streaming_monotonic_decoder_final_proj.zip -d ../models/onnx
    rm seamless_streaming_monotonic_decoder_final_proj.zip
    echo "onnx models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi
popd