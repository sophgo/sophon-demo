#!/bin/bash
pip3 install dfn

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir
mkdir -p ../data/images/
# test
python3 -m dfn --url http://219.142.246.77:65000/sharing/zPdGaKdL4
tar -xf lprnet_test.tar -C ../data/images/
rm lprnet_test.tar
# test_label.json
python3 -m dfn --url http://219.142.246.77:65000/sharing/aVAD4ZI8k
mv test_label.json ../data/images/
# test_md5_lmdb
python3 -m dfn --url http://219.142.246.77:65000/sharing/ui69UstuD
tar -xf lprnet_test_md5_lmdb.tar -C ../data/images/
rm lprnet_test_md5_lmdb.tar

# Final_LPRNet_model.pth
mkdir -p ../data/models/torch
python3 -m dfn --url http://219.142.246.77:65000/sharing/xkBLp6HMm
mv Final_LPRNet_model.pth ../data/models/torch
# LPRNet_model.torchscript
python3 -m dfn --url http://219.142.246.77:65000/sharing/6GjUqWhii
mv LPRNet_model.torchscript ../data/models/torch/LPRNet_model_trace.pt

# BM1684/lprnet_fp32_1b.bmodel
mkdir -p ../data/models/BM1684
python3 -m dfn --url http://219.142.246.77:65000/sharing/6Lf9LgmwP
mv lprnet_fp32_1b.bmodel ../data/models/BM1684
# BM1684/lprnet_fp32_4b.bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/ZR0kqKPnM
mv lprnet_fp32_4b.bmodel ../data/models/BM1684
# BM1684/lprnet_int8_1b.bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/Cjw1Pn0Ow
mv lprnet_int8_1b.bmodel ../data/models/BM1684
# BM1684/lprnet_int8_4b.bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/OHuaDca1i
mv lprnet_int8_4b.bmodel ../data/models/BM1684

# BM1684X/lprnet_fp32_1b.bmodel
mkdir -p ../data/models/BM1684X
python3 -m dfn --url http://219.142.246.77:65000/sharing/lpl2B0V2n
mv lprnet_fp32_1b.bmodel ../data/models/BM1684X
# BM1684X/lprnet_fp32_4b.bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/MvU65WuCM
mv lprnet_fp32_4b.bmodel ../data/models/BM1684X
# BM1684X/lprnet_int8_1b.bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/LIKqrYJDS
mv lprnet_int8_1b.bmodel ../data/models/BM1684X
# BM1684X/lprnet_int8_4b.bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/aw8J0d4Dk
mv lprnet_int8_4b.bmodel ../data/models/BM1684X
popd