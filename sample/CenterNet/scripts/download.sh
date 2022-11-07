#!/bin/bash
pip3 install dfn

scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir

mkdir -p ../data/models/torch
mkdir -p ../cpp/centernet_bmcv/results
mkdir -p ../python/results/

# ctdet_coco_dlav0_1x.pth
python3 -m dfn --url http://219.142.246.77:65000/sharing/tUugBQuYz
mv ctdet_coco_dlav0_1x.pth ../data/models/torch
echo "[Success] ctdet_coco_dlav0_1x.pth has been downloaded to path ../data/models/torch"

# ctdet_coco_dlav0_1x.torchscript.pt
python3 -m dfn --url http://219.142.246.77:65000/sharing/8A233koXT
mv ctdet_coco_dlav0_1x.torchscript.pt ../data/models/torch
echo "[Success] ctdet_coco_dlav0_1x.torchscript.pt has been downloaded to path ../data/models/torch"

# centernet test data
python3 -m dfn --url http://219.142.246.77:65000/sharing/xcOwsXXjq
mv ctdet_test.jpg ../data
echo "[Success] ctdet_test.jpg has been downloaded to path ../data"

# coco_class.txt
python3 -m dfn --url http://219.142.246.77:65000/sharing/jc0abY3Dp
mv coco_classes.txt ../data
echo "[Success] coco_classes.txt has been downloaded to path ../data"

# BM1684/centernet_fp32_1b.bmodel
mkdir -p ../data/models/BM1684
python3 -m dfn --url http://219.142.246.77:65000/sharing/Qih18Ufdx
mv centernet_fp32_1b.bmodel ../data/models/BM1684
echo "[Success] centernet_fp32_1b.bmodel has been downloaded to path ../data/models/BM1684"

# BM1684/centernet_fp32_4b.bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/ohl7vaCNO
mv centernet_fp32_4b.bmodel ../data/models/BM1684
echo "[Success] centernet_fp32_4b.bmodel has been downloaded to path ../data/models/BM1684"

# BM1684/centernet_int8_1b.bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/xIK7oAFzF
mv centernet_int8_1b.bmodel ../data/models/BM1684
echo "[Success] centernet_int8_1b.bmodel has been downloaded to path ../data/models/BM1684"

# BM1684/centernet_int8_4b.bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/99ouJ04sW
mv centernet_int8_4b.bmodel ../data/models/BM1684
echo "[Success] centernet_int8_4b.bmodel has been downloaded to path ../data/models/BM1684"

# BM1684X/centernet_fp32_1b.bmodel
mkdir -p ../data/models/BM1684X
python3 -m dfn --url http://219.142.246.77:65000/sharing/Jh0xszHfx
mv centernet_fp32_1b.bmodel ../data/models/BM1684X
echo "[Success] centernet_fp32_1b.bmodel has been downloaded to path ../data/models/BM1684X"

# BM1684X/centernet_fp32_4b.bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/7es1n2rZV
mv centernet_fp32_4b.bmodel ../data/models/BM1684X
echo "[Success] centernet_fp32_4b.bmodel has been downloaded to path ../data/models/BM1684X"

# BM1684X/centernet_int8_1b.bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/GXW9dTr6G
mv centernet_int8_1b.bmodel ../data/models/BM1684X
echo "[Success] centernet_int8_1b.bmodel has been downloaded to path ../data/models/BM1684X"

# BM1684X/centernet_int8_4b.bmodel
python3 -m dfn --url http://219.142.246.77:65000/sharing/Wkm0bfk4t
mv centernet_int8_4b.bmodel ../data/models/BM1684X
echo "[Success] centernet_int8_4b.bmodel has been downloaded to path ../data/models/BM1684X"

# quantize dataset
python3 -m dfn --url http://219.142.246.77:65000/sharing/I0B7AODBO
tar zxvf images.tar.gz
rm images.tar.gz
rm -rf ../data/images
mv images ../data

popd