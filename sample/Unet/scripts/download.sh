#!/bin/bash
pip3 install dfn

scripts_dir=$(dirname $(readlink -f "$0"))
echo $scripts_dir

pushd $scripts_dir

mkdir -p ../models/torch
mkdir -p ../models/BM1684
mkdir -p ../models/BM1684X

python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/u2Y3bx850
mv unet_carvana_scale0.5_epoch2.pth ../models/torch
echo "[Success] unet_carvana_scale0.5_epoch2.pth has been downloaded to path ../models/torch"

python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/X3IyNmoar
mv unet.pt ../models/torch
echo "[Success] unet.pt has been downloaded to path ../models/torch"

python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/tfz6xHK3s
mv unetBM1684.bmodel ../models/BM1684/unet_fp32_1b.bmodel
echo "[Success] BM1684/unet_fp32_b1.bmodel has been downloaded to path ../models"

python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/l0Io92JBO
mv unet1684X.bmodel ../models/BM1684X/unet_fp32_1b.bmodel
echo "[Success] BM1684X/unet_fp32_b1.bmodel has been downloaded to path ../models"

python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/8QitHtqqJ
mv carvana_video.mp4 ../datasets/carvana_video.mp4
echo "[Success] carvana_video.mp4 has been downloaded to path ../datasets"

python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/YlYgQKXqr
unzip carvana.zip -d ../datasets/
mv ../datasets/carvana ../datasets/test
echo "[Success] carvana.zip has been unzipped to path ../datasets/test/"

python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/2XzlJhbtu
unzip carvana_masks.zip -d ../datasets/
mv ../datasets/carvana_masks ../datasets/label
echo "[Success] carvana_masks has been unzipped to path ../datasets/label/"

popd
