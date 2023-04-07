#!/bin/bash
pip3 install dfn
scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir

mkdir -p ../data/images
mkdir -p ../data/models/paddle

# images
python3 -m dfn --url http://219.142.246.77:65000/sharing/zMHZeauPP
tar -xvf ppocr_img.tar -C ../data/images
rm ppocr_img.tar

# ch_PP-OCRv2_det_infer
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/sny6uBdN9
tar -xvf ch_PP-OCRv2_det_infer.tar -C ../data/models/paddle
rm ch_PP-OCRv2_det_infer.tar

# ch_PP-OCRv2_rec_infer 
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/fbIAcI119
tar -xvf ch_PP-OCRv2_rec_infer.tar -C ../data/models/paddle
rm ch_PP-OCRv2_rec_infer.tar 

# ch_ppocr_mobile_v2.0_cls_infer
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/IxA5hKuLf
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar -C ../data/models/paddle
rm ch_ppocr_mobile_v2.0_cls_infer.tar


# # BM1684
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/mtCGZt05Y
tar -xvf BM1684.tar -C ../data/models
rm BM1684.tar

# # BM1684X
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/g5AIefVYU
tar -xvf BM1684X.tar -C ../data/models
rm BM1684X.tar 


echo "All done!"
popd