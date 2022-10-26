#!/bin/bash
pip3 install dfn

scripts_dir=$(dirname $(readlink -f "$0"))
root_dir=$scripts_dir/..
data_dir=$root_dir/data

pushd $scripts_dir

model_dir=$data_dir/models
torch_dir=$model_dir/torch
if [ ! -d "$model_dir" ]; then
  echo "create model dir: $model_dir"
  mkdir -p $model_dir
fi

if [ ! -d "$torch_dir" ]; then
  echo "create torch dir: $torch_dir"
  mkdir -p $torch_dir
fi

python3 -m dfn --url http://219.142.246.77:65000/sharing/XW7bx6Axu
mv yolact_base_54_800000.pth $torch_dir

python3 -m dfn --url http://219.142.246.77:65000/sharing/DdpiMXolS
mv yolact_base_54_800000.trace.pt $torch_dir

bm1684_dir=$model_dir/BM1684
if [ ! -d "$bm1684_dir" ]; then
  echo "create torch dir: $bm1684_dir"
  mkdir -p $bm1684_dir
fi
# 1b
python3 -m dfn --url http://219.142.246.77:65000/sharing/2KbJDiRGL
mv yolact_base_54_800000_fp32_1b.bmodel $bm1684_dir
# 4b
python3 -m dfn --url http://219.142.246.77:65000/sharing/zHTjlC0On
mv yolact_base_54_800000_fp32_4b.bmodel $bm1684_dir

bm1684x_dir=$model_dir/BM1684X

if [ ! -d "$bm1684x_dir" ]; then
  echo "create torch dir: $bm1684x_dir"
  mkdir -p $bm1684x_dir
fi
#1b
python3 -m dfn --url http://219.142.246.77:65000/sharing/0fxjyMBIP
mv yolact_base_54_800000_fp32_1b.bmodel $bm1684x_dir
#4b
python3 -m dfn --url http://219.142.246.77:65000/sharing/kmAgh24DC
mv yolact_base_54_800000_fp32_4b.bmodel $bm1684x_dir

popd


