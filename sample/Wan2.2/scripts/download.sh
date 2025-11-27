#!/bin/bash

pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/SC11-FP300/v1.11_RC2/RC2/AI-Toolchain/vLLM/20251114_164933/docker-soph_vllm-0.7.3-20251114-2e82aebe-350d5894.tar.bz2
python3 -m dfss --url=open@sophgo.com:/SC11-FP300/v1.11_RC2/RC2/AI-Toolchain/Troch-TPU/20251114_162621/torch-tpu_20251114_350d5894.tar.gz
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Wan2.2/wan_assets_examples.tgz

mkdir ../packages
mv docker-soph_vllm-0.7.3-20251114-2e82aebe-350d5894.tar.bz2 ../packages/
mv torch-tpu_20251114_350d5894.tar.gz ../packages/

tar -zxvf ./wan_assets_examples.tgz -C ../python/
rm docker-soph_vllm-0.7.3-20251114-2e82aebe-350d5894.tar.bz2
rm torch-tpu_20251114_350d5894.tar.gz