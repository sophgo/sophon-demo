#!/bin/bash

pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/SC11-FP300/v1.12_RC1/vLLM/20251208_105109/docker-soph_vllm-0.11.0-e8dfa38b-aea40143-c6f4d740.tar.bz2
python3 -m dfss --url=open@sophgo.com:/SC11-FP300/v1.12_RC1/AI-Toolchain/Torch-TPU/20251208_001503/torch-tpu_20251208_c6f4d740.tar.gz
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Wan2.2/wan_assets_examples.tgz

mkdir ../packages
mv docker-soph_vllm-0.11.0-e8dfa38b-aea40143-c6f4d740.tar.bz2 ../packages/
mv torch-tpu_20251208_c6f4d740.tar.gz ../packages/

tar -zxvf ./wan_assets_examples.tgz -C ../python/