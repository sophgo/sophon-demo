#!/bin/bash
# ==============================================================================
# FunASR Nano-2512 — Model & Dataset Download
# ==============================================================================
set -e

res=$(which unzip)
if [ $? != 0 ]; then
    echo "Please install unzip on your system!"
    echo "  sudo apt install unzip"
    exit
fi

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --quiet

scripts_dir=$(dirname $(readlink -f "$0"))
pushd $scripts_dir

# -------------------------------------------------------------------
# 1. Datasets (aishell_S0764, shared with WeNet)
# -------------------------------------------------------------------
if [ ! -d "../datasets/aishell_S0764" ]; then
    mkdir -p ../datasets
    echo "Downloading aishell_S0764 test data..."
    python3 -m dfss --url=open@sophgo.com:sophon-demo/WeNet/datasets/aishell_S0764.zip
    unzip -q aishell_S0764.zip -d ../datasets
    rm aishell_S0764.zip
    echo "Datasets download OK!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# -------------------------------------------------------------------
# 2. BModel (BM1684X F16 + LLM w4bf16, for SE7-32)
# -------------------------------------------------------------------
if ! ls ../models/BM1684X/*.bmodel >/dev/null 2>&1; then
    mkdir -p ../models
    echo "Downloading BM1684X F16 + LLM bmodels..."
    python3 -m dfss --url=open@sophgo.com:sophon-demo/FunASR_Nano/BM1684X_F16.tar.gz
    tar xzf BM1684X_F16.tar.gz -C ../models/
    rm BM1684X_F16.tar.gz
    echo "BM1684X bmodels download OK!"
else
    echo "BM1684X bmodels exist! Remove them if you need to update."
fi

# -------------------------------------------------------------------
# 2b. BModel (BM1688 F16 + LLM w4bf16, for SE9-16)
# -------------------------------------------------------------------
if ! ls ../models/BM1688/*.bmodel >/dev/null 2>&1; then
    mkdir -p ../models
    echo "Downloading BM1688 F16 + LLM bmodels..."
    python3 -m dfss --url=open@sophgo.com:sophon-demo/FunASR_Nano/BM1688_F16.tar.gz
    tar xzf BM1688_F16.tar.gz -C ../models/
    rm BM1688_F16.tar.gz
    echo "BM1688 bmodels download OK!"
else
    echo "BM1688 bmodels exist! Remove them if you need to update."
fi

# -------------------------------------------------------------------
# 2b. BModel (BM1684X2 F16, only F16 — firmware does not support FP32)
# -------------------------------------------------------------------
if ! ls ../models/BM1684X2/*.bmodel >/dev/null 2>&1; then
    mkdir -p ../models
    echo "Downloading BM1684X2 F16 bmodels..."
    python3 -m dfss --url=open@sophgo.com:sophon-demo/FunASR_Nano/BM1684X2_F16.tar.gz
    tar xzf BM1684X2_F16.tar.gz -C ../models/
    rm BM1684X2_F16.tar.gz
    echo "BM1684X2 bmodels download OK!"
else
    echo "BM1684X2 bmodels exist! Remove them if you need to update."
fi

# -------------------------------------------------------------------
# 3. ONNX models (for recompilation)
# -------------------------------------------------------------------
if ! ls ../models/onnx/*.onnx >/dev/null 2>&1; then
    mkdir -p ../models
    echo "Downloading ONNX models..."
    python3 -m dfss --url=open@sophgo.com:sophon-demo/FunASR_Nano/onnx.tar.gz
    tar xzf onnx.tar.gz -C ../models/
    rm onnx.tar.gz
    echo "ONNX models download OK!"
else
    echo "ONNX models exist! Remove them if you need to update."
fi

popd

# -------------------------------------------------------------------
# 3b. Copy LLM tokenizer config to python/config/ (for the infer script)
# -------------------------------------------------------------------
for chip in BM1684X BM1688; do
    if [ -d "../models/$chip/config" ] && [ ! -f "../python/config/tokenizer.json" ]; then
        mkdir -p ../python/config
        cp ../models/$chip/config/* ../python/config/ 2>/dev/null
        echo "Copied $chip LLM tokenizer config to python/config/"
        break
    fi
done

# -------------------------------------------------------------------
# 4. Note about PyTorch model (only needed for recompilation)
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo "Note: The precompiled BModels already include the"
echo "Qwen3-0.6B LLM (w4bf16) running on TPU. You only need the"
echo "FunASR-Nano PyTorch weights if you want to recompile:"
echo "  - export ONNX (encoder/adapter): tools/export_onnx.py"
echo "  - extract LLM for llm_convert:    tools/extract_llm_weights.py"
echo ""
echo "To pre-download the PyTorch model:"
echo "  python3 -c \"from funasr import AutoModel; \\"
echo "      AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512',"
echo "               trust_remote_code=True)\""
echo "============================================"
