#!/bin/bash
# SAM3 模型与数据集下载脚本
# 用法：chmod +x scripts/download.sh && ./scripts/download.sh
#
# 通过 dfss 从 sophon-demo 文件服务器下载预编译 bmodel、onnx 及测试数据：
#   - BM1684X_504.zip   : BM1684X(SE7等) FP16 bmodel（ViT 5part + Neck + Grounding enc/dec + Text enc）
#   - BM1688_504.zip    : BM1688(SE9等) FP16 bmodel（同构，单核）
#   - onnx_504.zip      : 504×504 ViT 5 part + Neck 的 ONNX（含外置 .data，用于自行编译 bmodel）
#   - onnx_grounding_504.zip : Grounding Encoder + Decoder ONNX（用于自行编译）
#   - postproc_weights.zip : 后处理权重 post_process_weights.npz + seg_head_weights.npz
#       （bmodel 推理必需，从 sam3.pt 一次性提取的 head 权重，随交付集下发后推理自包含）
#   - datasets.zip      : truck.jpg / groceries.jpg / dog.jpg 测试图
# 注：bmodel 推理只需上述 bmodel + postproc_weights，不需要 sam3.pt。
#     sam3.pt 原始 PyTorch 权重仅"自行重新导出 onnx"时需要（HuggingFace 申请下载，见 README 第 4 节）。

res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --quiet
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

# datasets
if [ ! -d "../datasets" ];
then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM3/datasets.zip
    unzip datasets.zip -d ../
    rm datasets.zip
    echo "datasets download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi

# models
if [ ! -d "../models" ];
then
    mkdir ../models

    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM3/BM1684X_504.zip
    unzip BM1684X_504.zip -d ../models
    rm BM1684X_504.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM3/BM1688_504.zip
    unzip BM1688_504.zip -d ../models
    rm BM1688_504.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM3/onnx_504.zip
    unzip onnx_504.zip -d ../models
    rm onnx_504.zip

    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM3/onnx_grounding_504.zip
    unzip onnx_grounding_504.zip -d ../models
    rm onnx_grounding_504.zip

    # 后处理权重（bmodel 推理必需，从 sam3.pt 提取的 head 权重）
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM3/postproc_weights.zip
    unzip postproc_weights.zip -d ../models
    rm postproc_weights.zip

    echo "models download!"
else
    echo "Models folder exist! Remove it if you need to update."
fi

# 1008×1008 INT8 bmodel（PCIe / SoC 流式，可选）
# 默认不下（540MB，仅 1008 ViT+Neck 场景需要）。要 1008 int8 时：
#   DOWNLOAD_1008_INT8=1 ./scripts/download.sh
# 含 ViT Part1-4 int8 + Part0/Neck f16 回退（int8 推理路径所需）。
if [ "${DOWNLOAD_1008_INT8:-0}" = "1" ] && [ ! -f "../models/BM1684X/vit/sam3_vit_part1_int8_1b.bmodel" ]; then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM3/BM1684X_1008_int8.zip
    unzip BM1684X_1008_int8.zip -d ../models
    rm BM1684X_1008_int8.zip
    echo "1008 int8 models download!"
fi

# BM1688(SE9 等) 1008 int8 bmodel（SoC 流式，可选）
#   DOWNLOAD_1008_INT8_BM1688=1 ./scripts/download.sh
# 已在 SE9-16(BM1688 SoC) 实测：6 bmodel bmrt_test 全过 + 流式 e2e 跑通，
# 峰值显存 1.35GB/part < SoC 3GB（见 docs/export_bmodel.md §3.3）。
if [ "${DOWNLOAD_1008_INT8_BM1688:-0}" = "1" ] && [ ! -f "../models/BM1688/vit/sam3_vit_part1_int8_1b.bmodel" ]; then
    python3 -m dfss --url=open@sophgo.com:sophon-demo/SAM3/BM1688_1008_int8.zip
    unzip BM1688_1008_int8.zip -d ../models
    rm BM1688_1008_int8.zip
    echo "BM1688 1008 int8 models download!"
fi

popd
