#!/bin/bash
# DINOv2 模型与数据集下载脚本
#
# 用法：
#   bash scripts/download.sh          # 首次下载；已有完整目录时跳过
#   bash scripts/download.sh --force  # 强制重新下载并覆盖（升级版本时用）
#
# 从 sophon-demo 文件服务器（dfss）下载预编译 bmodel、ONNX 与测试数据：
#   - datasets.zip : imagenet_val_1k（每类 1 张共 1000 张 + label.txt）、
#                    cali_data（INT8 标定 npz 200 张）、test_input、
#                    imagenet_synset_to_idx.json、imagenet_classes.txt、
#                    dinov2_vitb14_reg4_linear_head.npz/.bin（线性头权重）
#   - onnx_224.zip : dinov2_vitb14_reg_224_1b.onnx（自行编译 bmodel 用）
#   - BM1684X.zip / BM1688.zip / CV186X.zip : 各芯片预编译 bmodel（F32/F16/INT8；
#     BM1688 另含 2core 双核版本）
#
# 注意：DINOv2 原始 PyTorch 权重（backbone 与官方线性头 .pth）按 Meta 的
# DINOv2 License 申请下载，**不随 dfss 分发**；仅在需要自行导出 ONNX 时才用到，
# 见 docs/DINOv2_Export_Guide.md。

FORCE=0
if [ "$1" = "--force" ] || [ "$1" = "-f" ]; then
    FORCE=1
fi

res=$(which unzip)
if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit 1
fi

# 只在缺失时安装 dfss，避免在离线环境/每次运行都访问 PyPI
if ! python3 -c "import dfss" >/dev/null 2>&1; then
    pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet || {
        echo "ERROR: 无法安装 dfss（离线环境请先手动安装）" >&2
        exit 1
    }
fi
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir

# 检查一组文件是否齐全；有任一缺失返回 1
function all_exist()
{
    for f in "$@"; do
        [ -e "$f" ] || return 1
    done
    return 0
}

function fetch()
{
    local zip=$1
    local name=$(basename "$zip")
    python3 -m dfss --url=open@sophgo.com:sophon-demo/DINOv2/$name || return 1
    if [ ! -f "$name" ]; then
        echo "ERROR: $name download failed." >&2
        return 1
    fi
    return 0
}

# datasets
datasets_ok=0
all_exist ../datasets/imagenet_val_1k/label.txt \
          ../datasets/imagenet_val_1k/img \
          ../datasets/imagenet_synset_to_idx.json \
          ../datasets/imagenet_classes.txt \
          ../datasets/cali_data \
          ../datasets/test_input/dinov2_1b_in_f32.npz \
          ../datasets/dinov2_vitb14_reg4_linear_head.npz \
          ../datasets/dinov2_vitb14_reg4_linear_head.bin && datasets_ok=1

if [ $FORCE -eq 1 ] || [ $datasets_ok -eq 0 ];
then
    fetch datasets.zip || exit 1
    unzip -o datasets.zip -d ../
    rm datasets.zip
    echo "datasets download!"
else
    echo "Datasets folder is complete, skip (use --force to update)."
fi

# models
models_ok=0
all_exist ../models/onnx/dinov2_vitb14_reg_224_1b.onnx \
          ../models/BM1684X/dinov2_vitb14_reg_fp32_1b.bmodel \
          ../models/BM1684X/dinov2_vitb14_reg_fp16_1b.bmodel \
          ../models/BM1684X/dinov2_vitb14_reg_int8_1b.bmodel \
          ../models/BM1688/dinov2_vitb14_reg_fp32_1b.bmodel \
          ../models/BM1688/dinov2_vitb14_reg_fp16_1b.bmodel \
          ../models/BM1688/dinov2_vitb14_reg_int8_1b.bmodel \
          ../models/BM1688/dinov2_vitb14_reg_fp32_1b_2core.bmodel \
          ../models/BM1688/dinov2_vitb14_reg_fp16_1b_2core.bmodel \
          ../models/BM1688/dinov2_vitb14_reg_int8_1b_2core.bmodel \
          ../models/CV186X/dinov2_vitb14_reg_fp32_1b.bmodel \
          ../models/CV186X/dinov2_vitb14_reg_fp16_1b.bmodel \
          ../models/CV186X/dinov2_vitb14_reg_int8_1b.bmodel && models_ok=1

if [ $FORCE -eq 1 ] || [ $models_ok -eq 0 ];
then
    mkdir -p ../models

    fetch onnx_224.zip || exit 1
    unzip -o onnx_224.zip -d ../models
    rm onnx_224.zip

    fetch BM1684X.zip || exit 1
    unzip -o BM1684X.zip -d ../models/
    rm BM1684X.zip

    fetch BM1688.zip || exit 1
    unzip -o BM1688.zip -d ../models/
    rm BM1688.zip

    fetch CV186X.zip || exit 1
    unzip -o CV186X.zip -d ../models/
    rm CV186X.zip

    echo "models download!"
else
    echo "Models folder is complete, skip (use --force to update)."
fi

popd
