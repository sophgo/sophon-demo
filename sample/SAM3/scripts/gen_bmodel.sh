#!/bin/bash
# SAM3 BModel 一键编译脚本
# ONNX → MLIR → BModel，覆盖 504×504（SoC 交付集）与 1008×1008（PCIe）两套分辨率。
#
# 504×504（默认，SoC 交付集）：ViT 5-part + Neck FPN + Grounding Encoder/Decoder + Text Encoder
#   Part 0: patch_embed + pos_embed + ln_pre  (input: [1,3,504,504])
#   Part 1-4: blocks 0-7 / 8-15 / 16-23 / 24-31  (input: [1,36,36,1024])
#   Neck:  [1,1024,36,36]
#   Grounding Enc/Dec + Text Enc
#
# 1008×1008（PCIe）：ViT 5-part + Neck FPN（+ 可选 Grounding/Text，--grounding 开启）
#   Part 0: patch_embed + pos_embed + ln_pre  (input: [1,3,1008,1008])
#   Part 1-4: blocks 0-7 / 8-15 / 16-23 / 24-31  (input: [1,72,72,1024])
#   Neck:  [1,1024,72,72]
#   Grounding Enc/Dec（grid=72, N=5184）+ Text Enc：PCIe f16 增项，--grounding 开启
#
# Usage（在 tpu_mlir docker 内）：
#   cd /workspace/git_commits/developer/sophon-demo/sample/SAM3/scripts
#   ./gen_bmodel.sh --res 504  --chip bm1684x --mode f16   # 504 SoC 交付集（默认 res=504）
#   ./gen_bmodel.sh --res 504  --chip bm1688  --mode f16   # BM1688 单核，图同构
#   ./gen_bmodel.sh --res 1008 --chip bm1684x --mode f32   # 1008 PCIe（仅 ViT+Neck 骨干）
#   ./gen_bmodel.sh --res 1008 --chip bm1684x --mode f16 --grounding  # 1008 全流程（含 grounding 出框）

set -e

TPUMLIR_ROOT=/workspace/git_commits/tpu-mlir
if [ -f "$TPUMLIR_ROOT/envsetup.sh" ]; then
    source "$TPUMLIR_ROOT/envsetup.sh"
fi

script_dir=$(dirname $(readlink -f "$0"))

# 默认参数
res="504"
target="bm1684x"
target_dir="BM1684X"
mode="f16"
batch_size=1
do_grounding_flag=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --res)   res="${2}"; shift 2 ;;
        --chip)  target="${2,,}"; target_dir="${target^^}"; shift 2 ;;
        --mode)  mode="${2}"; shift 2 ;;
        --batch) batch_size="${2}"; shift 2 ;;
        --grounding) do_grounding_flag=1; shift ;;
        *) echo "Invalid option: $1" >&2; exit 1 ;;
    esac
done

# 按分辨率设定路径与形状
if [ "$res" = "504" ]; then
    GRID=36
    IMG=504
    VIT_PARTS=(1 2 3 4)          # 5-part: part0 + parts 1-4
    FEAT_SHAPE="[[$batch_size,$GRID,$GRID,1024]]"
    NECK_SHAPE="[[$batch_size,1024,$GRID,$GRID]]"
    outdir="../models/${target_dir}_504"
    onnx_dir="../models/onnx_504"
    onnx_gr_dir="../models/onnx_grounding_504"
    cali_dir="../datasets/cali_data_504"
    suffix="_504"
    DO_GROUNDING=1
elif [ "$res" = "1008" ]; then
    GRID=72
    IMG=1008
    VIT_PARTS=(1 2 3 4)          # 5-part: part0 + parts 1-4 (blocks 0-31)
    FEAT_SHAPE="[[$batch_size,$GRID,$GRID,1024]]"
    NECK_SHAPE="[[$batch_size,1024,$GRID,$GRID]]"
    outdir="../models/$target_dir"
    onnx_dir="../models/onnx"
    onnx_gr_dir="../models/onnx_grounding_1008"
    cali_dir="../datasets/cali_data_1008"
    suffix=""
    # 1008 grounding+text 为 PCIe f16 增项（--grounding 开启，默认 0=仅骨干）。
    # SoC 显存不够（encoder O(N²) 16× + ViT 顶 3GB），仅 PCIe f16 可用。
    DO_GROUNDING=${do_grounding_flag:-0}
else
    echo "Error: --res must be 504 or 1008 (got $res)" >&2; exit 1
fi

echo "=========================================="
echo "SAM3 BModel Compilation (${IMG}x${IMG})"
echo "  Chip:   $target"
echo "  Mode:   $mode"
echo "  Batch:  $batch_size"
echo "  Output: $outdir"
echo "=========================================="

# 注：outdir/cali_dir 是相对 scripts/ 的路径（pushd "$script_dir" 后使用），
# mkdir 必须在 pushd 之后做，否则按调用 cwd 建目录会错位（见 pushd 后 mkdir）。

# Helper: 生成测试输入 NPZ
gen_test_input() {
    local name="$1" shape="$2"
    local npz_file="$cali_dir/${name}.npz"
    if [ -f "$npz_file" ]; then echo "[SKIP] $npz_file"; return 0; fi
    echo "Generating test input: $npz_file"
    python3 -c "
import numpy as np
np.random.seed(42)
shape = $shape
data = np.random.randn(*shape).astype(np.float32) * 0.5
np.savez('$npz_file', data=data)
print(f'  shape={data.shape}, range=[{data.min():.3f}, {data.max():.3f}]')
"
}

# Helper: ONNX → MLIR
gen_mlir() {
    local model_name="$1" onnx_file="$2" input_shapes="$3" mlir_file="$4" test_npz="$5" input_types="${6:-float32}"
    echo ""
    echo "--- $model_name: ONNX → MLIR ---"
    model_transform.py \
        --model_name "$model_name" --model_def "$onnx_file" \
        --input_shapes "$input_shapes" --input_types "$input_types" \
        --mlir "$mlir_file" --test_input "$test_npz" --test_result "${model_name}_top_outputs.npz"
    echo "[OK] $mlir_file ($(ls -lh $mlir_file | awk '{print $5}'))"
}

# Helper: MLIR → BModel（int8 时先 run_calibration 产 cali_table 再 deploy）
gen_bmodel() {
    local mlir_file="$1" bmodel_name="$2" tolerance="${3:-0.99,0.99}" cali_subdir="$4"
    echo ""
    echo "--- $bmodel_name: MLIR → BModel ---"
    local deploy_args=(--mlir "$mlir_file" --quantize ${mode^^} --chip "$target" --num_core 1 \
        --model "$bmodel_name" --tolerance "$tolerance" --compare_all)
    if [ "$mode" = "int8" ]; then
        if [ -z "$cali_subdir" ]; then
            echo "[ERROR] int8 需要校准集但未指定 cali_subdir" >&2; exit 1
        fi
        local cali_npz_dir="$cali_dir/$cali_subdir"
        if [ ! -d "$cali_npz_dir" ] || [ -z "$(ls -A "$cali_npz_dir" 2>/dev/null)" ]; then
            echo "[ERROR] int8 校准集目录为空或不存在: $cali_npz_dir" >&2
            echo "        请先运行 tools/prepare_int8_cali_1008.py 生成校准集" >&2
            exit 1
        fi
        local cali_table="${bmodel_name%.bmodel}.cali_table"
        echo "  [cali] run_calibration → $cali_table (dataset=$cali_npz_dir, input_num=${CALI_INPUT_NUM:-30}, tune_num=${CALI_TUNE_NUM:-0})"
        # --tune_num 0: 跳过 tune（tune 对 0-dim 张量崩 cali_math.py:265，且 24 样本 KLD 已够）
        run_calibration.py "$mlir_file" --dataset "$cali_npz_dir" \
            --chip "$target" --input_num "${CALI_INPUT_NUM:-30}" \
            --tune_num "${CALI_TUNE_NUM:-0}" -o "$cali_table"
        deploy_args+=(--calibration_table "$cali_table")
        # --opt 1: part3/4 用默认 opt=2 时 address-assign 阶段 tpuc-opt abort
        # (LmemAllocator.cpp:1645 llvm_unreachable "shape_secs is not valid")，opt=1 绕过
        deploy_args+=(--opt 1)
    fi
    model_deploy.py "${deploy_args[@]}"
    echo "[OK] $bmodel_name ($(ls -lh $bmodel_name | awk '{print $5}'))"
}

# 测试输入
gen_test_input "vit_part0${suffix}" "(1, 3, $IMG, $IMG)"
gen_test_input "vit_feat${suffix}"  "(1, ${GRID}, ${GRID}, 1024)"
gen_test_input "neck_input${suffix}" "(1, 1024, $GRID, $GRID)"

pushd "$script_dir"

# 在 scripts/ cwd 下建输出目录（outdir/cali_dir 均相对 scripts/）
mkdir -p "$outdir/vit" "$outdir/neck" "$cali_dir"
[ "$DO_GROUNDING" = "1" ] && mkdir -p "$outdir/grounding"

echo ""
echo "=== Compiling SAM3 ViT for ${target^^} (${IMG}x${IMG}) ==="

# ViT Part 0
echo ""; echo "--- ViT Part 0 (preproc) ---"
gen_mlir "sam3_vit_part0${suffix}" "$onnx_dir/sam3_vit_part0.onnx" \
    "[[$batch_size,3,$IMG,$IMG]]" "sam3_vit_part0${suffix}.mlir" "$cali_dir/vit_part0${suffix}.npz"
gen_bmodel "sam3_vit_part0${suffix}.mlir" "sam3_vit_part0_${mode}_${batch_size}b.bmodel" \
    "0.99,0.99" "$([ "$res" = "1008" ] && echo part0)"
mv "sam3_vit_part0_${mode}_${batch_size}b.bmodel" "$outdir/vit/"

# ViT Part 1..N
for part in "${VIT_PARTS[@]}"; do
    echo ""; echo "--- ViT Part $part (blocks) ---"
    gen_mlir "sam3_vit_part${part}${suffix}" "$onnx_dir/sam3_vit_part${part}.onnx" \
        "$FEAT_SHAPE" "sam3_vit_part${part}${suffix}.mlir" "$cali_dir/vit_feat${suffix}.npz"
    gen_bmodel "sam3_vit_part${part}${suffix}.mlir" "sam3_vit_part${part}_${mode}_${batch_size}b.bmodel" \
        "0.99,0.99" "$([ "$res" = "1008" ] && echo part${part})"
    mv "sam3_vit_part${part}_${mode}_${batch_size}b.bmodel" "$outdir/vit/"
done

# Neck FPN
# NOTE: torch≥2.10 的新 ONNX 导出器会省略 ConvTranspose 的 kernel_shape 属性
# （可从权重推断），但 tpu-mlir OnnxConverter 要求显式 kernel_shape，否则
# KeyError。这里补上：从权重 shape 取 kernel_shape，存成 inline 的 _fixed.onnx。
echo ""; echo "--- Neck (FPN) ---"
NECK_ONNX="$onnx_dir/sam3_neck_combined.onnx"
NECK_FIXED="sam3_neck${suffix}_fixed.onnx"
python3 -c "
import onnx
from onnx import helper, numpy_helper
m = onnx.load('$NECK_ONNX', load_external_data=True)
inits = {i.name: i for i in m.graph.initializer}
fixed = 0
for n in m.graph.node:
    if n.op_type == 'ConvTranspose' and not any(a.name == 'kernel_shape' for a in n.attribute):
        ks = list(numpy_helper.to_array(inits[n.input[1]]).shape[2:])
        n.attribute.append(helper.make_attribute('kernel_shape', ks))
        fixed += 1
for init in m.graph.initializer:
    if init.external_data:
        init.ClearField('external_data')
        init.data_location = 0
onnx.save(m, '$NECK_FIXED')
print(f'  neck ConvTranspose kernel_shape fixup: {fixed} nodes → $NECK_FIXED (inline)')
"
gen_mlir "sam3_neck${suffix}" "$NECK_FIXED" \
    "$NECK_SHAPE" "sam3_neck${suffix}.mlir" "$cali_dir/neck_input${suffix}.npz"
gen_bmodel "sam3_neck${suffix}.mlir" "sam3_neck_${mode}_${batch_size}b.bmodel" \
    "0.99,0.99" "$([ "$res" = "1008" ] && echo neck)"
mv "sam3_neck_${mode}_${batch_size}b.bmodel" "$outdir/neck/"

# ============================================================
# Grounding + Text Encoder（504 交付集 / 1008 PCIe f16 增项）
# ============================================================
if [ "$DO_GROUNDING" = "1" ]; then

    # Optional: re-export grounding ONNX from the PyTorch checkpoint before
    # compiling. Off by default — the pre-exported ONNX in $onnx_gr_dir is
    # used. To force a fresh export, set GND_EXPORT=1 and SAM3_CKPT=<path>
    # (checkpoint must be visible from this environment). The export script
    # patches TransformerDecoder._get_coords to use int H,W so that
    # torch.arange traces as a constant (no dynamic Range op), and returns
    # a 4-ch reference_boxes matching the host post-processing interface.
    if [ "${GND_EXPORT:-0}" = "1" ] && [ -n "${SAM3_CKPT:-}" ]; then
        echo ""; echo "=== Re-export Grounding ONNX ==="
        python3 "$script_dir/../tools/export_grounding_onnx.py" \
            --checkpoint "$SAM3_CKPT" --output_dir "$onnx_gr_dir" || \
            echo "[WARN] grounding export failed, will use existing ONNX"
    fi

    # Grounding Encoder
    echo ""; echo "=== Grounding Encoder ==="
    ENC_ONNX="$onnx_gr_dir/sam3_grounding_encoder.onnx"
    [ -f "$ENC_ONNX" ] || ENC_ONNX="$onnx_dir/sam3_grounding_encoder.onnx"
    if [ -f "$ENC_ONNX" ]; then
        # onnxsim: fold Shape/Range ops so tpu-mlir shape-infer succeeds.
        python3 -c "
import onnx, onnxsim
m = onnx.load('$ENC_ONNX')
ms, ok = onnxsim.simplify(m)
assert ok, 'encoder onnxsim failed'
onnx.save(ms, 'sam3_grounding_encoder${suffix}_sim.onnx')
print(f'  enc onnxsim: {len(ms.graph.node)} nodes')
"
        ENC_SIM="sam3_grounding_encoder${suffix}_sim.onnx"
        python3 -c "
import numpy as np
np.random.seed(42)
G=$GRID
np.savez('enc_in${suffix}.npz',
  src=np.random.randn(1,256,G,G).astype(np.float32)*0.5,
  src_pos=np.random.randn(1,256,G,G).astype(np.float32)*0.5,
  prompt=np.random.randn(1,32,256).astype(np.float32)*0.5,
  prompt_mask=np.zeros((1,32),dtype=bool))
"
        gen_mlir "sam3_grounding_encoder${suffix}" "$ENC_SIM" \
            "[[$batch_size,256,$GRID,$GRID],[$batch_size,256,$GRID,$GRID],[1,32,256],[1,32]]" \
            "sam3_grounding_encoder${suffix}.mlir" "enc_in${suffix}.npz"
        gen_bmodel "sam3_grounding_encoder${suffix}.mlir" "sam3_grounding_encoder_${mode}_${batch_size}b.bmodel"
        mv "sam3_grounding_encoder_${mode}_${batch_size}b.bmodel" "$outdir/grounding/"
    else
        echo "[SKIP] Grounding encoder ONNX not found"
    fi

    # Grounding Decoder
    # NOTE: after the _get_coords patch + onnxsim, spatial_shapes and
    # level_start_index are folded out as constants → 6 inputs:
    # memory, memory_pos, memory_mask, valid_ratios, prompt, prompt_mask.
    echo ""; echo "=== Grounding Decoder ==="
    DEC_ONNX="$onnx_gr_dir/sam3_grounding_decoder.onnx"
    [ -f "$DEC_ONNX" ] || DEC_ONNX="$onnx_dir/sam3_grounding_decoder.onnx"
    if [ -f "$DEC_ONNX" ]; then
        TOKENS=$(( GRID * GRID ))
        python3 -c "
import onnx, onnxsim
m = onnx.load('$DEC_ONNX')
ms, ok = onnxsim.simplify(m)
assert ok, 'decoder onnxsim failed'
onnx.save(ms, 'sam3_grounding_decoder${suffix}_sim.onnx')
print(f'  dec onnxsim: {len(ms.graph.node)} nodes')
"
        DEC_SIM="sam3_grounding_decoder${suffix}_sim.onnx"
        python3 -c "
import numpy as np
np.random.seed(42)
N=$TOKENS
np.savez('dec_in${suffix}.npz',
  memory=np.random.randn(N,1,256).astype(np.float32)*0.5,
  memory_pos=np.random.randn(N,1,256).astype(np.float32)*0.5,
  memory_mask=np.zeros((N,1),dtype=bool),
  valid_ratios=np.ones((1,1,2),dtype=np.float32),
  prompt=np.random.randn(1,32,256).astype(np.float32)*0.5,
  prompt_mask=np.zeros((1,32),dtype=bool))
"
        gen_mlir "sam3_grounding_decoder${suffix}" "$DEC_SIM" \
            "[[$TOKENS,1,256],[$TOKENS,1,256],[$TOKENS,1],[1,1,2],[1,32,256],[1,32]]" \
            "sam3_grounding_decoder${suffix}.mlir" "dec_in${suffix}.npz"
        gen_bmodel "sam3_grounding_decoder${suffix}.mlir" "sam3_grounding_decoder_${mode}_${batch_size}b.bmodel"
        mv "sam3_grounding_decoder_${mode}_${batch_size}b.bmodel" "$outdir/grounding/"
    else
        echo "[SKIP] Grounding decoder ONNX not found"
    fi

    # Text Encoder
    echo ""; echo "=== Text Encoder ==="
    TEXT_ONNX_STATIC="../models/onnx/sam3_text_encoder_static.onnx"
    if [ -f "$TEXT_ONNX_STATIC" ]; then
        python3 -c "
import numpy as np
tokens = np.random.randint(100, 5000, (1, 32), dtype=np.int64)
np.savez('${cali_dir}/text_tokens.npz', data=tokens)
print(f'  tokens shape={tokens.shape}, range=[{tokens.min()}, {tokens.max()}]')
"
        python3 -c "
import onnx, onnxsim
m = onnx.load('$TEXT_ONNX_STATIC')
ms, check = onnxsim.simplify(m, overwrite_input_shapes={'token_ids': [1, 32]})
assert check, 'onnxsim failed'
onnx.save(ms, 'sam3_text_encoder_simp.onnx')
print(f'  Simplified: {len(ms.graph.node)} nodes (was {len(m.graph.node)})')
"
        gen_mlir "sam3_text_encoder${suffix}" "sam3_text_encoder_simp.onnx" \
            "[[1,32]]" "sam3_text_encoder${suffix}.mlir" "$cali_dir/text_tokens.npz" "int64"
        gen_bmodel "sam3_text_encoder${suffix}.mlir" "sam3_text_encoder_${mode}_${batch_size}b.bmodel"
        mv "sam3_text_encoder_${mode}_${batch_size}b.bmodel" "$outdir/grounding/"
    else
        echo "[WARN] Text encoder ONNX not found at $TEXT_ONNX_STATIC, skipping"
    fi
fi

# 清理中间产物
rm -f *.npz *.mlir *.cali_table sam3_text_encoder_simp.onnx sam3_grounding_*_sim.onnx sam3_neck*_fixed.onnx

popd

echo ""
echo "=== Compilation Complete ==="
echo "Output models:"
find "$outdir" -name "*.bmodel" -type f -exec ls -lh {} \; 2>/dev/null || echo "  No bmodels generated"

echo ""
echo "Validate with:"
echo "  bmrt_test --bmodel $outdir/vit/sam3_vit_part0_${mode}_${batch_size}b.bmodel --dev_id 0"
