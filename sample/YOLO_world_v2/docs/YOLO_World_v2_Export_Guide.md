# YOLO_world_v2 模型导出

## 1. 主检测模型 (yoloworld_v2.onnx)
用 ultralytics 导出 yolov8s-worldv2，展开 `model.predict(x, txt_feats)` 使文本嵌入为显式第二输入。

```bash
# TPU-MLIR 容器内 (torch 2.0.1 + ultralytics 8.4.75)
pip install ultralytics  # 若未装
cd tools
python export_yoloworld_onnx.py --model_name yolov8s-worldv2 --img_size 640 --num_classes 80
```

产出 `models/onnx/yoloworld_v2.onnx`：in `images[1,3,640,640]`+`txt_feats[1,80,512]`, out `output[1,84,8400]`。

要点：
- `set_classes([""]*80)` + `model.fuse()` + `model.predict(x, txt_feats)`，opset 12 + onnxsim。
- ultralytics 8.4.75 会导出 6 个输出（Detect 头特征图），脚本已剪枝为单 `output`。
- 导出后用 onnxruntime 验证与 PyTorch 余弦相似度 > 0.9999。

## 2. 文本编码器 (clip_text_vitb32.onnx)
**必须用 torch 1.13 导出**（torch 2.0+ 的 `nn.MultiheadAttention` 走 SDPA，TPU-MLIR 会把 CustomMHA 替换版编译成退化图）。

```bash
# 一次性准备 torch 1.13 venv
python3 -m venv /opt/clip_export_venv
/opt/clip_export_venv/bin/pip install torch==1.13.1 onnx==1.14.1 onnxsim==0.4.17 \
    onnxruntime==1.16.3 ftfy regex "numpy<2" tqdm requests
/opt/clip_export_venv/bin/pip install --no-deps torchvision==0.14.1 Pillow
git clone --depth 1 https://github.com/openai/CLIP.git /opt/clip_src

# 导出
cd tools
PYTHONPATH=/opt/clip_src /opt/clip_export_venv/bin/python export_clip_text_onnx.py
```

产出：
- `models/onnx/clip_text_vitb32.onnx`：in `tokens[1,77]` int, out `text_features[1,77,512]`（ln_final 后，text_projection 前）。
- `models/text_projection_512_512.npy`：CLIP text_projection [512,512]，推理侧 numpy 点乘。
- `models/bpe_simple_vocab_16e6.txt.gz`：CLIP BPE 分词表。

推理侧 `python/clip/clip.py` 完成：argmax(EOT 取值) + dot(text_projection) + L2 归一化 → `txt_feats[1,80,512]`。

验证：onnx 输出与真实 OpenAI `encode_text`（归一化）余弦相似度 > 0.9999。

> CLIP 文本编码器是**因果掩码**（`build_attention_mask` 上三角 -inf）。自定义注意力实现必须应用 attn_mask，否则嵌入 cos≈0.30。
