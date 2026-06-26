#===----------------------------------------------------------------------===#
# CLIP ViT-B/32 文本编码器导出 — torch 1.13 版 (真实 nn.MultiheadAttention)
# torch 1.13 的 MHA 不走 SDPA, 可直接导出标准算子, TPU-MLIR 能正确编译。
# 在 /opt/clip_export_venv (torch 1.13) 中运行。
# onnx: in tokens[1,77] int -> out text_features[1,77,512] (ln_final 后, proj 前)
# text_projection(512x512) 另存 npy; argmax+proj+归一化 由推理侧 clip.py 完成
#===----------------------------------------------------------------------===#
import os, shutil, argparse
import numpy as np
import torch
import torch.nn as nn
import clip


class ClipTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final

    def forward(self, text):
        x = self.token_embedding(text).type(self.positional_embedding.dtype) + self.positional_embedding
        x = x.permute(1, 0, 2)              # [seq, N, width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)              # [N, seq, width]
        x = self.ln_final(x).type(self.positional_embedding.dtype)
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_model", default="ViT-B/32")
    ap.add_argument("--outdir", default="../models")
    ap.add_argument("--onnx_dir", default="../models/onnx")
    ap.add_argument("--out_name", default="clip_text_vitb32")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.onnx_dir, exist_ok=True)

    print(f"[clip] loading {args.clip_model} (torch {torch.__version__})...")
    model, _ = clip.load(args.clip_model, device="cpu")
    model.eval()

    text_proj = model.text_projection.detach().cpu().numpy().astype(np.float32)
    print(f"[clip] text_projection shape={text_proj.shape}")

    encoder = ClipTextEncoder(model).eval()
    dummy = torch.randint(0, 49408, (1, 77), dtype=torch.int32)

    onnx_path = os.path.join(args.onnx_dir, f"{args.out_name}.onnx")
    with torch.no_grad():
        torch.onnx.export(encoder, (dummy,), onnx_path, do_constant_folding=True,
                          opset_version=13, input_names=["tokens"],
                          output_names=["text_features"])
    print(f"[clip] saved -> {onnx_path}")

    import onnx
    from onnxsim import simplify as onnx_simplify
    sim, check = onnx_simplify(onnx.load(onnx_path))
    assert check
    onnx.save(sim, onnx_path)
    onnx.checker.check_model(onnx_path)
    for i in sim.graph.input:
        shp = [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"  input  {i.name}: {shp}")
    for o in sim.graph.output:
        shp = [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"  output {o.name}: {shp}")

    np.save(os.path.join(args.outdir, "text_projection_512_512.npy"), text_proj)
    bpe_src = os.path.join(os.path.dirname(clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
    bpe_dst = os.path.join(args.outdir, "bpe_simple_vocab_16e6.txt.gz")
    if os.path.exists(bpe_src):
        shutil.copy(bpe_src, bpe_dst)
        print(f"[clip] copied bpe vocab -> {bpe_dst}")

    # 验证 vs 真实 OpenAI encode_text (归一化)
    import onnxruntime as ort
    with torch.no_grad():
        ref = model.encode_text(dummy)
        ref_n = ref / ref.norm(dim=-1, keepdim=True)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"tokens": dummy.numpy()})[0]
    idx = dummy.argmax(dim=-1).numpy()
    emb = o[np.arange(o.shape[0]), idx] @ text_proj
    emb_n = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    cos = float(np.dot(ref_n.numpy().reshape(-1), emb_n.reshape(-1)) /
                (np.linalg.norm(ref_n) * np.linalg.norm(emb_n) + 1e-9))
    print(f"[clip] verify (onnx vs real OpenAI encode_text) cosine={cos:.6f}")
    assert cos > 0.9999
    print("[clip] done")


if __name__ == "__main__":
    main()
