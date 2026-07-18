#!/usr/bin/env python3
"""Export FunASR-Nano-2512 to ONNX (v2 — Qwen-compatible block format).

Block inputs:
  - block_i:     input_states(1,SEQ,HIDDEN), position_ids(1,SEQ), attention_mask(1,1,SEQ,SEQ)
  - block_cache_i: input_states(1,1,HIDDEN), position_ids(1,1), attention_mask(1,1,1,SEQ+1), past_k, past_v
cos/sin baked as register_buffer, position_ids kept via identity gate.
"""
import os, sys, argparse, struct
import torch, torch.nn as nn
import numpy as np
from tqdm import tqdm
torch.set_grad_enabled(False)

p = argparse.ArgumentParser()
p.add_argument('-s','--seq_length',type=int,default=512)
p.add_argument('--device',type=str,default='cpu')
args = p.parse_args()

folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'onnx')
os.makedirs(folder, exist_ok=True)
device = torch.device(args.device); dtype = torch.float32; SEQ = args.seq_length

# ============================================================
# Load
# ============================================================
print("Loading FunASR Nano...")
from funasr import AutoModel
model = AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512', trust_remote_code=True, device='cpu', disable_update=True)
m = model.model; m.eval(); m.llm.float()
for p in m.parameters(): p.requires_grad = False

cfg = m.llm.config
NL, H, KV, HD = cfg.num_hidden_layers, cfg.hidden_size, cfg.num_key_value_heads, m.llm.model.layers[0].self_attn.head_dim
print(f"layers={NL}, hidden={H}, kv_heads={KV}, head_dim={HD}, vocab={cfg.vocab_size}")

from transformers.models.qwen3.modeling_qwen3 import (Qwen3RotaryEmbedding, apply_rotary_pos_emb, eager_attention_forward)
rotary = Qwen3RotaryEmbedding(config=cfg)
pid_all = torch.arange(SEQ).unsqueeze(0)
cos_all, sin_all = [x.float() for x in rotary(x=torch.zeros(1,SEQ,H), position_ids=pid_all)]

# ============================================================
# Shared attention wrapper
# ============================================================
class AttnKV(nn.Module):
    def __init__(self, src):
        super().__init__()
        self.q_proj=src.q_proj; self.k_proj=src.k_proj; self.v_proj=src.v_proj
        self.q_norm=src.q_norm; self.k_norm=src.k_norm; self.o_proj=src.o_proj
        self.head_dim=src.head_dim; self.scaling=src.scaling
        self.sliding_window=src.sliding_window
        self.num_key_value_groups=src.num_key_value_groups
    def forward(self, hs, cos, sin, am):
        s=hs.shape[:-1]; sh=(*s,-1,self.head_dim)
        q=self.q_norm(self.q_proj(hs).view(sh)).transpose(1,2)
        k=self.k_norm(self.k_proj(hs).view(sh)).transpose(1,2)
        v=self.v_proj(hs).view(sh).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        o,_=eager_attention_forward(self,q,k,v,am,dropout=0.0,scaling=self.scaling,sliding_window=self.sliding_window)
        return self.o_proj(o.reshape(*s,-1).contiguous()), k.transpose(1,2), v.transpose(1,2)

class BlockFirst(nn.Module):
    def __init__(self, layer):
        super().__init__(); self.attn=AttnKV(layer.self_attn); self.mlp=layer.mlp
        self.ln1=layer.input_layernorm; self.ln2=layer.post_attention_layernorm
        self.register_buffer('cos_all', cos_all); self.register_buffer('sin_all', sin_all)
    def forward(self, hs, pid, am):
        L=pid.shape[1]
        cos=self.cos_all[:,:L,:]*1.0+0.0*pid[:,:,None].float().sum(-1,keepdim=True)
        sin=self.sin_all[:,:L,:]*1.0+0.0*pid[:,:,None].float().sum(-1,keepdim=True)
        r=hs; h=self.ln1(hs); ao,k,v=self.attn(h,cos,sin,am); h=r+ao; r=h
        h=self.ln2(h); h=self.mlp(h); return (r+h).float(),k.float(),v.float()

# ============================================================
# 1. Encoder
# ============================================================
print("[1/4] SANM Encoder...")
ds=torch.randn(1,100,560,dtype=dtype); dsl=torch.tensor([100],dtype=torch.int32)
torch.onnx.export(m.audio_encoder,(ds,dsl),os.path.join(folder,'sanm_encoder.onnx'),verbose=False,
    input_names=['speech','speech_lengths'],output_names=['encoder_out','encoder_out_lens'],
    dynamic_axes={'speech':{0:'batch',1:'audio_frames'},'speech_lengths':{0:'batch'},
                  'encoder_out':{0:'batch',1:'enc_frames'},'encoder_out_lens':{0:'batch'}},
    do_constant_folding=True,opset_version=14)
print(f"  ✅ {os.path.getsize(os.path.join(folder,'sanm_encoder.onnx'))//1024//1024} MB")

# ============================================================
# 2. Adapter
# ============================================================
print("[2/4] Audio Adapter...")
de=torch.randn(1,512,512,dtype=dtype); dl=torch.tensor([512],dtype=torch.int32)
torch.onnx.export(m.audio_adaptor,(de,dl),os.path.join(folder,'audio_adapter.onnx'),verbose=False,
    input_names=['encoder_out','encoder_out_lens'],output_names=['adaptor_out','adaptor_out_lens'],
    do_constant_folding=True,opset_version=14)
print(f"  ✅ {os.path.getsize(os.path.join(folder,'audio_adapter.onnx'))//1024//1024} MB")

# ============================================================
# 3. Blocks + Cache Blocks
# ============================================================
print(f"[3/4] {NL} blocks + {NL} cache blocks...")
hs1=torch.randn(1,SEQ,H,dtype=dtype); pid1=torch.arange(SEQ,dtype=torch.int32).unsqueeze(0)
am1=torch.full((1,1,SEQ,SEQ),float('-inf'),dtype=dtype)
for j in range(SEQ): am1[0,0,j,:j+1]=0.0

for i in tqdm(range(NL)):
    layer=m.llm.model.layers[i].float()
    # prefill
    bf=BlockFirst(layer).float().eval()
    torch.onnx.export(bf, (hs1,pid1,am1), os.path.join(folder,f'block_{i}.onnx'), verbose=False,
        input_names=['input_states','position_ids','attention_mask'],
        output_names=['hidden_states','past_k','past_v'],
        do_constant_folding=True, opset_version=15)
    # cache — no cos/sin baking needed, compute inline
    # For cache blocks, we export same structure with 5 inputs
    # (reuse BlockFirst with extended attention_mask)
    # Actually we need a separate cache block. Let me keep the v1 approach but fix inputs.
    # For now, cache blocks reuse the same BlockFirst ONNX — the inference code
    # handles KV concatenation externally (like Qwen sample does).

# Cache blocks: same architecture, just different input shapes
class BlockCache2(nn.Module):
    def __init__(self, layer):
        super().__init__(); self.attn=AttnKV(layer.self_attn); self.mlp=layer.mlp
        self.ln1=layer.input_layernorm; self.ln2=layer.post_attention_layernorm
        self.register_buffer('cos_all', cos_all); self.register_buffer('sin_all', sin_all)

    def forward(self, hs, pid, am, pk, pv):
        """pk,pv: (B, T_total, kv_heads, head_dim) — full history.
           Returns new hs, and concatenated k,v with new token appended."""
        L = pid.shape[1]  # =1 for decode
        T_total = pk.shape[1]  # current history length
        # Build full cos/sin for all positions
        cos_f = self.cos_all[:, :T_total+L, :] * 1.0 + 0.0 * pid.float().sum()
        sin_f = self.sin_all[:, :T_total+L, :] * 1.0 + 0.0 * pid.float().sum()
        # Run QKV
        s = hs.shape[:-1]; sh = (*s, -1, HD)
        q = self.attn.q_norm(self.attn.q_proj(hs).view(sh)).transpose(1,2)
        k_new = self.attn.k_norm(self.attn.k_proj(hs).view(sh)).transpose(1,2)
        v_new = self.attn.v_proj(hs).view(sh).transpose(1,2)
        q, k_new_r = apply_rotary_pos_emb(q, k_new, cos_f[:, T_total:, :], sin_f[:, T_total:, :])
        # Apply RoPE to past keys too (they already have RoPE, but we re-apply to the full sequence for ONNX compatibility)
        # Actually ONNX can't easily handle dynamic concat of past+present KV within a single ONNX graph.
        # The Qwen approach: let the HOST handle KV concat, ONNX block just processes new token.
        # So block_cache ONNX: 5 inputs (hs, pid, am, SPLIT past KV), outputs new hidden + new k_new, v_new
        q_final = q
        k_full = torch.cat([pk.transpose(1,2), k_new_r], dim=2)
        v_full = torch.cat([pv.transpose(1,2), v_new], dim=2)
        ao, _ = eager_attention_forward(self.attn, q_final, k_full, v_full, am,
            dropout=0.0, scaling=self.attn.scaling, sliding_window=self.attn.sliding_window)
        ao = ao.reshape(*s, -1).contiguous()
        ao = self.attn.o_proj(ao)
        r = hs; h = self.ln1(hs); h = r + ao; r = h
        h = self.ln2(h); h = self.mlp(h)
        return (r+h).float(), k_new_r.transpose(1,2).float(), v_new.transpose(1,2).float()

hs2 = torch.randn(1, 1, H, dtype=dtype)
pid2 = torch.tensor([[SEQ]], dtype=torch.int32)
am2 = torch.zeros(1, 1, 1, SEQ+1, dtype=dtype)
pk0 = torch.randn(1, SEQ, KV, HD, dtype=dtype)
pv0 = torch.randn(1, SEQ, KV, HD, dtype=dtype)

for i in tqdm(range(NL)):
    layer = m.llm.model.layers[i].float()
    bc = BlockCache2(layer).float().eval()
    torch.onnx.export(bc, (hs2, pid2, am2, pk0, pv0),
        os.path.join(folder, f'block_cache_{i}.onnx'), verbose=False,
        input_names=['input_states','position_ids','attention_mask','past_k','past_v'],
        output_names=['hidden_states','past_k','past_v'],
        do_constant_folding=True, opset_version=15)

# ============================================================
# 4. Embedding + LM Head + Greedy
# ============================================================
print("[4/4] Embedding / LM Head / Greedy...")
ew = m.llm.model.embed_tokens.weight.data.float().numpy().astype(np.float16)
with open(os.path.join(folder,'embedding.bin'),'wb') as f: f.write(ew.tobytes())
print(f"  ✅ embedding.bin ({os.path.getsize(os.path.join(folder,'embedding.bin'))//1024//1024} MB)")

lw = m.llm.lm_head.weight.data.float().numpy().astype(np.float16)
with open(os.path.join(folder,'lm_head.bin'),'wb') as f: f.write(lw.tobytes())
print(f"  ✅ lm_head.bin")

nw = m.llm.model.norm.weight.data.float().numpy().astype(np.float16)
with open(os.path.join(folder,'norm.bin'),'wb') as f: f.write(nw.tobytes())
print(f"  ✅ norm.bin")

class GreedyHead(nn.Module):
    def forward(self, x): _, t = torch.topk(x.float(), 1); return t
torch.onnx.export(GreedyHead().eval(), (torch.randn(1,cfg.vocab_size,dtype=dtype),),
    os.path.join(folder,'greedy_head.onnx'), verbose=False,
    input_names=['m_logits'],output_names=['token'],
    do_constant_folding=True,opset_version=15)
print(f"  ✅ greedy_head.onnx")

# ============================================================
# Verify
# ============================================================
import onnx, onnxruntime as ort
for name in ['sanm_encoder','audio_adapter','block_0','block_cache_0','greedy_head']:
    m2 = onnx.load(os.path.join(folder,f'{name}.onnx'))
    onnx.checker.check_model(m2)
    print(f"  ✅ {name}.onnx valid")

sf = sorted(os.listdir(folder))
total = sum(os.path.getsize(os.path.join(folder,f))/1024/1024 for f in sf)
print(f"\n{'='*60}\nExported {len(sf)} files ({total:.0f} MB) to {folder}/\n{'='*60}")
