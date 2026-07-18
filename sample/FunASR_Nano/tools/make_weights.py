import torch, numpy as np, os
from funasr import AutoModel
model = AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512', trust_remote_code=True, device='cpu', disable_update=True)
m = model.model.llm
D = 'models/BM1684X'
ew = m.model.embed_tokens.weight.data.float().numpy().astype('float16')
with open(f'{D}/embedding.bin', 'wb') as f: f.write(ew.tobytes())
lw = m.lm_head.weight.data.float().numpy().astype('float16')
with open(f'{D}/lm_head.bin', 'wb') as f: f.write(lw.tobytes())
nw = m.model.norm.weight.data.float().numpy().astype('float16')
with open(f'{D}/norm.bin', 'wb') as f: f.write(nw.tobytes())
print(f"embedding: {ew.shape} {os.path.getsize(f'{D}/embedding.bin')/1024/1024:.0f}MB")
print(f"lm_head:   {lw.shape} {os.path.getsize(f'{D}/lm_head.bin')/1024/1024:.0f}MB")
