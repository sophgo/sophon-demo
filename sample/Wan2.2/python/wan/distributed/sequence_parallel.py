# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

from ..modules.model import sinusoidal_embedding_1d, build_freqs_real, get_cos_sin
from .util import all_to_all, gather_forward, get_rank, get_world_size
from .ulysses import distributed_attention


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
    ])

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]

    if self.iter == 0:
        freqs = build_freqs_real(1024, self.dim // self.num_heads, dtype=torch.float32)
        cos_full, sin_full = [], []
        for (f, h, w) in grid_sizes.tolist():
            cos, sin = get_cos_sin(f, h, w, freqs)
            cos_full.append(cos)
            sin_full.append(sin)
        self.freqs = (torch.stack(cos_full).to(x), torch.stack(sin_full).to(x))

    # time embeddings
    if self.iter % 2 == 0:
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('tpu', dtype=torch.bfloat16):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).to(torch.bfloat16))
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16
            e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
            e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]
            self.cache['e'].clear()
            self.cache['e0'].clear()
            self.cache['e'][self.iter] = e
            self.cache['e0'][self.iter] = e0

    # context
    context_lens = None
    if self.iter < 2:
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        self.cache['context'][self.iter] = context

    # arguments
    kwargs = dict(
        e=self.cache['e0'][self.iter//2*2],
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=self.cache['context'][self.iter%2],
        context_lens=context_lens,
        cache=self.cache)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, self.cache['e'][self.iter//2*2])

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)

    self.iter += 1
    return [u.to(torch.bfloat16) for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    # gather q/k/v sequence
    q = all_to_all(q, scatter_dim=2, gather_dim=1)
    k = all_to_all(k, scatter_dim=2, gather_dim=1)
    v = all_to_all(v, scatter_dim=2, gather_dim=1)

    cos, sin = freqs
    scale = q.shape[-1]**-0.5
    x = torch.zeros_like(q)
    torch.ops.my_ops.llava_attention(x, q, k, v, cos, sin, None, scale)

    # scatter q/k/v sequence
    x = all_to_all(x, scatter_dim=1, gather_dim=2)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
