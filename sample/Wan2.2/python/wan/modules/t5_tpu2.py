# Modified from transformers.models.t5.modeling_t5
# SPDX-License-Identifier: Apache-2.0
# Adapted from
# https://github.com/Wan-Video/Wan2.2
# Copyright 2025 The vLLM team.
# Copyright 2024-2025 The Alibaba Wan Team Authors.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import HuggingfaceTokenizer

__all__ = [
    'T5Encoder',
    'T5EncoderModel',
]


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        a=torch.pow(x.float(),2)
        b= a.mean(dim=-1, keepdim=True)
        c= b + self.eps
        d=torch.rsqrt(c)
        x = x.cpu() * d.cpu()
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        res = self.weight * x.to(a.device)
        return  res


class T5Attention(nn.Module):

    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1,
                             -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        q_permuted = q.permute(0, 2, 1, 3)
        k_permuted = k.permute(0, 2, 1, 3)
        v_permuted = v.permute(0, 2, 1, 3)

        attn2 = torch.matmul(q_permuted, k_permuted.transpose(-2, -1)) + attn_bias
        attn2 = F.softmax(attn2.float(), dim=-1).type_as(attn2)
        x2 = torch.matmul(attn2, v_permuted)
        x = x2.permute(0, 2, 1, 3)

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):

    def __init__(self, dim, dim_ffn, dropout=0.1):
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None): 
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=e)
        x = x + self.ffn(self.norm2(x))
        return x


class T5RelativeEmbedding(nn.Module):

    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - \
            torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos.to(torch.int32))
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(
            0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        num_buckets = self.num_buckets // 2
        rel_buckets = torch.where(rel_pos > 0, num_buckets, 0).float()
        rel_buckets = rel_buckets.to(rel_pos.device)
        rel_pos = torch.abs(rel_pos.float())

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.cpu().float() / max_exact) /
                                     math.log(self.max_dist / max_exact) *
                                     (num_buckets - max_exact)).long()
        rel_pos_large=rel_pos_large.to(torch.float32)
        rel_pos_large =rel_pos_large.to(rel_pos.device)
        rel_pos = rel_pos.to(rel_pos.device)
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))

        a=torch.where(rel_pos.to(torch.int32) < max_exact, rel_pos, rel_pos_large)
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return  rel_buckets


class T5Encoder(nn.Module):

    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5Encoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout) for _ in range(24)
        ])
        self.norm = T5LayerNorm(dim)

    def forward(self, ids, mask=None):
 
        x = self.token_embedding(ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask, pos_bias=None)
        x = self.norm(x)
        x = self.dropout(x)
        return x

def _t5(name,
        encoder_only=False,
        decoder_only=False,
        return_tokenizer=False,
        tokenizer_kwargs={},
        dtype=torch.float32,
        device='cpu',
        **kwargs):
    # sanity check

    assert  (encoder_only)
    model_cls = T5Encoder

    # init model
    with torch.device(device):
        model = model_cls(**kwargs)

    # set device
    model = model.to(dtype=dtype, device=device)

    # init tokenizer
    if return_tokenizer:
        from .tokenizers import HuggingfaceTokenizer
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    cfg = dict(
        vocab=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        num_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1)
    cfg.update(**kwargs)
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:

    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device= torch.device("cpu"),
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device).eval().requires_grad_(False)
        logging.info(f'loading {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts, device):
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)
        seq_lens = mask.gt(0).sum(dim=1).to(torch.int32)
        ids =ids.to(torch.int32)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = seq_lens.to(device)
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]
