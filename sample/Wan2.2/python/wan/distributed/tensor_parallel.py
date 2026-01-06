import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from ..modules.model import WanModel, WanAttentionBlock, WanLayerNorm, WanRMSNorm, WanSelfAttention, padding_kv


class TPWanRMSNorm(WanRMSNorm):

    def __init__(self, dim, eps=1e-5):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        super().__init__(dim, eps)

    def forward(self, x):
        denom = x.pow(2).to(torch.float32).mean(dim=-1, keepdim=True)
        dist.all_reduce(denom, op=dist.ReduceOp.SUM)
        denom = (denom / self.world_size).to(torch.bfloat16)
        y = x * torch.rsqrt(denom + self.eps) * self.weight
        return y


class TPLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        super().__init__(in_features, out_features, bias, device, dtype)

    # def forward(self, input):
    #     if self.weight.data.dtype != torch.float32:
    #         self.weight.data = self.weight.data.to(dtype=torch.float32)
    #         self.bias.data = self.bias.data.to(dtype=torch.float32)
    #     y = F.linear(input.to(torch.float32), self.weight, None)
    #     if self.rank == 0:
    #         y += self.bias
    #     return y

    def forward(self, input):
        y = F.linear(input, self.weight, None)
        if self.rank == 0:
            y += self.bias
        return y


class TPWanSelfAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim // self.world_size)
        self.k = nn.Linear(dim, dim // self.world_size)
        self.v = nn.Linear(dim, dim // self.world_size)
        self.o = TPLinear(dim // self.world_size, dim)
        self.norm_q = TPWanRMSNorm(dim // self.world_size, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = TPWanRMSNorm(dim // self.world_size, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads // self.world_size, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        cos, sin = freqs
        scale = q.shape[-1]**-0.5
        x = torch.zeros_like(q)
        torch.ops.my_ops.llava_attention(x, q, k, v, cos, sin, None, scale)

        # output
        x = x.flatten(2)
        x = self.o(x)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x.to(torch.bfloat16)
        return x


class TPWanCrossAttention(TPWanSelfAttention):

    def forward(self, x, context, context_lens, cache=None):
        b, n, d = x.size(0), self.num_heads // self.world_size, self.head_dim

        # compute query, key, value, mask
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        scale = q.shape[-1]**-0.5
        if True:
            if cache is not None:
                if self.iter < 2:
                    k = self.norm_k(self.k(context)).view(b, -1, n, d)
                    v = self.v(context).view(b, -1, n, d)
                    k, v, mask = padding_kv(q, k, v)
                    cache['cross_attn_k'].setdefault(self.iter%2, {})[self.block_id] = k
                    cache['cross_attn_v'].setdefault(self.iter%2, {})[self.block_id] = v
                    cache['cross_attn_mask'][self.iter%2] = mask
                else:
                    k = cache['cross_attn_k'][self.iter%2][self.block_id]
                    v = cache['cross_attn_v'][self.iter%2][self.block_id]
                    mask = cache['cross_attn_mask'][self.iter%2]
            else:
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                k, v, mask = padding_kv(q, k, v)
            x = torch.empty_like(q)
            torch.ops.my_ops.llava_attention(x, q, k, v, None, None, mask, scale)
        else:
            if cache is not None:
                if self.iter < 2:
                    k = self.norm_k(self.k(context)).view(b, -1, n, d)
                    v = self.v(context).view(b, -1, n, d)
                    cache['cross_attn_k'].setdefault(self.iter%2, {})[self.block_id] = k
                    cache['cross_attn_v'].setdefault(self.iter%2, {})[self.block_id] = v
                else:
                    k = cache['cross_attn_k'][self.iter%2][self.block_id]
                    v = cache['cross_attn_v'][self.iter%2][self.block_id]
            else:
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
            x = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), attn_mask=None).transpose(1,2)

        # output
        x = x.flatten(2)
        x = self.o(x)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x.to(torch.bfloat16)
        return x


class TPWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        assert dim % num_heads == 0 and num_heads % self.world_size == 0 and ffn_dim % self.world_size == 0

        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = TPWanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = TPWanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim // self.world_size), nn.GELU(approximate='tanh'),
            TPLinear(ffn_dim // self.world_size, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        tp_params = [
            ('self_attn.norm_q.weight', 0),
            ('self_attn.norm_k.weight', 0),
            ('self_attn.q.weight', 0),
            ('self_attn.q.bias', 0),
            ('self_attn.k.weight', 0),
            ('self_attn.k.bias', 0),
            ('self_attn.v.weight', 0),
            ('self_attn.v.bias', 0),
            ('self_attn.o.weight', 1),
            ('cross_attn.norm_q.weight', 0),
            ('cross_attn.norm_k.weight', 0),
            ('cross_attn.q.weight', 0),
            ('cross_attn.q.bias', 0),
            ('cross_attn.k.weight', 0),
            ('cross_attn.k.bias', 0),
            ('cross_attn.v.weight', 0),
            ('cross_attn.v.bias', 0),
            ('cross_attn.o.weight', 1),
            ('ffn.0.weight', 0),
            ('ffn.0.bias', 0),
            ('ffn.2.weight', 1),
        ]

        def _shard(x, dim, world_size, rank):
            assert x.shape[dim] % world_size == 0
            return x.chunk(world_size, dim=dim)[rank]

        for name, dim in tp_params:
            k = prefix + name
            if k in state_dict:
                state_dict[k] = _shard(state_dict[k], dim, self.world_size, self.rank)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        cache=None,
    ):
        assert e.dtype == torch.bfloat16
        with torch.amp.autocast('tpu', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.bfloat16

        # self-attention
        y = self.self_attn(
            self.norm1(x).to(torch.bfloat16) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        with torch.amp.autocast('tpu', dtype=torch.bfloat16):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, cache)
            y = self.ffn(
                self.norm2(x).to(torch.bfloat16) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
            y = y.to(torch.bfloat16)
            with torch.amp.autocast('tpu', dtype=torch.bfloat16):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        torch.tpu.synchronize()
        return x


class TPWanModel(WanModel):

    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        assert dist.is_initialized(), "Distributed process group is not initialized."
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim, num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)
        self.blocks = nn.ModuleList([
            TPWanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])
