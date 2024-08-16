import os
import time
import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function
from .untool import Tool, make_np2c, data_type, data_type_map

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        _, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[ 1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-10000).triu_(1).flip(1)

        self.mask = mask
    
    def embedding(self, x: Tensor):
        return self.token_embedding(x)
    
    def attention(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        x_embedding = self.token_embedding(x)
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            x_embedding
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        # TODO: replace self.positon_embedding with a npz file
        x = x.to(xa.dtype)

        return self.attention(x, xa, kv_cache=kv_cache)

class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, args):
        super().__init__()
        self.dims = dims
        self.model_name = args["model_name"]
        self.encoder = None
        self.decoder = None
        self.encoder_infer = None
        self.logits_decoder_infer = None
        self.decoder_main_infer = None
        self.decoder_post_infer = None
        self.decoder_loop_infer = None
        self.bmodel_dir = args["bmodel_dir"]
        self.beam_size = args["beam_size"]
        self.padding_size = args["padding_size"]
        self.chip_mode = args["chip_mode"]
        self.chip = args["chip"]

        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = all_heads.to_sparse()

        # get positional embedding from npz file
        positional_embedding_path = os.path.join(os.path.dirname(__file__), "assets", f"positional_embedding_{self.model_name}.npz")
        assert os.path.exists(positional_embedding_path), f"{positional_embedding_path} not found"
        self.positional_embedding = torch.tensor(np.load(positional_embedding_path)["positional_embedding"])

        ########################################
        ## Using untool to load BModel
        ########################################
        start_time = time.time()
        quant_str = "all_quant"
        encoder_bmodel_path           = f"{quant_str}_encoder_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_{self.chip}_f16.bmodel"
        logits_decoder_bmodel_path    = f"{quant_str}_logits_decoder_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_{self.chip}_f16.bmodel"
        decoder_main_bmodel_path      = f"{quant_str}_decoder_main_with_kvcache_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_{self.chip}_f16.bmodel"
        decoder_post_bmodel_path      = f"{quant_str}_decoder_post_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_{self.chip}_f16.bmodel"
        decoder_loop_bmodel_path      = f"{quant_str}_decoder_loop_with_kvcache_and_rearrange_{self.model_name}_{self.beam_size}beam_{self.padding_size}pad_{self.chip}_f16.bmodel"
        encoder_bmodel_path           = os.path.join(self.bmodel_dir, encoder_bmodel_path)
        logits_decoder_bmodel_path    = os.path.join(self.bmodel_dir, logits_decoder_bmodel_path)
        decoder_main_bmodel_path      = os.path.join(self.bmodel_dir, decoder_main_bmodel_path)
        decoder_post_bmodel_path      = os.path.join(self.bmodel_dir, decoder_post_bmodel_path)
        decoder_loop_bmodel_path      = os.path.join(self.bmodel_dir, decoder_loop_bmodel_path)
        assert os.path.exists(encoder_bmodel_path), f"{encoder_bmodel_path} not found"
        assert os.path.exists(logits_decoder_bmodel_path), f"{logits_decoder_bmodel_path} not found"
        assert os.path.exists(decoder_main_bmodel_path), f"{decoder_main_bmodel_path} not found"
        assert os.path.exists(decoder_post_bmodel_path), f"{decoder_post_bmodel_path} not found"
        assert os.path.exists(decoder_loop_bmodel_path), f"{decoder_loop_bmodel_path} not found"

        self.tool                  = Tool(self.chip_mode)
        self.handle                = self.tool.bmhandle(args["devid"])
        self.bmrt1                 = self.tool.bmrt(self.handle)
        self.bmrt2                 = self.tool.bmrt(self.handle)
        self.bmrt3                 = self.tool.bmrt(self.handle)
        self.bmrt4                 = self.tool.bmrt(self.handle)
        self.bmrt5                 = self.tool.bmrt(self.handle)
        self.encoder_handle        = self.tool.create_model(encoder_bmodel_path.encode("utf-8"), self.bmrt1)
        self.logits_decoder_handle = self.tool.create_model(logits_decoder_bmodel_path.encode("utf-8"), self.bmrt2)
        self.decoder_main_handle   = self.tool.create_model(decoder_main_bmodel_path.encode("utf-8"), self.bmrt3)
        self.decoder_post_handle   = self.tool.create_model(decoder_post_bmodel_path.encode("utf-8"), self.bmrt4)
        self.decoder_loop_handle   = self.tool.create_model(decoder_loop_bmodel_path.encode("utf-8"), self.bmrt5)
        self.runtime1              = self.tool.create_un_runtime(self.handle)
        self.runtime2              = self.tool.create_un_runtime(self.handle)
        self.runtime3              = self.tool.create_un_runtime(self.handle)
        self.runtime4              = self.tool.create_un_runtime(self.handle)
        self.runtime5              = self.tool.create_un_runtime(self.handle)

        self.tool.set_bmodel_info(self.runtime1, self.encoder_handle)
        self.tool.set_bmodel_info(self.runtime2, self.logits_decoder_handle)
        self.tool.set_bmodel_info(self.runtime3, self.decoder_main_handle)
        self.tool.set_bmodel_info(self.runtime4, self.decoder_post_handle)
        self.tool.set_bmodel_info(self.runtime5, self.decoder_loop_handle)
        self.tool.set_stage(self.runtime1, 0)
        self.tool.set_stage(self.runtime2, 0)
        self.tool.set_stage(self.runtime3, 0)
        self.tool.set_stage(self.runtime4, 0)
        self.tool.set_stage(self.runtime5, 0)
        self.tool.init_all_tensors(self.runtime1)
        self.tool.init_all_tensors(self.runtime2)
        self.tool.init_all_tensors(self.runtime3)
        self.tool.init_all_tensors(self.runtime4)
        self.tool.init_all_tensors(self.runtime5)
        self.tool.malloc_device_address(self.runtime1)
        self.tool.malloc_device_address(self.runtime2)
        self.tool.malloc_device_address(self.runtime3)
        self.tool.malloc_device_address(self.runtime4)
        self.tool.malloc_device_address(self.runtime5)

        for i in range(self.dims.n_text_layer * 4):
            # self.tool.set_input_tensor(self.runtime5, i + 3, self.tool.get_output_tensor(self.runtime3, i + 1))
            self.tool.set_input_tensor(self.runtime5, i + 4, self.tool.get_output_tensor(self.runtime3, i + 1))
        for i in range(self.dims.n_text_layer * 2):
            self.tool.set_output_tensor(self.runtime5, i + 1, self.tool.get_input_tensor(self.runtime5, i + 4))

        self.init_time = time.time() - start_time
        print(f"\nTPU bmodel init time: {self.init_time}s")

        self.time = 0
        self.main_loop_cnt = 0
        self.call_encoder = 0
        self.call_logits_decoder= 0
        self.call_decoder_loop= 0
        self.call_decoder_firstly= 0
        self.call_decoder_post = 0
        self.call_decoder_with_kvcache = 0
        self.max_ctx = 0
        self.call_encoder_time = 0
        self.call_logits_decoder_time = 0
        self.call_decoder_loop_time = 0
        self.call_decoder_firstly_time = 0
        self.call_decoder_post_time = 0
        self.call_decoder_with_kvcache_time = 0
    
    def init_cnt(self):
        self.main_loop_cnt = 0
        self.call_encoder = 0
        self.call_logits_decoder= 0
        self.call_decoder_loop= 0
        self.call_decoder_firstly= 0
        
    
    def print_cnt(self):
        def print_cnt(text, cnt, n):
            print(f"{text:<{n}} {cnt}")
        print()
        print_cnt("Call encoder times:", self.call_encoder, 50)
        print_cnt("", f"{self.call_encoder_time}s", 50)
        print_cnt("Call logits decoder times:", self.call_logits_decoder, 50)
        print_cnt("", f"{self.call_logits_decoder_time}s", 50)
        print_cnt("Call decoder firstly times:", self.call_decoder_firstly, 50)
        print_cnt("", f"{self.call_decoder_firstly_time}s", 50)
        print_cnt("Call decoder loop:", self.call_decoder_loop, 50)
        print_cnt("", f"{self.call_decoder_loop_time}s", 50)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.alignment_heads = mask.to_sparse()

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        # print("{:=^100}".format(" model.logits "))
        # TODO: condition of multi-channel audio
        start_time = time.time()

        logits_decoder_info = self.tool.model_info(self.logits_decoder_handle)
        tokens_input_dtype = data_type_map[logits_decoder_info['input_dtypes'][0]]
        audio_features_input_dtype = data_type_map[logits_decoder_info['input_dtypes'][1]]
        tokens = tokens.numpy().astype(tokens_input_dtype)
        audio_features = audio_features.numpy().astype(audio_features_input_dtype)
        tokens = tokens if tokens.flags.c_contiguous else np.ascontiguousarray(tokens)
        audio_features = audio_features if audio_features.flags.c_contiguous else np.ascontiguousarray(audio_features)

        self.tool.copy_data_from_numpy(self.tool.get_input_tensor(self.runtime2, 0), make_np2c(tokens), data_type[tokens_input_dtype])
        self.tool.copy_data_from_numpy(self.tool.get_input_tensor(self.runtime2, 1), make_np2c(audio_features), data_type[audio_features_input_dtype])
        self.tool.force_host_to_device(self.tool.get_input_tensor(self.runtime2, 0), self.handle)
        self.tool.force_host_to_device(self.tool.get_input_tensor(self.runtime2, 1), self.handle)

        logits = np.empty(logits_decoder_info[0]['output_shapes'][0], dtype=data_type_map[logits_decoder_info['output_dtypes'][0]])
        self.tool.copy_data_from_numpy(self.tool.get_output_tensor(self.runtime2, 0), make_np2c(logits), logits_decoder_info['output_dtypes'][0])
        self.tool.inference(self.runtime2)
        self.tool.copy_output_data_to_host(self.runtime2)

        logits = torch.from_numpy(logits)

        t = time.time() - start_time
        self.time += t
        # print(f"logits inference time: {time.time() - start_time} seconds")
        self.call_logits_decoder_time += t
        self.call_logits_decoder += 1
        return logits

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        print("{:=^100}".format(" model.forward "))
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865
    
    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
