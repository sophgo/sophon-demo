# ported from: https://github.com/neonbjb/tortoise-tts

import functools
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions

from .gpt_inference import GPT2InferenceModel
from .latent_encoder import ConditioningEncoder
from .perceiver_encoder import PerceiverResampler
# from tpu_perf.infer import SGInfer, nptype
import sophon.sail as sail


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02, relative=False):
        super().__init__()
        # nn.Embedding
        self.emb = torch.nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            return self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def build_hf_gpt_transformer(
    model_dim,
    max_mel_seq_len,
    max_text_seq_len,
):
    mel_pos_emb = (
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim)
        if max_mel_seq_len != -1
        else functools.partial(null_position_embeddings, dim=model_dim)
    )
    text_pos_emb = (
        LearnedPositionEmbeddings(max_text_seq_len, model_dim)
        if max_mel_seq_len != -1
        else functools.partial(null_position_embeddings, dim=model_dim)
    )
    return None, mel_pos_emb, text_pos_emb, None, None

class GPT(nn.Module):
    def __init__(
        self,
        start_text_token=261,
        stop_text_token=0,
        model_dim=512,
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_conditioning_inputs=1,
        code_stride_len=1024,
        number_text_tokens=256,
        num_audio_tokens=8194,
        start_audio_token=8192,
        stop_audio_token=8193,
        tpu_inference_config=None
    ):
        """
        Args:

        """
        super().__init__()

        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.num_audio_tokens = num_audio_tokens
        self.start_audio_token = start_audio_token
        self.stop_audio_token = stop_audio_token
        self.start_prompt_token = start_audio_token
        self.stop_prompt_token = stop_audio_token
        self.heads = heads
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.max_gen_mel_tokens = max_mel_tokens - self.max_conditioning_inputs - 2
        self.max_mel_tokens = -1 if max_mel_tokens == -1 else max_mel_tokens + 2 + self.max_conditioning_inputs
        self.max_text_tokens = -1 if max_text_tokens == -1 else max_text_tokens + 2
        self.code_stride_len = code_stride_len
        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=heads)
        self.conditioning_dropout = nn.Dropout1d(0.1)

        self.text_embedding = nn.Embedding(self.number_text_tokens, model_dim)
        self.mel_embedding = nn.Embedding(self.num_audio_tokens, model_dim)

        (
            _,
            self.mel_pos_embedding,
            self.text_pos_embedding,
            _,
            _,
        ) = build_hf_gpt_transformer(
            model_dim,
            self.max_mel_tokens,
            self.max_text_tokens,
        )

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens)
        self.mel_head = nn.Linear(model_dim, self.num_audio_tokens)

        # XTTS v2
        self.conditioning_perceiver = PerceiverResampler(
            dim=model_dim,
            depth=2,
            dim_context=model_dim,
            num_latents=32,
            dim_head=64,
            heads=8,
            ff_mult=4,
            use_flash_attn=False,
        )

    def init_gpt_for_inference(self, kv_cache=True, tpu_inference_config=None):
        self.gpt_inference = GPT2InferenceModel(
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
            tpu_inference_config=tpu_inference_config,
        )
        gpt_inner_path = tpu_inference_config["gpt_inner_path"]
        self.devid = tpu_inference_config["devid"]
        # self.gpt_inner_inference = SGInfer(gpt_inner_path, 1, [0])
        # load bmodel
        self.gpt_inner_inference = sail.Engine(gpt_inner_path, self.devid, sail.IOMode.SYSIO)
        self.handle = sail.Handle(0)
        self.graph_name = self.gpt_inner_inference.get_graph_names()[0]
        self.input_names = self.gpt_inner_inference.get_input_names(self.graph_name)
        # get output
        self.output_names = self.gpt_inner_inference.get_output_names(self.graph_name)

    def set_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, code_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with stop_audio_token in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        for b in range(len(code_lengths)):
            actual_end = code_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_audio_token
        return mel_input_tokens

    def get_logits(
        self,
        first_inputs,
        first_head,
        second_inputs=None,
        second_head=None,
        prompt=None,
    ):
        if prompt is not None:
            offset = prompt.shape[1]
            if second_inputs is not None:
                emb = torch.cat([prompt, first_inputs, second_inputs], dim=1)
            else:
                emb = torch.cat([prompt, first_inputs], dim=1)

        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        attn_mask = None

        static_seq_len = 256
        emb_padding = F.pad(emb,(0, 0, 0, static_seq_len - emb.shape[1]))
        bmodel_inputs = {self.input_names[0]: emb_padding.numpy()}
        bmodel_outputs = self.gpt_inner_inference.process(self.graph_name, bmodel_inputs)
        last_hidden_state = torch.from_numpy(bmodel_outputs[self.output_names[0]])[:, :emb.shape[1], :]
        gpt_out = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        return enc[:, : first_inputs.shape[1]], enc[:, -second_inputs.shape[1] :]

    def get_style_emb(self, cond_input, return_latent=False):
        """
        cond_input: (b, 80, s) or (b, 1, 80, s)
        conds: (b, 1024, s)
        """
        conds = None
        if not return_latent:
            if cond_input.ndim == 4:
                cond_input = cond_input.squeeze(1)
            conds = self.conditioning_encoder(cond_input)  # (b, d, s)
            conds = self.conditioning_perceiver(conds.permute(0, 2, 1)).transpose(1, 2)  # (b, d, 32)
        else:
            # already computed
            conds = cond_input.unsqueeze(1)
        return conds

    def forward(
        self,
        text_inputs,
        text_lengths,
        audio_codes,
        wav_lengths,
        cond_mels=None,
        cond_idxs=None,
        cond_lens=None,
        cond_latents=None,
    ):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, 1, 80,s)
        cond_idxs: cond start and end indexs, (b, 2)

        """

        max_text_len = text_lengths.max()
        code_lengths = torch.ceil(wav_lengths / self.code_stride_len).long() + 3
        max_mel_len = code_lengths.max()

        if max_mel_len > audio_codes.shape[-1]:
            audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[-1]))

        # 💖 Lovely assertions
        assert (
            max_mel_len <= audio_codes.shape[-1]
        ), f" ❗ max_mel_len ({max_mel_len}) > audio_codes.shape[-1] ({audio_codes.shape[-1]})"
        assert (
            max_text_len <= text_inputs.shape[-1]
        ), f" ❗ max_text_len ({max_text_len}) > text_inputs.shape[-1] ({text_inputs.shape[-1]})"

        # Append stop token to text inputs
        text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1), value=self.stop_text_token)

        # Append silence token to mel codes
        audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=self.stop_audio_token)

        # Pad mel codes with stop_audio_token
        audio_codes = self.set_mel_padding(
            audio_codes, code_lengths - 3
        )  # -3 to get the real code lengths without consider start and stop tokens that was not added yet

        # Build input and target tensors
        # Prepend start token to inputs and append stop token to targets
        text_inputs, text_targets = self.set_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        audio_codes, mel_targets = self.set_inputs_and_targets(
            audio_codes, self.start_audio_token, self.stop_audio_token
        )

        # Compute text embeddings + positional embeddings
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        # Compute mel embeddings + positional embeddings
        mel_emb = self.mel_embedding(audio_codes) + self.mel_pos_embedding(audio_codes)

        # Compute speech conditioning input
        if cond_latents is None:
            cond_latents = self.get_style_emb(cond_mels).transpose(1, 2)

        # Get logits
        sub = -5

        text_logits, mel_logits = self.get_logits(
            text_emb,
            self.text_head,
            mel_emb,
            self.mel_head,
            prompt=cond_latents,
        )
        return mel_logits[:, :sub]  # sub to prevent bla.

    def compute_embeddings(
        self,
        cond_latents,
        text_inputs,
    ):
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)
        emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        emb = torch.cat([cond_latents, emb], dim=1)
        self.gpt_inference.store_prefix_emb(emb)
        gpt_inputs = torch.full(
            (
                emb.shape[0],
                emb.shape[1] + 1,  # +1 for the start_audio_token
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        gpt_inputs[:, -1] = self.start_audio_token
        return gpt_inputs

    def generate(
        self,
        cond_latents,
        text_inputs,
        **hf_generate_kwargs,
    ):
        gpt_inputs = self.compute_embeddings(cond_latents, text_inputs)
        gen = self.gpt_inference.generate(
            gpt_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_mel_tokens + gpt_inputs.shape[-1],
            **hf_generate_kwargs,
        )
        if "return_dict_in_generate" in hf_generate_kwargs:
            return gen.sequences[:, gpt_inputs.shape[1] :], gen
        return gen[:, gpt_inputs.shape[1] :]
