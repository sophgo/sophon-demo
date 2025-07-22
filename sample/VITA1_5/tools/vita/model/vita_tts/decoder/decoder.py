import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict, List, Tuple, Optional, Union
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.cache_utils import DynamicCache

from vita.model.vita_tts.encoder.encoder import add_encoder_args
from vita.model.vita_tts.masks import *
IGNORE_ID = -1

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index=-1):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=ignore_index)
        
    def forward(self, logits, target, target_subsampling_factor=1):
        """
        logits: B*T1*D
        target: B*T2
        """
        logits = logits[:, :target.shape[1], :]
        logits = logits.transpose(1, 2)
        target = target.to(torch.long)
        loss = self.criterion(logits, target)
        return loss

class LLM2TTSCodecAR(torch.nn.Module):
    """E2E module.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (namespace): argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Extend arguments for transducer."""
        group = parser.add_argument_group("TDNN model setting")

        group.add_argument('--encoder-pre-norm-type',
                           default='ln', type=str, help="Type of input norm.")
        group.add_argument('--encoder-drop-rate', default=0.0,
                           type=float, help="Dropout rate for output.")
        group.add_argument('--encoder-criterion', default='cross-entropy',
                           type=str, help="Criterion for output")
        group.add_argument('--encoder-upsample-rate', default=1, type=int)
        group.add_argument('--kv-cache-prefix-finetune', default=0, type=int)

        group = add_encoder_args(group)

        return parser

    def __init__(self, idim, odim, args):
        """Initialize transducer modules.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs
            args (Namespace): argument Namespace containing options

        """
        super(LLM2TTSCodecAR, self).__init__()
        self.seq_length = 512
        self.token_length = 0
        self.idim = args.idim
        self.odim = args.odim
        self.encoder_pre_norm_type = args.encoder_pre_norm_type
        self.encoder_drop_rate = args.encoder_drop_rate
        self.encoder_criterion = args.encoder_criterion
        self.encoder_upsample_rate = args.encoder_upsample_rate
        self.reporter = None

        self.vocab_size = self.odim
        config = LlamaConfig(vocab_size=self.vocab_size + 4, hidden_size=args.transformer_attention_dim, 
                            intermediate_size=args.transformer_linear_units, 
                            num_hidden_layers=args.transformer_num_blocks, 
                            num_attention_heads=args.transformer_attention_heads, max_position_embeddings=2048, 
                            bos_token_id=self.vocab_size + 1, 
                            eos_token_id=self.vocab_size + 2, pad_token_id=self.vocab_size + 3,
                            attention_dropout=args.transformer_dropout_rate)
        self.config = config
        self.embedding = nn.Embedding(self.vocab_size + 4, self.idim, padding_idx=self.vocab_size + 3)
        self.init_pre_nn(config)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.out_fnn = nn.Linear(args.encoder_output_dim, self.vocab_size + 4)

        self.kv_cache_prefix_finetune = args.kv_cache_prefix_finetune
        if self.kv_cache_prefix_finetune:
            self.init_kv_cache_prefix(config)
            self.embedding.eval()
            self.layers.eval()
            self.norm.eval()
            self.rotary_emb.eval()
            self.out_fnn.eval()
            for (name, param) in self.embedding.named_parameters():
                param.requires_grad = False
            for (name, param) in self.layers.named_parameters():
                param.requires_grad = False
            for (name, param) in self.norm.named_parameters():
                param.requires_grad = False
            for (name, param) in self.rotary_emb.named_parameters():
                param.requires_grad = False
            for (name, param) in self.out_fnn.named_parameters():
                param.requires_grad = False

        if self.encoder_criterion == 'ce':
            self.criterion = CrossEntropyLoss(ignore_index=self.vocab_size + 3)
    
    def init_kv_cache_prefix(self, config):
        self.layers_prefix = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb_prefix = LlamaRotaryEmbedding(config=config)
    
    def kv_cache_prefix_forward(self, prefix, prefix_lens, past_key_values):
        inputs_embeds = prefix
        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + \
                                      inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb_prefix(hidden_states, position_ids)
        next_decoder_cache = None
        batch_size, max_len, _ = prefix.size()
        input_mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool, device=prefix.device)
        for i in range(batch_size):
            input_mask[i, :prefix_lens[i], :prefix_lens[i]] = True
        attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(inputs_embeds.dtype).min
        for decoder_layer in self.layers_prefix:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            next_decoder_cache = layer_outputs[1]
        past_key_values = next_decoder_cache
    
    def init_pre_nn(self, config):
        self.layers_pre_nn = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers // 2)]
        )
        self.rotary_emb_pre_nn = LlamaRotaryEmbedding(config=config)
    
    def rebuild_modules(self):
        from onnx_rebuilder import Rotary,Decoder,ModelMapper,Lm,GreedyHead
        tts_map = {
            'decoder': {
                'self_attn': 'self_attn',
                'mlp': 'mlp',
                'input_layernorm': 'input_layernorm',
                'post_attention_layernorm': 'post_attention_layernorm'
            },
            'attention': {
                'q_proj': 'q_proj',
                'k_proj': 'k_proj',
                'v_proj': 'v_proj',
                'o_proj': 'o_proj',
            }
        }
        setattr(self.config, "seq_length", self.seq_length)
        rotary = Rotary(self.config)
        setattr(self.config, "rotary", rotary)
        setattr(self.config, "model_map", tts_map)
        for i in range(len(self.layers_pre_nn)):
            self.layers_pre_nn[i] = Decoder(self.layers_pre_nn[i], self.config)
        
        self.layers_new = []
        for i in range(len(self.layers)):
            self.layers_new.append(Decoder(self.layers[i], self.config))
        setattr(self.config, "final_layernorm_", self.norm)
        setattr(self.config, "lm_", self.out_fnn)
        self.lm = Lm(self.config)
        self.greedy = GreedyHead()
        return

    def pre_nn_forward_old(self, hidden, hidden_lens):
        inputs_embeds = hidden
        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + \
                                      inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb_pre_nn(hidden_states, position_ids)
        next_decoder_cache = None
        batch_size, max_len, _ = hidden.size()
        input_mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool, device=hidden.device)
        for i in range(batch_size):
            input_mask[i, :hidden_lens[i], :hidden_lens[i]] = True
        attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(inputs_embeds.dtype).min
        for decoder_layer in self.layers_pre_nn:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        return hidden_states
    
    def pre_nn_forward_pad(self, inputs_embeds, position_ids, export_dir=None):
        hidden_states = inputs_embeds
        attention_mask = torch.ones((self.seq_length, self.seq_length)) * -10000.0
        for i in range(1, self.token_length):
            for j in range(1, self.token_length):
                attention_mask[i][j] = 0.0
        attention_mask = attention_mask.view(1, 1, self.seq_length, self.seq_length)
        i = 0
        for decoder_layer in self.layers_pre_nn:
            if export_dir is not None:
                print("export layer_pre_nn_{}.onnx".format(i))
                torch.onnx.export(
                    decoder_layer,
                    (hidden_states, position_ids, attention_mask),
                    export_dir+f'/layer_pre_nn_{i}.onnx',
                    verbose=False,
                    input_names=["input_states", "position_ids", "attention_mask"],
                    output_names=["hidden_states"],
                    do_constant_folding=False, # set False to keep original name
                    opset_version=15
                )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]
            i += 1
        return hidden_states
    
    def transformer_infer(self, inputs_embeds, cache_position, past_key_values):
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # next_decoder_cache = None
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            # next_decoder_cache = layer_outputs[1]
        return hidden_states
    
    def forward_first(self, inputs_embeds, position_ids, export_dir=None):
        hidden_states = inputs_embeds
        # next_decoder_cache = None
        k_cache, v_cache = [], []
        attention_mask = torch.ones((inputs_embeds.shape[1], inputs_embeds.shape[1])).float() * -10000.0
        for i in range(self.token_length):
            for j in range(self.token_length):
                if j <= i:
                    attention_mask[i][j] = 0.0
        attention_mask = attention_mask.view(1, 1, inputs_embeds.shape[1], inputs_embeds.shape[1])
        for i, decoder_layer in enumerate(self.layers_new):
            hidden_states, kv = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            k, v = kv
            # k[:, self.token_length:, :, :] = 0
            # v[:, self.token_length:, :, :] = 0
            k_cache.append(k)
            v_cache.append(v)
            if export_dir is not None:
                print("export tts_block_{}.onnx".format(i))
                torch.onnx.export(
                    decoder_layer,
                    (hidden_states, position_ids, attention_mask),
                    export_dir+f'/block_{i}.onnx',
                    verbose=False,
                    input_names=["hidden_states", "position_ids", "attention_mask"],
                    output_names=["hidden_states", "past_k", "past_v"],
                    do_constant_folding=False, # set False to keep original name
                    opset_version=15
                )
        return hidden_states, k_cache, v_cache
    
    def forward_next(self, inputs_embeds, position_ids, k_cache, v_cache, export_dir=None):
        hidden_states = inputs_embeds
        # next_decoder_cache = None
        attention_mask = torch.zeros((1, 1, 1, k_cache[0].shape[1] + 1))
        attention_mask[:, :, :, self.token_length-1:] = -10000.0
        for i, decoder_layer in enumerate(self.layers_new):
            hidden_states, kv = decoder_layer(
                hidden_states,
                position_ids,
                attention_mask,
                (k_cache[i], v_cache[i])
            )
            k, v = kv
            k_cache[i][:, self.token_length-1:self.token_length, :, :] = k[:, :, :, :]
            v_cache[i][:, self.token_length-1:self.token_length, :, :] = v[:, :, :, :]
            if export_dir is not None:
                onnx_path = export_dir+f'/block_cache_{i}.onnx'
                if os.path.exists(onnx_path):
                    continue
                print("export tts_block_cache_{}.onnx".format(i))
                torch.onnx.export(
                    decoder_layer,
                    (hidden_states, position_ids, attention_mask, (k_cache[i], v_cache[i])),
                    onnx_path,
                    verbose=False,
                    input_names=["input_states", "position_ids", "attention_mask", "history_k", "history_v"],
                    output_names=["hidden_states", "past_k", "past_v"],
                    do_constant_folding=False, # set False to keep original name
                    opset_version=15
                )
        return hidden_states

    def infer_old(self, hidden, top_k, prefix, penalty_window_size, penalty, max_tokens=1000, export_dir=None):
        # Pass through pre_nn
        hidden = self.pre_nn_forward(hidden, [hidden.size(1)])
        # Concat bos embedding
        bos_emb = self.embedding(torch.full((1, 1), self.vocab_size, dtype=torch.long, device=hidden.device))
        hidden = torch.cat([bos_emb, hidden], dim=1)
        # init past key values
        past_key_values = DynamicCache.from_legacy_cache(None)
        # Pass through the prefix nar decoder
        if prefix is not None and self.kv_cache_prefix_finetune:
            self.kv_cache_prefix_forward(prefix, [prefix.size(1)], past_key_values)
        inputs_embeds = hidden
        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], \
                                      device=inputs_embeds.device)
        hidden_states = self.transformer_infer(inputs_embeds, cache_position, past_key_values)

        # init generated tokens
        cur_token = torch.full((1, 1), self.vocab_size + 1, dtype=torch.long, device=hidden.device)
        generated_tokens = torch.full((1, 1), self.vocab_size + 1, dtype=torch.long, device=hidden.device)
        # generate tokens
        for i in range(max_tokens):
            inputs_embeds = self.embedding(cur_token)
            past_seen_tokens = past_key_values.get_seq_length()
            if prefix is not None:
                past_seen_tokens = past_seen_tokens - prefix.size(1)
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], \
                                          device=inputs_embeds.device)
            hidden_states = self.transformer_infer(inputs_embeds, cache_position, past_key_values)
            hidden_states = self.norm(hidden_states)

            # Project to vocabulary size
            logits = self.out_fnn(hidden_states)

            # apply penalty
            if penalty_window_size > 0:
                for token in set(generated_tokens[0][-penalty_window_size:]):
                    logits[:, :, token] /= penalty

            # top k sampling
            output = logits.squeeze(0).squeeze(0)
            probs = torch.nn.functional.softmax(output, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
            probs = probs / probs.sum()
            next_token_id = torch.multinomial(probs, 1).unsqueeze(0)

            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
            cur_token = next_token_id

            # eos
            if next_token_id == self.vocab_size + 2:
                break
            yield next_token_id
            
    def infer(self, hidden, top_k, prefix, penalty_window_size, penalty, max_tokens=1000, export_dir=None):
        # Pass through pre_nn
        # Concat bos embedding
        if export_dir is not None:
            print("export tts_embedding.onnx")
            input = torch.full((1, 1), self.vocab_size, dtype=torch.long, device=hidden.device)
            torch.onnx.export(
                self.embedding,
                input,
                export_dir+f'/tts_embedding.onnx',
                verbose=False,
                input_names=["input"],
                output_names=["output"],
                do_constant_folding=False, # set False to keep original name
                opset_version=15
            )
        self.token_length = hidden.shape[1] + 1 
        position_ids = [0] + list(range(self.token_length)) + (self.seq_length - self.token_length - 1) * [0]
        position_ids = torch.tensor([position_ids])
        # hidden_ = self.pre_nn_forward(hidden, [hidden.size(1)], export_dir)
        hidden = torch.nn.functional.pad(hidden,(0, 0, 1, self.seq_length - hidden.shape[1] - 1))
        hidden = self.pre_nn_forward_pad(hidden, position_ids, export_dir)
        bos_emb = self.embedding(torch.full((1, 1), self.vocab_size, dtype=torch.long, device=hidden.device))
        hidden[0][0] = bos_emb

        # init past key values
        past_key_values = DynamicCache.from_legacy_cache(None)
        # Pass through the prefix nar decoder
        if prefix is not None and self.kv_cache_prefix_finetune:
            self.kv_cache_prefix_forward(prefix, [prefix.size(1)], past_key_values)
        past_seen_tokens = 0
        # cache_position = torch.arange(past_seen_tokens, past_seen_tokens + hidden.shape[1], \
        #                               device=hidden.device)
        print("forward_first")
        # hidden_states = self.transformer_infer(hidden, cache_position, past_key_values)
        position_ids = list(range(self.token_length)) + (self.seq_length - self.token_length) * [0]
        position_ids = torch.tensor([position_ids])
        _, k_cache, v_cache = self.forward_first(hidden, position_ids, export_dir)
        # init generated tokens
        cur_token = torch.full((1, 1), self.vocab_size + 1, dtype=torch.long, device=hidden.device)
        generated_tokens = torch.full((1, 1), self.vocab_size + 1, dtype=torch.long, device=hidden.device)
        # generate tokens
        print("forward_next")

        for i in range(max_tokens):
            self.token_length += 1
            if self.token_length == self.seq_length:
                print("exceed seq_length")
                break
            inputs_embeds = self.embedding(cur_token)
            past_seen_tokens = past_key_values.get_seq_length()
            if prefix is not None:
                past_seen_tokens = past_seen_tokens - prefix.size(1)
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], \
                                          device=inputs_embeds.device)
            position_ids = torch.tensor([[self.token_length - 1]])
            # hidden_states = self.transformer_infer(inputs_embeds, cache_position, past_key_values)
            hidden_states = self.forward_next(inputs_embeds, position_ids, k_cache, v_cache, export_dir)
            logits = self.lm(hidden_states)
            _, next_token_id = torch.topk(logits, 1)
            next_token_id = next_token_id.squeeze(0)
            cur_token = next_token_id
            
            # Project to vocabulary size
            # hidden_states = self.norm(hidden_states)
            # logits = self.out_fnn(hidden_states)
            # apply penalty
            # if penalty_window_size > 0:
            #     for token in set(generated_tokens[0][-penalty_window_size:]):
            #         logits[:, :, token] /= penalty

            # # top k sampling
            # output = logits.squeeze(0).squeeze(0)
            # probs = torch.nn.functional.softmax(output, dim=-1)
            # top_k_probs, top_k_indices = torch.topk(probs, top_k)
            # probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
            # probs = probs / probs.sum()
            # next_token_id = torch.multinomial(probs, 1).unsqueeze(0)

            # generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
            # cur_token = next_token_id

            # eos
            if next_token_id == self.vocab_size + 2:
                break
            yield next_token_id
        if export_dir is not None:
            print("exporting lm and greedy")
            lmhead_file = f'{export_dir}/lm_head.pt' 
            module = torch.jit.trace(self.lm.forward, hidden_states)
            torch.jit.save(module, lmhead_file)
            greedy_file = f'{export_dir}/greedy_head.onnx'
            torch.onnx.export(
                self.greedy, (logits),
                greedy_file,
                verbose=False,
                input_names=['m_logits'],
                output_names=['token'],
                do_constant_folding=True,
                opset_version=15)