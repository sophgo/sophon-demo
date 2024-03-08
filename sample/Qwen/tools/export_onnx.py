#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

parser = argparse.ArgumentParser(description='export onnx.')
parser.add_argument('--model_path', type=str, help='path to the torch model.')
parser.add_argument('--onnx_path', type=str, help='path to save onnx model.')

args = parser.parse_args()

model_path = args.model_path
folder = args.onnx_path

if not torch.cuda.is_available():
    raise ValueError("Need a GPU for load model and export onnx type.")

device = torch.device("cuda:0")
origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto").eval()
config = origin_model.config
transformer = origin_model.transformer
layers = transformer.h

SEQ_LENGTH = config.seq_length
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS

for param in origin_model.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.wte(input_ids)
        return out.float()


class QwenBlock(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = transformer.rotary_emb(SEQ_LENGTH)
        self.cos_emb = self.rotary_emb[0].view(SEQ_LENGTH, HEAD_DIM)
        self.sin_emb = self.rotary_emb[1].view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask):
        cos_pos = self.cos_emb[position_ids].unsqueeze(2)
        sin_pos = self.sin_emb[position_ids].unsqueeze(2)
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb_list=[[cos_pos, sin_pos]],
            # registered_causal_mask=attention_mask,
            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class QwenBlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = transformer.rotary_emb(SEQ_LENGTH)
        self.cos_emb = self.rotary_emb[0].view(SEQ_LENGTH, HEAD_DIM)
        self.sin_emb = self.rotary_emb[1].view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        cos_pos = self.cos_emb[position_ids].unsqueeze(2)
        sin_pos = self.sin_emb[position_ids].unsqueeze(2)
        hidden_states, past_kv = self.layer(
            hidden_states,
            layer_past=(past_k, past_v),
            attention_mask=attention_mask,
            rotary_pos_emb_list=[[cos_pos, sin_pos]],
            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.ln_f(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token


def convert_block(layer_id):
    # input
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).bfloat16().to(device)
    position_ids = torch.tensor(
        [range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).bfloat16().to(device)
    model = QwenBlock(layer_id)
    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_block_cache(layer_id):
    # input
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).bfloat16().to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).bfloat16().to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).bfloat16().to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).bfloat16().to(device)
    model = QwenBlockCache(layer_id)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        f'{folder}/block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=[
            'input_states', 'position_ids', 'attention_mask', 'history_k',
            'history_v'
        ],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_embedding():
    model = Embedding()
    input = torch.tensor([range(SEQ_LENGTH)]).to(device)
    torch.onnx.export(model, (input),
                      f'{folder}/embedding.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_lm_head():
    model = LmHead()
    input = torch.randn(1, HIDDEN_SIZE).bfloat16().to(device)
    torch.onnx.export(model, (input),
                      f'{folder}/lm_head.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['token'],
                      do_constant_folding=True,
                      opset_version=15)

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
for i in range(NUM_LAYERS):
    print("convert_block_{}".format(i))
    convert_block_cache(i)
    convert_block(i)

print("convert_embedding")
convert_embedding()

print("convert_lm_head")
convert_lm_head()

