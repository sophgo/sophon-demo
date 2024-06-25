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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='export onnx.')
parser.add_argument('--model_path', type=str, help='path to the torch model.')
parser.add_argument('--seq_length', type=int, default=512, help="sequence length")

args = parser.parse_args()

model_path = args.model_path
folder = "./models/onnx"

origin_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval()
origin_model.generation_config = GenerationConfig.from_pretrained(model_path)
config = origin_model.config
transformer = origin_model.model
layers = transformer.layers

SEQ_LENGTH = args.seq_length
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS

for param in origin_model.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return transformer.embed_tokens(input_ids)


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids,
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits, 1)
        return token


def convert_block(layer_id):
    # input
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE))
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
    attention_mask = torch.randn((1, 1, SEQ_LENGTH, SEQ_LENGTH))
    model = Block(layer_id)

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
    np.random.seed(42)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE))
    position_ids = torch.tensor([range(1)], dtype=torch.long)
    attention_mask = torch.randn((1, 1, 1, SEQ_LENGTH + 1))
    past_k = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM))
    past_v = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM))
    model = BlockCache(layer_id)

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
    input = torch.tensor([range(SEQ_LENGTH)])
    torch.onnx.export(model, (input),
                      f'{folder}/embedding.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      dynamic_axes={"input_ids": {
                          0: "length"
                      }},
                      do_constant_folding=True,
                      opset_version=15)


def convert_lm_head():
    model = LmHead()
    input = torch.randn(1, HIDDEN_SIZE)
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

