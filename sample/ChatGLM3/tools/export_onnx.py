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
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser(description='export onnx.')
parser.add_argument('--model_path', type=str, default='/workspace/open-source/chatglm3-6b', help='path to chatglm3-6b model folder.')
parser.add_argument('--onnx_path', type=str, default='./models/onnx/', help='output folder for chatglm3-6b onnxs')

args = parser.parse_args()

model_path = args.model_path
folder = args.onnx_path

origin_model = AutoModel.from_pretrained(model_path,
                                         trust_remote_code=True).float().eval()
config = origin_model.config
transformer = origin_model.transformer
layers = transformer.encoder.layers

SEQ_LENGTH = transformer.seq_length
NUM_LAYERS = config.num_layers
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
        return transformer.embedding.word_embeddings(input_ids)


class GlmBlock(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask):
        rotary_pos_emb = transformer.rotary_pos_emb(SEQ_LENGTH)[position_ids]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            rotary_pos_emb=rotary_pos_emb)
        return hidden_states, past_kv


class GlmBlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        rotary_pos_emb = transformer.rotary_pos_emb(SEQ_LENGTH)[position_ids]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            kv_cache=(past_k, past_v),
                                            rotary_pos_emb=rotary_pos_emb)
        present_k, present_v = past_kv
        return hidden_states, present_k[1:], present_v[1:]


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.encoder.final_layernorm(hidden_states)
        m_logits = transformer.output_layer(hidden_states)
        _, token = torch.topk(m_logits, 1)
        return token


def convert_block(layer_id):
    # input
    hidden_states = torch.randn((SEQ_LENGTH, 1, HIDDEN_SIZE))
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
    attention_mask = torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH),
                                dtype=torch.bool).tril(diagonal=0)
    model = GlmBlock(layer_id)
    # hiddeng_states = model(input_ids, position_ids)

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
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE))
    position_ids = torch.tensor([range(1)], dtype=torch.long)
    attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1),
                                dtype=torch.bool).tril(diagonal=0)
    past_k = torch.randn((SEQ_LENGTH, 1, 2, HEAD_DIM))
    past_v = torch.randn((SEQ_LENGTH, 1, 2, HEAD_DIM))
    model = GlmBlockCache(layer_id)
    # hiddeng_states = model(input_ids, position_ids)

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