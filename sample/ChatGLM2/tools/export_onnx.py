#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

#export PYTHONPATH=/workspace/chatglm2-6b:$PYTHONPATH
import datetime
import math
import unittest
import torch
import random
import sys
from transformers import AutoModel, AutoTokenizer
from tokenization_chatglm import ChatGLMTokenizer
import os
import numpy as np

import argparse


parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('--path', type=str, default='/workspace/chatglm2-6b', help='path of chatglm2-6b')
CHATGLM2_PATH = parser.parse_args().path

# CHATGLM2_PATH = "/workspace/chatglm2-6b"
folder = "models/onnx/"

origin_model = AutoModel.from_pretrained(CHATGLM2_PATH,
                                         trust_remote_code=True).float()
origin_model.eval()
transformer = origin_model.transformer
MAX_LEN = transformer.seq_length
for param in origin_model.parameters():
    param.requires_grad = False
num_layers = transformer.encoder.num_layers
layers = transformer.encoder.layers
tokenizer = AutoTokenizer.from_pretrained(CHATGLM2_PATH, trust_remote_code=True)


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
        rotary_pos_emb = transformer.rotary_pos_emb(MAX_LEN)[position_ids]
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
        rotary_pos_emb = transformer.rotary_pos_emb(MAX_LEN)[position_ids]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            kv_cache=(past_k, past_v),
                                            rotary_pos_emb=rotary_pos_emb)
        past_k, past_v = past_kv
        return hidden_states, past_k[1:], past_v[1:]


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.encoder.final_layernorm(hidden_states)
        m_logits = transformer.output_layer(hidden_states)
        _, token = torch.topk(m_logits, 1)
        return token


def convert_glm_block(layer_id):
    # input
    hidden_states = torch.randn((MAX_LEN, 1, 4096))
    position_ids = torch.tensor([range(MAX_LEN)], dtype=torch.long)
    attention_mask = torch.ones((1, 1, MAX_LEN, MAX_LEN),
                                dtype=torch.bool).tril(diagonal=0)
    model = GlmBlock(layer_id)
    # hiddeng_states = model(input_ids, position_ids)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        folder + f'glm_block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_glm_block_cache(layer_id):
    # input
    hidden_states = torch.randn((1, 1, 4096))
    position_ids = torch.tensor([range(1)], dtype=torch.long)
    attention_mask = torch.ones((1, 1, 1, MAX_LEN + 1),
                                dtype=torch.bool).tril(diagonal=0)
    past_k = torch.randn((MAX_LEN, 1, 2, 128))
    past_v = torch.randn((MAX_LEN, 1, 2, 128))
    model = GlmBlockCache(layer_id)
    # hiddeng_states = model(input_ids, position_ids)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        folder+f'glm_block_cache_{layer_id}.onnx',
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
    torch.onnx.export(model, (torch.tensor([0, 1, 2, 3])),
                      folder+f'embedding.onnx',
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
    input = torch.randn(1, 4096)
    torch.onnx.export(model, (input),
                      folder+f'lm_head.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['token'],
                      do_constant_folding=True,
                      opset_version=15)


def test_net_with_mask():
    embed = Embedding()
    blocks = [GlmBlock(i) for i in range(num_layers)]
    block_kvs = [GlmBlockCache(i) for i in range(num_layers)]
    ids = tokenizer.encode('你好')
    query = '你好'
    print(query)
    promt = tokenizer.build_prompt(query)
    ids = tokenizer.encode(promt)
    print("input ids:{}".format(ids))
    token_len = len(ids)
    ids = ids + (MAX_LEN - token_len) * [0]
    input_ids = torch.tensor(ids).view(MAX_LEN)
    out = embed(input_ids).view(MAX_LEN, 1, 4096)
    position_ids = list(range(token_len)) + (MAX_LEN - token_len) * [0]
    position_ids = torch.tensor([position_ids])
    attention_mask = torch.ones((MAX_LEN, MAX_LEN))
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0
    attention_mask = attention_mask.view(1, 1, MAX_LEN, MAX_LEN).bool()
    k_cache = []
    v_cache = []

    for i in range(num_layers):
        out, kv_cache = blocks[i](out, position_ids, attention_mask)
        k, v = kv_cache
        k[MAX_LEN - token_len:] = k[:token_len]
        v[MAX_LEN - token_len:] = v[:token_len]
        k[:MAX_LEN - token_len] = 0
        v[:MAX_LEN - token_len] = 0
        k_cache.append(k)
        v_cache.append(v)
    out = out[token_len - 1:token_len].view(1, 4096)
    lm = LmHead()
    token = lm(out).view(1)
    out_ids = [int(token)]
    word = tokenizer._convert_id_to_token(int(token[0]))
    print(word, end="")
    while token > 2 and token_len < 64:
        token_len += 1
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, 4096)
        position_ids = torch.tensor([[token_len - 1]])
        attention_mask = torch.ones((1, 1, 1, MAX_LEN + 1))
        attention_mask[:, :, :, MAX_LEN + 1 - token_len:] = 0
        for i in range(num_layers):
            out, k_cache[i], v_cache[i] = block_kvs[i](out, position_ids,
                                                       attention_mask,
                                                       k_cache[i], v_cache[i])
            k_cache[i][:MAX_LEN - token_len] = 0
            v_cache[i][:MAX_LEN - token_len] = 0
        token = lm(out).view(1)
        out_ids.append(int(token))
        word = tokenizer._convert_id_to_token(int(token[0]))
        print(word, end="")
    print("\noutput_ids:{}".format(out_ids))


#test_net_with_mask()

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

#export models
for i in range(num_layers):
    print("convert_block_{}".format(i))
    convert_glm_block_cache(i)
    convert_glm_block(i)
convert_embedding()
convert_lm_head()
