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
import warnings
import numpy as np
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('--model_path', type=str,
                    default="./InternVL2-4B/", help='path to the torch model')

args = parser.parse_args()

model_path = args.model_path
is_2B = "InternVL2-2B" in model_path
is_4B = "InternVL2-4B" in model_path
is_8B = "InternVL2-8B" in model_path
if is_2B:
    model_name = "internvl2-2b"
elif is_4B:
    model_name = "internvl2-4b"
elif is_8B:
    model_name = "internvl2-8b"
else:
    print("Cannot recognize model_type in model_path:", model_path)
    exit(1)
    
execution_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(execution_dir, "../models/onnx/"+model_name) # folder for LLM
if not os.path.exists(folder):
    os.makedirs(folder)

origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="cpu").eval()

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.language_model.model
layers = transformer.layers
SEQ_LENGTH = config.llm_config.max_position_embeddings
NUM_LAYERS = config.llm_config.num_hidden_layers
HIDDEN_SIZE = config.llm_config.hidden_size
NUM_ATTENTION_HEADS = config.llm_config.num_attention_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE = config.llm_config.vocab_size
DOWNSAMPLE_RATIO = config.downsample_ratio
ID_EOS = config.llm_config.eos_token_id
print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')

vit = origin_model.vision_model
VIT_HIDDEN_SIZE = config.vision_config.hidden_size
IMAGE_SIZE = config.vision_config.image_size
CHANNELS = config.vision_config.num_channels

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = transformer.get_input_embeddings()

    def forward(self, input_ids):
        hidden_states = self.embed(input_ids)
        return hidden_states


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

        position_ids = torch.tensor(
            [range(SEQ_LENGTH)], dtype=torch.long)
        value_states = torch.randn(
            (1, SEQ_LENGTH, config.llm_config.num_key_value_heads, HEAD_DIM)).bfloat16()
        if is_4B:
            self.rotary_emb = self.layer.self_attn.rotary_emb
            self.cos, self.sin = self.rotary_emb(
                value_states, position_ids, SEQ_LENGTH)
        else:
            self.rotary_emb = self.layer.attention.rotary_emb
            self.cos, self.sin = self.rotary_emb(value_states, SEQ_LENGTH)
        self.cos = self.cos.view(SEQ_LENGTH, HEAD_DIM)
        self.sin = self.sin.view(SEQ_LENGTH, HEAD_DIM)


    def forward(self, hidden_states, position_ids, attention_mask):
        cos_pos = self.cos[position_ids]
        sin_pos = self.sin[position_ids]
        
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids,
                                            use_cache=True,
                                            rotary_pos_emb_list=(cos_pos, sin_pos),
                                            )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        position_ids = torch.tensor(
            [range(SEQ_LENGTH)], dtype=torch.long)
        value_states = torch.randn(
            (1, SEQ_LENGTH, config.llm_config.num_key_value_heads, HEAD_DIM)).bfloat16()
        if is_4B:
            self.rotary_emb = self.layer.self_attn.rotary_emb
            self.cos, self.sin = self.rotary_emb(
                value_states, position_ids, SEQ_LENGTH)
        else:
            self.rotary_emb = self.layer.attention.rotary_emb
            self.cos, self.sin = self.rotary_emb(value_states, SEQ_LENGTH)
        self.cos = self.cos.view(SEQ_LENGTH, HEAD_DIM)
        self.sin = self.sin.view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        cos_pos = self.cos[position_ids]
        sin_pos = self.sin[position_ids]
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True,
                                            rotary_pos_emb_list=(cos_pos, sin_pos),
                                            )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lm_head = origin_model.language_model.get_output_embeddings()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = self.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token


class VisionTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extract_feature = origin_model.extract_feature

    def forward(self, pixel_values):
        vit_embeds = self.extract_feature(pixel_values)
        return vit_embeds


def convert_vision_transformer():
    model = VisionTransformer()
    pixel_values = torch.randn(
        (1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).bfloat16()
    torch.onnx.export(model, pixel_values,
                      f'{folder}/vision_transformer.onnx',
                      verbose=False,
                      input_names=['pixel_values'],
                      output_names=['vit_embeds'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).bfloat16()
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
    attention_mask = torch.ones(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).bfloat16()
    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).bfloat16()
    position_ids = torch.tensor([range(1)], dtype=torch.long)
    attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1)).bfloat16()
    past_k = torch.randn(
        (1, SEQ_LENGTH, config.llm_config.num_key_value_heads, HEAD_DIM)).bfloat16()
    past_v = torch.randn(
        (1, SEQ_LENGTH, config.llm_config.num_key_value_heads, HEAD_DIM)).bfloat16()

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
    input_ids = torch.tensor([range(SEQ_LENGTH)])

    torch.onnx.export(model, (input_ids),
                      f'{folder}/embedding.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_lm_head():
    model = LmHead()
    input = torch.randn(1, HIDDEN_SIZE).bfloat16()

    torch.onnx.export(model, (input),
                      f'{folder}/lm_head.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['m_logits'],
                      do_constant_folding=True,
                      opset_version=15)


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image)
    return pixel_values


def test_net_with_mask():
    embed = Embedding()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]
    vit_infer = VisionTransformer()
    prefix = "<|system|>\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|end|><|user|>\n<img>"
    prefix_ids = tokenizer.encode(prefix)
    query = "</img>请简单的描述图片中的内容<|end|><|assistant|>\n"
    query_ids = tokenizer.encode(query)
    image_ids = [0] * 256
    prefix_len = len(prefix_ids)
    ids = prefix_ids + image_ids + query_ids
    jpg = "../python_demo/image2.jpg"
    pixel_values = load_image(jpg, max_num=1).to(
        torch.bfloat16)  # [1, 3, 448, 448]
    vit_embeds = vit_infer(pixel_values)  # [1, 256, 3072]
    ID_IM_END = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ID_END = tokenizer.convert_tokens_to_ids("<|end|>")
    token_len = len(ids)
    ids = ids + (SEQ_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SEQ_LENGTH)
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)  # [1, 512, 3072]
    out[:, prefix_len:prefix_len+256, :] = vit_embeds

    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids])
    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float() * -10000.0
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(
        1, 1, SEQ_LENGTH, SEQ_LENGTH)
    k_cache = []
    v_cache = []
    for i in range(NUM_LAYERS):
        out, k, v = blocks[i](out.bfloat16(), position_ids,
                              attention_mask.bfloat16())
        k[:, :, token_len:, :] = 0
        v[:, :, token_len:, :] = 0
        k_cache.append(k)
        v_cache.append(v)

    out = out[:, token_len - 1:token_len].view(1, 1, HIDDEN_SIZE)
    lm = LmHead()
    token = lm(out.bfloat16()).view(1)
    out_ids = [int(token)]
    while int(token) not in [ID_EOS, ID_IM_END, ID_END] and token_len < SEQ_LENGTH:
        token_len += 1
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]])
        attention_mask = torch.zeros(
            (1, 1, 1, SEQ_LENGTH + 1)).float()
        attention_mask[:, :, :, token_len-1:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out.bfloat16(), position_ids,
                                     attention_mask.bfloat16(),
                                     k_cache[i].bfloat16(), v_cache[i].bfloat16())
            k_cache[i][:, token_len-1:token_len, :, :] = k[:, :, :, :]
            v_cache[i][:, token_len-1:token_len, :, :] = v[:, :, :, :]
        token = lm(out.bfloat16()).view(1)
        out_ids.append(int(token))
    words = tokenizer.decode(out_ids)
    print(words)
    print("\noutput_ids:{}".format(out_ids))


# test_net_with_mask()
# exit()

# export models
print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block_cache(i)
    convert_block(i)

print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
convert_lm_head()
print("Done!")

print(f'Convert Vision Transformer')
convert_vision_transformer()
print("Done!")
