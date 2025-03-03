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
import sys
import torch
import argparse
from tqdm import tqdm
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from support.janus import MultiModalityCausalLM, VLChatProcessor

torch.set_grad_enabled(False)
parser = argparse.ArgumentParser(description='export onnx.')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model.')
parser.add_argument('-s', '--seq_length', type=int, default=2048, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument('-f', '--folder', type=str, default='../models/onnx')
args = parser.parse_args()

device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
else:
    dtype = torch.float16

processor = VLChatProcessor.from_pretrained(args.model_path)
tokenizer = processor.tokenizer

origin_model = MultiModalityCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map=device
).eval()

folder = args.folder
if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(folder+'/vit')

for param in origin_model.parameters():
    param.requires_grad = False

llm = origin_model.language_model
mlp = origin_model.aligner
vit = origin_model.vision_model
config = llm.config
transformer = llm.model
layers = transformer.layers

# text config
IMAGE_SIZE = 384
SEQ_LENGTH = args.seq_length
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE = config.vocab_size
print(f'Layers: {NUM_LAYERS}\n\
Hidden size: {HIDDEN_SIZE}\n\
Query heads: {NUM_ATTENTION_HEADS}\n\
KV heads: {NUM_KEY_VALUE_HEADS}\n')

class VisionTransformer(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pixel_values):
        images = vit(pixel_values)
        images_embeds = mlp(images)
        return images_embeds


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return transformer.embed_tokens(input_ids)


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = self.layer.self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype)
        position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids,
                                            position_embeddings=(self.cos, self.sin),
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = self.layer.self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype)
        position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            position_embeddings=(self.cos, self.sin),
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states) 
        m_logits = llm.lm_head(hidden_states)
        return m_logits


class GreedyHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, m_logits):
        _, token = torch.topk(m_logits.float(), 1)
        return token

# refs:https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
class PenaltySampleHead(torch.nn.Module):

    def __init__(self, top_k = 50, min_tokens_to_keep = 5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, :self.min_tokens_to_keep] = True

    def forward(self, m_logits, input_ids, top_p, temperature, penalty):
        # repeat penalty
        logits = torch.gather(m_logits, 1, input_ids)
        logits = torch.where(logits < 0, logits * penalty, logits / penalty)
        m_logits.scatter_(1, input_ids, logits)

        # top_k
        logits, token = torch.topk(m_logits.float(), self.top_k)

        # temperature
        logits = logits / temperature

        # top_p
        cumulative_probs = logits.softmax(dim=1).cumsum(dim=1)
        mask = cumulative_probs < top_p
        mask = mask + self.keep_matrix
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token


def convert_vision_transformer():
    model = VisionTransformer()
    pixel_values = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE)).to(dtype)

    torch.onnx.export(
        model, (pixel_values),
        f'{folder}/vit/vision_transformer.onnx',
        verbose=False,
        input_names=["pixel_values"],
        output_names=['images_embed'],
        do_constant_folding=True)

def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.ones(1, 1, SEQ_LENGTH, SEQ_LENGTH).to(dtype).to(device)
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
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).to(dtype).to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM))
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM))

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
    input_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.int32).to(device)

    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')

def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, HIDDEN_SIZE).to(dtype).to(device)

    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')

def convert_greedy_head():   
    model = GreedyHead()
    m_logits = torch.randn(1, VOCAB_SIZE).to(dtype).to(device)

    torch.onnx.export(
        model, (m_logits),
        f'{folder}/greedy_head.onnx',
        verbose=False,
        input_names=['m_logits'],
        output_names=['token'],
        do_constant_folding=True,
        opset_version=15)

def convert_penalty_sample_head():   
    model = PenaltySampleHead()
    m_logits = torch.randn(1, VOCAB_SIZE).to(dtype).to(device)
    input_ids = torch.tensor([range(SEQ_LENGTH)]).to(device)
    top_p = torch.tensor([0.8]).to(device)
    temperature = torch.tensor([0.98]).to(device)
    penalty = torch.tensor([0.98]).to(device)

    torch.onnx.export(
        model, (m_logits, input_ids, top_p, temperature, penalty),
        f'{folder}/penalty_sample_head.onnx',
        verbose=False,
        input_names=[
            'm_logits', 'input_ids', 'top_p', 'temperature',
            'penalty'
        ],
        output_names=['probs', 'token'],
        do_constant_folding=True,
        opset_version=15)

def test_net_with_mask(image_path):
    query = "describe this image"
    images = [Image.open(image_path).convert("RGB")]
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{query}",
            "images": [image_path],
        },
        {
            "role": "<|Assistant|>",
            "content": ""
        },
    ]
    print(f'image: "{image_path}"')
    print(f'query: "{query}"\n')

    inputs = processor(
        conversations=conversation,
        images=images
    )

    # init inputs
    token_len = inputs['input_ids'].shape[1]
    input_ids = torch.zeros(SEQ_LENGTH).unsqueeze(0).to(torch.int32)
    input_ids[:,:token_len] = inputs['input_ids'].to(torch.int32)
    input_ids[input_ids < 0] = 0 # ignore the image embeddings
    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids])
    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float() * -10000.0
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(1, 1, SEQ_LENGTH, SEQ_LENGTH)

    # init models
    vision = VisionTransformer()
    embed = Embedding()
    lm_head = LmHead()
    greedy_head = GreedyHead()
    blocks = []
    block_kvs = []
    for i in range(NUM_LAYERS):
        blocks.append(Block(i))
        block_kvs.append(BlockCache(i))

    # inference
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)
    images_embed = vision(inputs['pixel_values'].squeeze(0))
    # system prompt 0~41
    # valid image token 42~617
    # user input 618~token_len
    out[:,42:618,:] = images_embed
    k_cache = []
    v_cache = []
    for i in range(NUM_LAYERS):
        out, k, v = blocks[i](out, position_ids, attention_mask)
        k_cache.append(k)
        v_cache.append(v)
    out = out[:, token_len - 1: token_len].view(1, 1, HIDDEN_SIZE)
    token = greedy_head(lm_head(out)).view(1)
    out_ids = []
    while int(token) != processor.tokenizer.eos_token_id:
        out_ids.append(int(token))
        word = processor.tokenizer.decode([int(token)])
        print(word, end="")
        token_len += 1
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]])
        attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float()
        attention_mask[:, :, :, token_len-1:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out, position_ids, attention_mask, k_cache[i], v_cache[i])
            k_cache[i][:,token_len-1:token_len] = k
            v_cache[i][:,token_len-1:token_len] = v
        token = greedy_head(lm_head(out)).view(1)
    print("\noutput_ids:{}".format(out_ids))

# test_net_with_mask('../python_demo/test.jpg')

print(f'Convert vision transformer')
convert_vision_transformer()

print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block(i)
    convert_block_cache(i)
 
print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
convert_lm_head()
convert_greedy_head()
convert_penalty_sample_head()
print("Done")
