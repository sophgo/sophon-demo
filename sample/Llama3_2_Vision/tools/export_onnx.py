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
import transformers
from tqdm import tqdm
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx.')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model.')
parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument('-f', '--folder', type=str, default='./tmp/onnx')
args = parser.parse_args()

device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
else:
    dtype = torch.bfloat16

model = MllamaForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=dtype,
    device_map=device,
)
processor = AutoProcessor.from_pretrained(args.model_path)

folder = args.folder
if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(folder+'/vit')

origin_model = model.eval()
for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
llm = origin_model.language_model
transformer = llm.model
layers = transformer.layers
vision = origin_model.vision_model
linear = origin_model.multi_modal_projector

# text config
SEQ_LENGTH = args.seq_length
NUM_LAYERS = config.text_config.num_hidden_layers
HIDDEN_SIZE = config.text_config.hidden_size
NUM_ATTENTION_HEADS = config.text_config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.text_config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE = config.text_config.vocab_size
CROSS_ATTN_LAYERS = config.text_config.cross_attention_layers
print(f'\nLLM config:\n\
    Layers: {NUM_LAYERS}\n\
    Hidden size: {HIDDEN_SIZE}\n\
    Query heads: {NUM_ATTENTION_HEADS}\n\
    KV heads: {NUM_KEY_VALUE_HEADS}\n\
    CrossAttn layers: {CROSS_ATTN_LAYERS}')

# vision config
IMAGE_SIZE = config.vision_config.image_size
NUM_TILES = config.vision_config.max_num_tiles
PATCH_SIZE = config.vision_config.patch_size
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2 + 1
PADDING_PATCHES = (8 - (NUM_PATCHES % 8)) % 8
print(f'\nVIT config:\n\
    Image size: {IMAGE_SIZE}\n\
    Max tiles: {NUM_TILES}\n\
    Patch size: {PATCH_SIZE}\n\
    Num Patches: {NUM_PATCHES}\n')


class VisionTransformer(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask):
        out = vision(pixel_values, aspect_ratio_ids, aspect_ratio_mask)[0]
        out = linear(out).reshape(NUM_TILES, NUM_PATCHES, HIDDEN_SIZE)
        return out


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
        hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
        position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
        self.cos, self.sin = transformer.rotary_emb(hidden_states, position_ids)

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                              attention_mask=attention_mask,
                                              position_ids=position_ids,
                                              use_cache=True,
                                              position_embeddings=(self.cos, self.sin))
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
        position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
        self.cos, self.sin = transformer.rotary_emb(hidden_states, position_ids)

    def forward(self, hidden_states, position_ids, attention_mask, past_k, past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                               attention_mask=attention_mask,
                                               position_ids=position_ids,
                                               past_key_value=(past_k, past_v),
                                               use_cache=True,
                                               position_embeddings=(self.cos, self.sin))
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class CrossBlock(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, cross_attention_states, text_row_mask,
    attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            cross_attention_states=cross_attention_states,
                                            full_text_row_masked_out_mask=text_row_mask,
                                            cross_attention_mask=attention_mask,
                                            attention_mask=None,
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class CrossBlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, attention_mask, past_k, past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            cross_attention_states=None,
                                            full_text_row_masked_out_mask=None,
                                            cross_attention_mask=attention_mask,
                                            attention_mask=None,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True)
        return hidden_states


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
    pixel_values = torch.randn(
        (NUM_TILES, 3, IMAGE_SIZE, IMAGE_SIZE)).to(dtype).to(device)
    aspect_ratio_ids = torch.randint(1, 5, (1,1), dtype=torch.int32)
    aspect_ratio_mask = torch.randint(0, 2, (1,4), dtype=torch.int32)

    torch.onnx.export(
        model, (pixel_values, aspect_ratio_ids, aspect_ratio_mask),
        f'{folder}/vit/vision_transformer.onnx',
        verbose=False,
        input_names=["pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"],
        output_names=['cross_attention_states'],
        do_constant_folding=True,
        opset_version=15
    )

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

def convert_cross_block(layer_id):
    model = CrossBlock(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE))
    cross_attention_states = torch.randn((NUM_TILES, NUM_PATCHES, HIDDEN_SIZE))
    text_row_mask = torch.ones((1, SEQ_LENGTH, 1))
    attention_mask = -1000 * torch.ones((1, 1, SEQ_LENGTH, NUM_TILES * NUM_PATCHES))

    torch.onnx.export(
        model, (hidden_states, cross_attention_states, text_row_mask, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'cross_attention_states', 'text_row_mask', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)

def convert_cross_block_cache(layer_id):
    model = CrossBlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE))
    attention_mask = -1000 * torch.ones((1, 1, 1, NUM_TILES * NUM_PATCHES))
    history_k = torch.randn((1, NUM_ATTENTION_HEADS, NUM_TILES * NUM_PATCHES, HEAD_DIM))
    history_v = torch.randn((1, NUM_ATTENTION_HEADS, NUM_TILES * NUM_PATCHES, HEAD_DIM))

    torch.onnx.export(
        model, (hidden_states, attention_mask, history_k, history_v),
        f'{folder}/block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'attention_mask', 'history_k', 'history_v'],
        output_names=['hidden_states'],
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

def prepare_cross_attention_mask(
    cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
    dtype: str,
):
    # reshape so it can be used by attn module
    text_total_length = cross_attention_mask.shape[1]
    cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
    cross_attention_mask = cross_attention_mask.view(1, 1, text_total_length, -1)

    # invert the mask
    inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(
        inverted_cross_attn_mask.to(torch.bool), -10000.
    )

    # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = -10000.
    full_text_row_masked_out_mask = (
        (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
    )
    cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask

def test_net_with_mask(image_path):
    # prepare inputs
    image = Image.open(image_path)
    query = 'What does this picture describe'
    print(f'image: "{image_path}"')
    print(f'query: "{query}"\n')
    messages = [
        {"role": "user", "content": [
            {"type": "image"}
        ]}
    ]
    messages.append({"role":"user","content":[{"type": "text", "text": query}]})
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt")
    token_len = inputs['input_ids'].shape[1]
    ori_token_len = token_len
    input_ids = torch.zeros(SEQ_LENGTH).unsqueeze(0).to(torch.long)
    input_ids[:,:token_len] = inputs['input_ids'].to(torch.long)
    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids])
    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float() * -10000.0
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(1, 1, SEQ_LENGTH, SEQ_LENGTH)
    cross_attn_mask = torch.ones((1, 1, SEQ_LENGTH, NUM_TILES*NUM_PATCHES)).float() * -10000.0
    text_row_mask = torch.zeros((1, SEQ_LENGTH, 1)).float()
    # valid text token start from 6
    text_row_mask[:,6:token_len] = 1.
    cross_attn_mask[:,:,:token_len] = prepare_cross_attention_mask(
        inputs['cross_attention_mask'], NUM_PATCHES, dtype)

    # init models
    vit = VisionTransformer()
    embed = Embedding()
    lm_head = LmHead()
    greedy_head = GreedyHead()
    blocks = []
    block_kvs = []
    for i in range(NUM_LAYERS):
        if i not in CROSS_ATTN_LAYERS:
            blocks.append(Block(i))
            block_kvs.append(BlockCache(i))
        else:
            blocks.append(CrossBlock(i))
            block_kvs.append(CrossBlockCache(i))

    # inference
    vit_out = vit(inputs['pixel_values'].squeeze(0).squeeze(0),
                  inputs['aspect_ratio_ids'],
                  inputs['aspect_ratio_mask'].squeeze(0))
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)
    k_cache = []
    v_cache = []
    for i in range(NUM_LAYERS):
        if i not in CROSS_ATTN_LAYERS:
            out, k, v = blocks[i](out, position_ids, attention_mask)
            k_cache.append(k)
            v_cache.append(v)
        else:
            out, k, v = blocks[i](out, vit_out, text_row_mask, cross_attn_mask)
            k_cache.append(k)
            v_cache.append(v)
    out = out[:, token_len - 1: token_len].view(1, 1, HIDDEN_SIZE)
    token = greedy_head(lm_head(out)).view(1)
    out_ids = [int(token)]
    word = processor.decode([int(token)])
    print(word, end="")
    while int(token) != processor.tokenizer.eos_token_id and token_len < ori_token_len + 50:
        token_len += 1
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]])
        attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float()
        attention_mask[:, :, :, token_len:SEQ_LENGTH] = -10000.0
        cross_attn_mask = prepare_cross_attention_mask(
            inputs['cross_attention_mask'], NUM_PATCHES, dtype)[:,:,-1]
        for i in range(NUM_LAYERS):
            if i not in CROSS_ATTN_LAYERS:
                out, k, v = block_kvs[i](out, position_ids, attention_mask, k_cache[i], v_cache[i])
                k_cache[i][:,token_len:token_len+1] = k
                v_cache[i][:,token_len:token_len+1] = v
            else:
                out = block_kvs[i](out, cross_attn_mask, k_cache[i], v_cache[i])
        token = greedy_head(lm_head(out)).view(1)
        out_ids.append(int(token))
        word = processor.decode([int(token)])
        if int(token) != processor.tokenizer.eos_token_id:
            print(word, end="")
    print("\noutput_ids:{}".format(out_ids))

# test_net_with_mask('../python_demo/test.jpg')

print(f'Convert vision transformer')
convert_vision_transformer()

print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    if i not in CROSS_ATTN_LAYERS:
        convert_block(i)
        convert_block_cache(i)
    else:
        convert_cross_block(i)
        convert_cross_block_cache(i)
 
print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
convert_lm_head()
convert_greedy_head()
convert_penalty_sample_head()
print("Done")
