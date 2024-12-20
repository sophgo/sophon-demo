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
from tqdm import tqdm
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
torch.set_grad_enabled(False)


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.embed_tokens(input_ids)
        return out.float()


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        # SEQ_LENGTH = 175
        value_states = torch.zeros(
            (1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM), dtype=dtype).to(device)
        position_ids = torch.tensor(3*[[range(SEQ_LENGTH)]], dtype=torch.long).to(device)
        self.rotary_emb = self.layer.self_attn.rotary_emb
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)
        self.cos = self.cos.transpose(1,2)
        self.sin = self.sin.transpose(1,2)

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=(self.cos, self.sin),
            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        value_states = torch.randn(
            (1, 1, NUM_KEY_VALUE_HEADS, HEAD_DIM), dtype=dtype).to(device)
        position_ids = torch.tensor(3*[[range(SEQ_LENGTH)]], dtype=torch.long).to(device)
        self.rotary_emb = self.layer.self_attn.rotary_emb
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)
        self.cos = self.cos.transpose(1,2)
        self.sin = self.sin.transpose(1,2)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=(past_k, past_v),
            position_ids=position_ids,
            position_embeddings=(self.cos, self.sin),
            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()

class VisionTransformer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.rotary_pos_emb_full = ViT.rotary_pos_emb(100)

    def forward(self, hidden_states, pos_ids, attention_mask):
        hidden_states = ViT.patch_embed(hidden_states)
        # hidden_states = ViT.patch_embed(hidden_states[:pixel_length[0],:])
        # rotary_pos_emb = ViT.rot_pos_emb(image_grid_thw)
        # max_grid_size = image_grid_thw[:, 1:].max()
        rotary_pos_emb = self.rotary_pos_emb_full[pos_ids].flatten(1)
        
        # cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
        #     dim=0, dtype=torch.int32
        # )
        # cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in ViT.blocks:
            hidden_states = blk(hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)

        return ViT.merger(hidden_states), 


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
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
    

def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).float().to(device)
    position_ids = torch.tensor(
        3*[[range(SEQ_LENGTH)]], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).float().to(device)
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
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).float().to(device)
    position_ids = torch.tensor(3*[[range(1)]], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).float().to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).float().to(device)

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

def convert_vision_transformer():
    # make input
    x = torch.randn(VISION_SEQ_LENGTH, 1176).to(dtype=torch.float32, device=device)
    # thw = torch.randn(3).to(dtype=torch.int32, device=device)
    thw = torch.tensor([[2, 20, 36]])
    # length = torch.tensor([1440], dtype=torch.int32, device=device)
    pos_ids = torch.randn(x.shape[0], 2).to(dtype=torch.int32, device=device)
    # cu_seqlens = torch.tensor([[0, 720, 1440]])
    attention_mask = torch.zeros([1, x.shape[0], x.shape[0]], device=device, dtype=torch.float32)

    # # trace
    model = VisionTransformer()
    torch.onnx.export(
        model, (x, pos_ids, attention_mask),
        f'{vit_folder}/vision_transformer.onnx',
        verbose=False,
        input_names=['input_states', 'pos_ids', 'attention_mask'],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=15
    )

def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.int32).to(device)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')


def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).float().to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')


def convert_greedy_head():   
    model = GreedyHead()
    m_logits = torch.randn(1, VOCAB_SIZE)

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
    m_logits = torch.randn(1, VOCAB_SIZE)
    input_ids = torch.tensor([range(SEQ_LENGTH)])
    top_p = torch.tensor([0.8])
    temperature = torch.tensor([0.98])
    penalty = torch.tensor([0.98])

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


def setup_environment():
    import numpy as np
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return

def load_model():
    # setup environment
    setup_environment()

    # set
    device = torch.device(args.device)
    if args.device == "cpu":
        dtype = torch.float
        torch.set_num_threads(args.num_threads)
    else:
        # dtype = torch.float16
        dtype = torch.bfloat16

    # load model
    model_path = args.model_path
    origin_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, trust_remote_code=True, attn_implementation='eager',
        torch_dtype=dtype, device_map="cpu"
    ).eval()

    for param in origin_model.parameters():
        param.requires_grad = False
    return origin_model, device, dtype

def convert():
    # export models
    print(f'Convert block & block_cache')
    for i in tqdm(range(NUM_LAYERS)):
    # for i in tqdm(range(1)):
        convert_block(i)
        convert_block_cache(i)

    print(f'Convert embedding')
    convert_embedding()

    print(f'Convert lm_head')
    convert_lm_head()
    convert_greedy_head()
    convert_penalty_sample_head()

    print(f'Convert Vision Transformer')
    convert_vision_transformer()
    print("Done")


def test_net_with_mask():
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": "./../python_demo/image1.jpg",
    #             },
    #             {"type": "text", "text": "Describe this image and tell a story."},
    #         ],
    #     }
    # ]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "carvana_video.mp4",
                    # "max_pixels": 360 * 420,
                    # "fps": 1.0,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = inputs.input_ids
    pixel_values = inputs.pixel_values_videos
    grid_thw = inputs.video_grid_thw

    # make model:
    embed = Embedding()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]
    greedy = GreedyHead()

    # prefill
    input_ids_prefill = torch.zeros(1, SEQ_LENGTH).to(torch.int32)
    input_ids_prefill[:, :input_ids.shape[-1]] = input_ids
    attention_mask_prefill = torch.zeros(1, SEQ_LENGTH)
    attention_mask_prefill[:, :input_ids.shape[-1]] = inputs.attention_mask

    vit_infer = VisionTransformer()
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    attention_mask_vit = torch.full(
        [1, pixel_values.shape[0], pixel_values.shape[0]], torch.finfo(torch.float32).min, device=device, dtype=torch.float32
    )
    for i in range(1, len(cu_seqlens)):
        attention_mask_vit[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
    # attention_mask_vit = torch.zeros([1, pixel_values.shape[0], pixel_values.shape[0]], device=device, dtype=torch.bool)
    # for i in range(1, len(cu_seqlens)):
    #     attention_mask_vit[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
    # length = torch.tensor([1], dtype=torch.int32, device=device)
    # length[0] = pixel_values.shape[-2]
    pos_ids = []
    for t, h, w in grid_thw:
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // config.vision_config.spatial_merge_size,
            config.vision_config.spatial_merge_size,
            w // config.vision_config.spatial_merge_size,
            config.vision_config.spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // config.vision_config.spatial_merge_size,
            config.vision_config.spatial_merge_size,
            w // config.vision_config.spatial_merge_size,
            config.vision_config.spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)

    # prefill vit
    pixel_values_prefill = torch.zeros([VISION_SEQ_LENGTH, 1176]).to(dtype=torch.float32, device=device)
    pixel_values_prefill[:pixel_values.shape[0],:] = pixel_values
    pos_ids_prefill = torch.zeros([VISION_SEQ_LENGTH, 2]).to(dtype=torch.int32, device=device)
    pos_ids_prefill[:pos_ids.shape[0],:] = pos_ids
    attention_mask_vit_prefill = torch.zeros([1, VISION_SEQ_LENGTH, VISION_SEQ_LENGTH], device=device, dtype=torch.bool)
    attention_mask_vit_prefill[0,:pos_ids.shape[0],:pos_ids.shape[0]] = attention_mask_vit

    image_embeds = vit_infer(pixel_values_prefill, pos_ids_prefill, attention_mask_vit_prefill)  # [150, 1536]
    # image_embeds = vit_infer(pixel_values, pos_ids, attention_mask_vit)  # [150, 1536]
    inputs_embeds = torch.zeros((1, SEQ_LENGTH, HIDDEN_SIZE)).to(device)

    inputs_embeds = embed(input_ids_prefill)
    inputs_embeds = inputs_embeds.view(1, SEQ_LENGTH, HIDDEN_SIZE)
    image_mask = (input_ids_prefill == config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    image_embeds = image_embeds[0].to(device, dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    ID_IM_END = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ID_END = tokenizer.convert_tokens_to_ids("<|end|>")
    
    position_ids, rope_deltas = Qwen2VLForConditionalGeneration(config).get_rope_index(
        input_ids_prefill, None, grid_thw, attention_mask_prefill
    )

    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float() * -10000.0

    for i in range(input_ids.shape[-1]):
        for j in range(input_ids.shape[-1]):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(
        1, 1, SEQ_LENGTH, SEQ_LENGTH).to(device)


    k_cache = []
    v_cache = []
    for i in tqdm(range(NUM_LAYERS)):
        inputs_embeds, k, v = blocks[i](inputs_embeds.to(dtype), position_ids,
                              attention_mask.to(dtype))
        k[:, input_ids.shape[-1]:, :, :] = 0
        v[:, input_ids.shape[-1]:, :, :] = 0
        k_cache.append(k)
        v_cache.append(v)
    inputs_embeds = inputs_embeds[:, input_ids.shape[-1] - 1:input_ids.shape[-1]].view(1, 1, HIDDEN_SIZE)
    lm = LmHead()

    token = greedy(lm(inputs_embeds.to(dtype))).view(1)
    out_ids = [int(token)]
    token_len = input_ids.shape[-1]
    valid_position_ids = position_ids.numpy().max()

    while int(token) not in [ID_IM_END, ID_END] and token_len < SEQ_LENGTH:
        token_len += 1
        input_id = torch.tensor([token]).to(device)
        out = embed(input_id).view(1, 1, HIDDEN_SIZE)
        valid_position_ids += 1
        position_ids = torch.tensor(3*[[[valid_position_ids]]]).to(device)

        attention_mask = torch.zeros(
            (1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
        attention_mask[:, :, :, token_len-1:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out.to(dtype), position_ids,
                                     attention_mask.to(dtype),
                                     k_cache[i].to(dtype), v_cache[i].to(dtype))
            k_cache[i][:, token_len-1:token_len, :, :] = k[:, :, :, :]
            v_cache[i][:, token_len-1:token_len, :, :] = v[:, :, :, :]
        token = greedy(lm(out.to(dtype))).view(1)
        out_ids.append(int(token))
    words = tokenizer.decode(out_ids)
    print(words)
    output_text = processor.batch_decode(
        out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("\noutput_ids:{}".format(out_ids))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export onnx')
    parser.add_argument('-m', '--model_path', type=str, default="Qwen/Qwen2-VL-7B-Instruct", help='path to the torch model')
    parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
    parser.add_argument('-vs', '--vision_seq_length', type=int, default=1024, help="vision input max sequence length")
    parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
    # TODO add these params
    # parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    # parser.add_argument('--max_pos_len', type=int, default=8704, help="max position length")
    # parser.add_argument('--generation_mode', type=str, default="default", choices=["default", "lmhead_with_topk"], help="generation mode")
    args = parser.parse_args()

    # processor & tokenizer
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # load model
    origin_model, device, dtype = load_model()
    config = origin_model.config
    transformer = origin_model.model
    ViT = origin_model.visual
    layers = transformer.layers

    SEQ_LENGTH = args.seq_length
    VISION_SEQ_LENGTH = args.vision_seq_length
    BATCH_SIZE = 1 # args.batch_size
    NUM_LAYERS = config.num_hidden_layers
    HIDDEN_SIZE = config.hidden_size
    NUM_ATTENTION_HEADS = config.num_attention_heads
    NUM_KEY_VALUE_HEADS = config.num_key_value_heads
    HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
    VOCAB_SIZE = config.vocab_size
    print(f"Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n")
    print("\033[31m修改了load model方式，将attn_implementation由sdpa改为了eager，不然无法导出onnx\033[0m")
    
    # create folders to save onnx or pt
    execution_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(execution_dir, "../models/onnx/llm") # folder for LLM
    if not os.path.exists(folder):
        os.makedirs(folder)

    vit_folder = os.path.join(execution_dir, "../models/onnx/vit") # folder for VIT
    if not os.path.exists(vit_folder):
        os.makedirs(vit_folder)

    # test pt
    # test_net_with_mask()

    # convert
    convert()

