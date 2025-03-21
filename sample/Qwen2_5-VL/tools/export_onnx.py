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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
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
        self.rotary_pos_emb_full = ViT.rotary_pos_emb(VISION_LENGTH) # max_grid_size << VISION_LENGTH
        self.cos = self.rotary_pos_emb_full.cos()
        self.sin = self.rotary_pos_emb_full.sin()

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states = ViT.patch_embed(hidden_states)
        self.cos = self.cos[position_ids].flatten(1).unsqueeze(1).repeat(1,1,2).unsqueeze(0)
        self.sin = self.sin[position_ids].flatten(1).unsqueeze(1).repeat(1,1,2).unsqueeze(0)

        # hidden_states = ViT.blocks[0](hidden_states, attention_mask=attention_mask, rotary_pos_emb=(self.cos, self.sin))
        for blk in ViT.blocks:
            hidden_states = blk(hidden_states, attention_mask=attention_mask, rotary_pos_emb=(self.cos, self.sin))
        hidden_states = ViT.merger(hidden_states)
        return hidden_states


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        return m_logits

class LmHeadWithTopK(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token

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
        (1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor(
        3*[[range(SEQ_LENGTH)]], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).to(dtype).to(device)
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
    position_ids = torch.tensor(3*[[range(1)]], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).to(dtype).to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)

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
    x = torch.randn(max_pixels, 1176).to(dtype=torch.float32, device=device)
    position_ids = torch.randn(max_pixels, 2).to(dtype=torch.int32, device=device)
    attention_mask = torch.zeros([1, 1, max_pixels, max_pixels], device=device, dtype=torch.float32)

    # export onnx
    model = VisionTransformer()
    torch.onnx.export(
        model, (x, position_ids, attention_mask),
        f'{folder}/vit/vision_transformer.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
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
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(dtype).to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')

def convert_lm_head_with_topk():
    model = LmHeadWithTopK()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(dtype).to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')

def convert_greedy_head():   
    model = GreedyHead()
    m_logits = torch.randn(1, VOCAB_SIZE).to(device)

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
    m_logits = torch.randn(1, VOCAB_SIZE).to(device)
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
        device_map="cpu"
    else:
        dtype = torch.bfloat16
        device_map="cuda:0"

    # load model
    model_path = args.model_path
    origin_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, trust_remote_code=True, attn_implementation='eager',
        torch_dtype=dtype, device_map=device_map
    ).eval()

    for param in origin_model.parameters():
        param.requires_grad = False
    return origin_model, device, dtype

def convert():
    # create folders to store onnx
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder + '/vit'):
        os.makedirs(folder + '/vit')

    # export models
    print(f'Convert block & block_cache')
    for i in tqdm(range(NUM_LAYERS)):
        convert_block(i)
        convert_block_cache(i)

    print(f'Convert embedding')
    convert_embedding()

    print(f'Convert lm_head')
    convert_lm_head_with_topk()
    # convert_lm_head()
    # convert_greedy_head()
    # convert_penalty_sample_head()

    print(f'Convert Vision Transformer')
    convert_vision_transformer()
    print("Done")


def get_image_messages(path, resized_height, resized_width):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": path,
                    "resized_height": resized_height,
                    "resized_width": resized_width,
                },
                {"type": "text", "text": "Describe this image and tell a story."},
            ],
        }
    ]
    return messages

def get_video_messages(path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]
    return messages

def vit_launch(pixel_values, grid_thw):
    # vit
    t, h, w  = grid_thw[0].tolist()

    vit_infer = VisionTransformer()
    cu_seqlens = [h * w * i for i in range(t+1)]
    attention_mask = torch.full(
        [1, 1, pixel_values.shape[0], pixel_values.shape[0]], -10000, device=device, dtype=torch.float32
    )
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

    pos_ids = []
    s_size = config.vision_config.spatial_merge_size
    hpos_ids = torch.Tensor([x for n in range(0, h, s_size) for _ in range(w // s_size) for x in (n, n, n+1, n+1)])
    wpos_ids = torch.Tensor([k for _ in range(h // s_size) for e in range(0, w, s_size) for k in [e, e+1, e, e+1]])
    pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)

    # prefill vit
    pixel_values_prefill = torch.zeros([max_pixels, 1176]).to(dtype=torch.float32, device=device)
    pixel_values_prefill[:pixel_values.shape[0],:] = pixel_values
    pos_ids_prefill = torch.zeros([max_pixels, 2]).to(dtype=torch.int32, device=device)
    pos_ids_prefill[:pos_ids.shape[0],:] = pos_ids
    attention_mask_vit_prefill = torch.ones([1, max_pixels, max_pixels], device=device, dtype=torch.float) * -10000
    attention_mask_vit_prefill[0,:pos_ids.shape[0],:pos_ids.shape[0]] = attention_mask

    vit_out = vit_infer(pixel_values_prefill, pos_ids_prefill, attention_mask_vit_prefill)
    return vit_out

def get_prefill_posid(grid_thw, vit_offset, token_length):
    text_len = vit_offset[0]
    valid_vit_length = len(vit_offset)

    grid_thw = grid_thw.tolist()
    llm_grid_t = grid_thw[0][0]
    llm_grid_h = grid_thw[0][1] // config.vision_config.spatial_merge_size
    llm_grid_w = grid_thw[0][2] // config.vision_config.spatial_merge_size
    t_position_ids = [i for i in range(text_len, llm_grid_t + text_len) for _ in range(llm_grid_h * llm_grid_w)]
    h_position_ids = [i+text_len for i in range(llm_grid_h) for _ in range(llm_grid_w)] * llm_grid_t
    w_position_ids = list(range(text_len, llm_grid_w+text_len)) * llm_grid_h * llm_grid_t

    st_idx = max(w_position_ids) + 1
    tail_text_len = token_length - valid_vit_length - text_len
    t_position_ids = list(range(text_len)) + t_position_ids + list(range(st_idx, st_idx+tail_text_len)) + [1] * (SEQ_LENGTH - token_length)
    h_position_ids = list(range(text_len)) + h_position_ids + list(range(st_idx, st_idx+tail_text_len)) + [1] * (SEQ_LENGTH - token_length)
    w_position_ids = list(range(text_len)) + w_position_ids + list(range(st_idx, st_idx+tail_text_len)) + [1] * (SEQ_LENGTH - token_length)
    position_ids = torch.Tensor([t_position_ids, h_position_ids, w_position_ids]).long().reshape(3, 1, -1)

    max_pos = st_idx + tail_text_len - 1
    return position_ids, max_pos

def test_net_with_mask(mode, messages):
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
    del image_inputs, video_inputs
    torch.cuda.empty_cache()
    input_ids = inputs.input_ids
    if mode == "image":
        pixel_values = inputs.pixel_values
        grid_thw = inputs.image_grid_thw
    elif mode == "video":
        pixel_values = inputs.pixel_values_videos
        grid_thw = inputs.video_grid_thw

    # make model
    embed = Embedding()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]
    greedy = GreedyHead()

    # vit
    vit_embeds = vit_launch(pixel_values, grid_thw)
    del pixel_values
    torch.cuda.empty_cache()
    # embedding
    input_ids_prefill = torch.zeros(1, SEQ_LENGTH).to(torch.int32).to(device)
    input_ids_prefill[:, :input_ids.shape[-1]] = input_ids
    inputs_embeds = embed(input_ids_prefill)
    inputs_embeds = inputs_embeds.view(1, SEQ_LENGTH, HIDDEN_SIZE)

    if mode == "image":
        vit_offset = torch.where(input_ids==config.image_token_id)[1].tolist()
    elif mode == "video":
        vit_offset = torch.where(input_ids==config.video_token_id)[1].tolist()

    valid_vit_length = len(vit_offset)
    inputs_embeds[:,vit_offset[0]:vit_offset[0]+valid_vit_length, :] = vit_embeds[:valid_vit_length]

    ID_IM_END = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ID_END = tokenizer.convert_tokens_to_ids("<|end|>")

    position_ids, max_pos = get_prefill_posid(grid_thw, vit_offset, len(input_ids[0]))
    
    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float() * -10000.0

    for i in range(input_ids.shape[-1]):
        for j in range(input_ids.shape[-1]):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(1, 1, SEQ_LENGTH, SEQ_LENGTH).to(device)

    k_cache = []
    v_cache = []
    for i in tqdm(range(NUM_LAYERS)):
        inputs_embeds, k, v = blocks[i](inputs_embeds.to(dtype), position_ids,
                              attention_mask.to(dtype))
        k[:, input_ids.shape[-1]:, :, :] = 0
        v[:, input_ids.shape[-1]:, :, :] = 0
        k_cache.append(k)
        v_cache.append(v)
        del k, v
        torch.cuda.empty_cache()
    inputs_embeds = inputs_embeds[:, input_ids.shape[-1] - 1:input_ids.shape[-1]].view(1, 1, HIDDEN_SIZE)
    lm = LmHead()

    token = greedy(lm(inputs_embeds.to(dtype))).view(1)
    out_ids = [int(token)]
    token_len = input_ids.shape[-1]
    valid_position_ids = max_pos

    while int(token) not in [ID_IM_END, ID_END] and token_len < SEQ_LENGTH:
        token_len += 1
        input_id = torch.tensor([token]).to(device)
        out = embed(input_id).view(1, 1, HIDDEN_SIZE)
        del input_id
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
            del k, v
            torch.cuda.empty_cache()
        token = greedy(lm(out.to(dtype))).view(1)
        out_ids.append(int(token))
    words = tokenizer.decode(out_ids)
    print(words)
    # output_text = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # print(output_text)
    # print("\noutput_ids:{}".format(out_ids))

def test_model_generate(messages):
    print("-------------model generate-------------")
    # Preparation for inference
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

    # Inference: Generation of the output
    generated_ids = origin_model.generate(**inputs, max_new_tokens=128, do_sample=False, num_beams=1)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

def test_video(path):
    messages = get_video_messages(path)
    test_net_with_mask("video", messages)

def test_image(path, resized_height, resized_width):
    messages = get_image_messages(path, resized_height, resized_width)
    test_net_with_mask("image", messages)

    # test_model_generate(messages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export onnx')
    parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
    parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-s', '--seq_length', type=int, default=2048, help="sequence length")
    parser.add_argument('-i', '--vision_length', type=int,default=600, help="vision_length = max_image_width // patch_size * max_image_height // patch_size")
    parser.add_argument('-n', '--num_threads', type=int, default=32, help='The number of threads used for torch if device is cpu')
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
    BATCH_SIZE = args.batch_size
    NUM_LAYERS = config.num_hidden_layers
    HIDDEN_SIZE = config.hidden_size
    NUM_ATTENTION_HEADS = config.num_attention_heads
    NUM_KEY_VALUE_HEADS = config.num_key_value_heads
    HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
    VOCAB_SIZE = config.vocab_size
    print(f"Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n")
    folder = f"./tmp/onnx"

    VISION_LENGTH = args.vision_length
    max_pixels = VISION_LENGTH * 4
    if SEQ_LENGTH < VISION_LENGTH:
        raise ValueError("SEQ_LENGTH < VISION_LENGTH")
 
    print("\033[31m建议导出模型之前，根据需要运行test_image与test_video\033[0m")
    print("\033[31m特别是修改seq_length与vision_length后，需要用真实图片或真实视频运行一遍以检查可行性 \033[0m")
    print("\033[31m如果输入为图片时，注意resized_height与resized_width，避免resize导致图片质量损失 \033[0m")


    # test_image(path = "./../python_demo/test.jpg", resized_height=280, resized_width=420)
    # test_video(path = "./sample.mp4")

    # convert
    convert()
