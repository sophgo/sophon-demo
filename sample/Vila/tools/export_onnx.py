#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from builder import load_pretrained_model
import torch
import os
import argparse

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
parser.add_argument('-s', '--seq_length', type=int, default=4096, help="sequence length")


args = parser.parse_args()

model_path = args.model_path
model_name = "vila1.5-3b"
tokenizer, oringin_model, image_processor, context_len = load_pretrained_model(model_path, model_name, None)
device = torch.device("cuda")
oringin_model.eval().to(device)
llama = oringin_model.llm.model
NUM_FRAMES=1
LAYERS=32
HIDDEN_SIZE = 2560
SEQ_LENGTH = args.seq_length
NUM_KEY_VALUE_HEADS = 20
HEAD_DIM = HIDDEN_SIZE // NUM_KEY_VALUE_HEADS
folder='models/onnx'

path = folder
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Directory '{path}' was created.")
else:
    print(f"Directory '{path}' already exists.")

class Vision_Embedding(torch.nn.Module):
    def __init__(self, vision_tower, mm_projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector
    def forward(self, images):
        image_forward_out = self.vision_tower(
            images.to(device),
            output_hidden_states=True,
        )
        image_feature = image_forward_out.hidden_states[-2]
        return self.mm_projector(image_feature)
    



class Embedding(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model.embed_tokens(input_ids)


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = llama.layers[layer_id]

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
        self.layer_id = layer_id
        self.layer = llama.layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v

class LmHeadWithTopK(torch.nn.Module):

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, hidden_states):
        hidden_states = self.llm.model.norm(hidden_states)
        m_logits = self.llm.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token
        
def convert_vision_embedding():
    model = Vision_Embedding(oringin_model.vision_tower.vision_tower, oringin_model.mm_projector)
    images = torch.randn(NUM_FRAMES, 3, 384, 384, dtype=torch.float16).to(device=device)
    torch.onnx.export(
        model, images,
        f"{folder}/vision_embedding.onnx",
        input_names=["images"],
        output_names=["hidden_states"],
        do_constant_folding=True,
        opset_version=15
    )

def convert_embedding():
    model = Embedding(llama)
    input_ids = torch.tensor([range(SEQ_LENGTH)]).to(device=device)

    torch.onnx.export(model, (input_ids),
                      f'{folder}/embedding.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE), dtype=torch.float16).to(device=device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device=device)
    attention_mask = -1000 * torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype=torch.float32).triu(diagonal=1).to(device=device)

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
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE), dtype=torch.float16).to(device=device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device=device)
    attention_mask = -1000 * torch.ones((1, 1, 1, SEQ_LENGTH + 1), dtype=torch.float32).triu(diagonal=1).to(device=device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM), dtype=torch.float16).to(device=device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM), dtype=torch.float16).to(device=device)

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
    
def convert_lm_head_with_topk():
    model = LmHeadWithTopK(oringin_model.llm)
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.float16).to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head_with_topk.pt')

def convert_lm_head():
    model = LmHeadWithTopK(oringin_model.llm)
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.float16).to(device)
    torch.onnx.export(model, (hidden_states),
                    f'{folder}/lm_head.onnx',
                    verbose=False,
                    input_names=['hidden_states'],
                    output_names=['topk'],
                    do_constant_folding=True,
                    opset_version=15)

convert_vision_embedding()
convert_embedding()
convert_lm_head()

for i in range(LAYERS):
    convert_block(i)
    convert_block_cache(i)