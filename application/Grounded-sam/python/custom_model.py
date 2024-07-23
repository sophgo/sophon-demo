#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sys
import os 
script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录
script_dir = os.path.dirname(script_path)

sys.path.append(os.path.join(script_dir,"sam"))
sys.path.append(os.path.join(script_dir,"groundingdino"))

import numpy as np
from transformers import BertTokenizerFast
from groundingdino_pil import GroundingDINO
from sam_encoder import SamEncoder
from sam_opencv import SAM_b

class CustomSamEncoder(SamEncoder):
    def __init__(self, embedding_bmodel, dev_id, img_size=1024):
        super().__init__(embedding_bmodel, dev_id)


# 包装函数
def create_grounding_dino(bmodel_path, text_prompt, tokenizer_path, text_threshold, box_threshold, dev_id, output_dir=None, token_spans=None):
    class Args:
        pass

    args = Args()
    args.bmodel = bmodel_path
    args.text_prompt = text_prompt
    args.tokenizer_path = tokenizer_path
    args.token_spans = token_spans
    args.output_dir = output_dir
    args.text_threshold = text_threshold
    args.box_threshold = box_threshold
    args.dev_id = dev_id

    return GroundingDINO(args)

def create_SAM_b(input_point, decode_bmodel, dev_id):
    class Args:
        pass
    
    args = Args()
    args.auto = 0
    args.decode_bmodel = decode_bmodel
    args.dev_id = dev_id
    args.input_point = input_point

    return SAM_b(args)

def create_SamEncoder(embedding_bmodel, decode_bmodel, dev_id):
    class Args:
        pass
    
    args = Args()
    args.decode_bmodel = decode_bmodel
    args.dev_id = dev_id
    args.embedding_bmodel = embedding_bmodel

    return SamEncoder(args)