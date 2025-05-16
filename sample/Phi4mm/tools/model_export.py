#!/usr/bin/env python3
import os
import warnings
import logging
import argparse

warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("onnx").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoConfig,
)
from onnx_rebuilder import *

GREEN_COLOR = "\033[92m"  # ANSI escape code for green text
RED_COLOR = "\033[91m"
RESET_COLOR = "\033[0m"

def merge_lora_linear(lora_linear, lora_key='vision'):
    if not hasattr(lora_linear, 'base_layer'):
        return
    W0 = lora_linear.base_layer.weight.data
    if lora_key not in lora_linear.lora_A or lora_key not in lora_linear.lora_B:
        print(f"Warning: {lora_key} branch not found in {lora_linear}")
        return
    WA = lora_linear.lora_A[lora_key].weight.data
    WB = lora_linear.lora_B[lora_key].weight.data

    # 获取 alpha（支持 dict 或 float）
    alpha = 1.0
    if hasattr(lora_linear, 'lora_alpha'):
        if isinstance(lora_linear.lora_alpha, dict):
            alpha = lora_linear.lora_alpha.get(lora_key, 1.0)
        elif isinstance(lora_linear.lora_alpha, (torch.nn.ParameterDict, torch.nn.ModuleDict)):
            alpha = getattr(lora_linear.lora_alpha, lora_key, 1.0)
        else:
            alpha = lora_linear.lora_alpha
    r = WA.shape[0]
    scaling = alpha / r

    W_delta = torch.matmul(WB, WA) * scaling
    W0.add_(W_delta)

    in_features = lora_linear.base_layer.in_features
    out_features = lora_linear.base_layer.out_features
    bias = lora_linear.base_layer.bias is not None
    new_linear = torch.nn.Linear(in_features, out_features, bias=bias).to(W0.dtype)
    new_linear.weight.data.copy_(lora_linear.base_layer.weight.data)
    if bias:
        new_linear.bias.data.copy_(lora_linear.base_layer.bias.data)
    return new_linear

def recursive_merge_lora(module, lora_key='vision'):
    for name, child in module.named_children():
        # 如果是 lora.Linear，则合并
        if child.__class__.__name__ == 'Linear' and hasattr(child, 'base_layer') and hasattr(child, 'lora_A'):
            new_linear = merge_lora_linear(child, lora_key)
            if new_linear is not None:
                setattr(module, name, new_linear)
        else:
            recursive_merge_lora(child, lora_key)

class ModelExporter:
    def __init__(self, args):
        super().__init__()
        self.model_path = args.model_path
        self.seq_length = args.seq_length
        self.embedding_disk = args.embedding_disk
        self.lmhead_with_topk = 0
        self.out_dir = args.out_dir
        self.onnx_dir = os.path.join(self.out_dir, f"onnx_seq{self.seq_length}")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.onnx_dir, exist_ok=True)

        # for vision language model
        self.visual = None
        self.visual_model = None

        # load original weight, save config and tokenizer
        self.load_pretrained()

        # rebuild original weight to onnx model
        self.onnx_rebuilder = OnnxRebuilder(self.onnx_dir,
                                            self.model_path,
                                            self.seq_length,
                                            self.model_type,
                                            self.embedding_disk,
                                            self.lmhead_with_topk,
                                            self.config,)
        self.rebuild_model()

    def load_pretrained(self):
        self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.model_type = self.config.model_type

        if 'qwen2_vl' == self.model_type:
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_path)
        elif 'qwen2_5_vl' == self.model_type:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path)
        elif 'mllama' == self.model_type:
            from transformers import MllamaForConditionalGeneration
            self.model = MllamaForConditionalGeneration.from_pretrained(self.model_path)
        elif 'llama' == self.model_type:
            from transformers import LlamaForCausalLM
            self.model = LlamaForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
        else:
            if "ForCausalLM" in self.config.architectures[0]:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path, trust_remote_code=True, low_cpu_mem_usage=True,
                            torch_dtype=torch.float, device_map='cpu')
                    # recursive_merge_lora(self.model.model.layers, lora_key='vision')
                except:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path, trust_remote_code=True)
            elif "Model" in self.config.architectures[0]:
                self.model = AutoModel.from_pretrained(
                    self.model_path, trust_remote_code=True)
            else:
                raise ValueError(f"Unsupported Architectures:[ {self.config.architectures[0]} ]")

        self.model = self.model.cpu().eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def rebuild_model(self):
        self.onnx_rebuilder.model_map = self.onnx_rebuilder.model_mapper.get_map(self.model.config)

        # load config
        ModelMapper.do_map(self.onnx_rebuilder, self.model.config, self.onnx_rebuilder.model_map['config'])
        # rebuild config
        self.onnx_rebuilder.rebuild_config()

        # load modules
        ModelMapper.do_map(self.onnx_rebuilder, self.model, self.onnx_rebuilder.model_map['model'])
        # rebuild modules
        self.onnx_rebuilder.rebuild_modules()

    def export(self):
        with torch.no_grad():
            # self.onnx_rebuilder.test_net_with_mask_new()
            # return
            self.onnx_rebuilder.export_config()
            self.onnx_rebuilder.export_embed()
            self.onnx_rebuilder.export_lm_head()
            if not self.lmhead_with_topk:
                self.onnx_rebuilder.export_greedy_head()
                self.onnx_rebuilder.export_penalty_sample_head()
            self.onnx_rebuilder.export_block()
            self.onnx_rebuilder.export_block_cache()
            self.onnx_rebuilder.export_visual()
            self.onnx_rebuilder.export_speech()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='original weight, like ./Qwen2-7B-Instruct')
    parser.add_argument('-s', '--seq_length', type=int, required=True,
                        help="sequence length")
    parser.add_argument('--embedding_disk', action='store_true',
                        help='export embedding as bin file and inference by cpu')
    parser.add_argument('--out_dir', type=str, default='./tmp',
                        help='output onnx/bmodel path, default `./tmp`')
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model_exporter = ModelExporter(args)
    with torch.no_grad():
        model_exporter.export()
