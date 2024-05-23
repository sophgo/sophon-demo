#===----------------------------------------------------------------------===#
#
# Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import torch
import argparse
import sys
sys.path.append("../")
from tools.export_whisper.model import ModelDimensions
from tools.export_whisper.utils import (
    optional_int,
)

class Gather(torch.nn.Module):
    def __init__(self):
        super(Gather, self).__init__()

    def forward(self, input, index):
        return input[index]

def export_gather():
    from tools.export_whisper import available_models

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    args = parser.parse_args().__dict__
    name = args["model"]
    if name == "base":
        dims = ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=512,
            n_audio_head=8,
            n_audio_layer=6,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=512,
            n_text_head=8,
            n_text_layer=6
        )
    elif name == "large-v2":
        dims = ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32
        )
    elif name == "tiny":
        dims = ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4
        )
    elif name == "medium":
        dims = ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=1024,
            n_audio_head=16,
            n_audio_layer=24,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=1024,
            n_text_head=16,
            n_text_layer=24
        )
    elif name == "small":
        dims = ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=768,
            n_audio_head=12,
            n_audio_layer=12,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=768,
            n_text_head=12,
            n_text_layer=12
        )
    gather = Gather()
    input = torch.randn(args["beam_size"], dims.n_text_ctx, dims.n_text_state)
    index = torch.tensor(range(args["beam_size"]))
    torch.onnx.export(
        gather,
        (input, index),
        f"kvcache_rearrange_{name}_5beam_448pad.onnx",
        verbose=True,
        opset_version=15
    )

if __name__ == "__main__":
    export_gather()