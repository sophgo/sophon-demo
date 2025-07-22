import sys
import time
import numpy as np
import torch
import argparse

from typing import Tuple, Dict, Optional

from vita.model.vita_tts.encoder.transformer import Transformer
from vita.model.vita_tts.encoder.subsampling import Subsampling

def add_encoder_args(group):
    """Add Encoder common arguments."""
    group.add_argument(
        "--encoder-layer-config",
        type=str,
        default="tdnn-dtc",
        help="Layer config of encoder. Format layername-layername-...",
    )
    group.add_argument(
        "--encoder-input-dim",
        type=int,
        default=256,
        help="Input dim of encoder. Must equal to the input dim of the first Component (default=40)"
    )
    group.add_argument(
        "--encoder-output-dim",
        type=int,
        default=256,
        help="Output dim of encoder. Must enqual to the output dim of the last Component ! (default=256)"
    )
    group = Transformer.add_arguments(group)
    group = Subsampling.add_arguments(group)
    return group

def assign_args_from_dict(args, dict, prefix_key=None):
    if prefix_key is not None:
        dict = dict[prefix_key]
    for k, v in dict.items():
        k_args = k.replace('-', '_') 
        if hasattr(args, k_args):
            setattr(args, k_args, dict[k])
    return args

class speechEncoder(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            overview_conf = None,
            para_conf = None,
            global_cmvn = None):
        super(speechEncoder, self).__init__()

        parser = argparse.ArgumentParser()
        add_encoder_args(parser)
        args, _ = parser.parse_known_args()
        assign_args_from_dict(args, overview_conf)

        self.config = args.encoder_layer_config.split('-')
        encoder_input_dim = args.encoder_input_dim
        encoder_output_dim = args.encoder_output_dim
        prev_output_dim = encoder_input_dim
        prev_component_name = "encoder"

        self.global_cmvn = global_cmvn
        self.enc = torch.nn.ModuleList([])
        for name in self.config:
            assign_args_from_dict(args, para_conf[name])
            if len(name.split('_'))  == 2:
                name = name.split('_')[0]
            elif len(name.split('_'))  == 1:
                name = name
            else:
                print("WRONG CONFIG! {} is not valid".format("encoder", name))
                sys.exit()
            if name == "transformer":
                self.enc.append(Transformer(args))
            elif name == "subsampling":
                self.enc.append(Subsampling(args))
            else:
                print("{} is not supported now!".format(name))
                return NotImplemented                
            component_input_dim = getattr(args, name + "_input_dim")
            if component_input_dim != prev_output_dim:
                print("WRONG CONFIG! --{}-output-dim ({}) does not equal to --{}-input-dim ({})"
                        .format(prev_component_name, prev_output_dim, name, component_input_dim))
                sys.exit()
            prev_output_dim = getattr(args, name + "_output_dim")
            prev_component_name = name
        
        if (prev_output_dim != encoder_output_dim):
            print("WRONG CONFIG! --{}-output-dim ({}) does not equal to --{}-output-dim ({}, the last component)"
                        .format("encoder", encoder_output_dim, name, prev_output_dim))
            sys.exit()
        
        self._output_size = encoder_output_dim

        num_params = sum(p.numel() for p in self.parameters())
        print('the number of speech encoder params: {}M'.format(num_params/1024/1024))

    def output_size(self) -> int:
        return self._output_size
    
    def forward(self, xs, ilens, decoding_chunk_size=None, num_decoding_left_chunks=None):
        """
        Forward pass through the encoder.

        Parameters:
        - xs: torch.Tensor, shape (batch_size, sequence_length, input_dim)
            The input tensor containing the sequence of input vectors.
            - batch_size: The number of sequences in the batch.
            - sequence_length: The length of each sequence.
            - input_dim: The dimensionality of each input vector.

        - ilens: torch.Tensor, shape (batch_size,)
            The lengths of each sequence in the batch, used for padding masks.

        - decoding_chunk_size: int, optional (default=None)
            The size of chunks to use for decoding

        - num_decoding_left_chunks: int, optional (default=None)
            The number of left chunks to use for decoding

        Returns:
        - xs: torch.Tensor, shape (batch_size, sequence_length, encoded_dim)
            The encoded output tensor, where encoded_dim is the dimensionality of the encoded representation.

        - masks: torch.Tensor, shape (batch_size, 1, sequence_length)
            The padding mask tensor, where True indicates valid elements and False indicates padded elements.
        """
        if decoding_chunk_size is not None and num_decoding_left_chunks is not None:
            for layer in self.enc:
                if hasattr(layer, "chunk_size"):
                    layer.chunk_size = decoding_chunk_size
                if hasattr(layer, "left_chunks"):
                    layer.left_chunks = num_decoding_left_chunks
                if hasattr(layer, "transformer_dynamic_chunks"):
                    layer.transformer_dynamic_chunks = False

        assert(len(xs.shape)) == 3
        T = xs.size(1)
        masks = ~make_pad_mask(ilens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        for module in self.enc:
            xs, ilens, masks = module(xs, ilens, masks)
        return xs, masks

    def infer(self, xs_pad, buffer, buffer_index, buffer_out, pe_index):
        if self.global_cmvn is not None:
            xs_pad = self.global_cmvn(xs_pad)
        for module in self.enc:
            xs_pad, buffer, buffer_index, buffer_out, pe_index = module.infer(xs_pad, 
                                            buffer, buffer_index, buffer_out, pe_index)
        return xs_pad, buffer, buffer_index, buffer_out, pe_index
