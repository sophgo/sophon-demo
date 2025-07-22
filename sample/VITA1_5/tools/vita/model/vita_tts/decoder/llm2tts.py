import os
import sys
import copy
import json
import torch
import random
import argparse
import subprocess
import numpy as np
import soundfile as sf
import subprocess
import concurrent.futures

from vita.model.vita_tts.decoder.decoder import LLM2TTSCodecAR
from vita.model.vita_tts.decoder.ticodec.vqvae_tester import VqvaeTester

class llm2TTS():
    def __init__(self, model_path):
        # self.model = self.get_model(model_path).cuda().to(
        #                             torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        #                             )
        self.model = self.get_model(model_path).to('cpu').to(torch.float32)
        self.model.rebuild_modules()
        self.infer = self.model.infer

        self.codec_model = VqvaeTester(config_path=model_path + "/codec/model.json", 
                                        model_path=model_path + "/codec/final.pt",
                                        sample_rate=24000)
        self.codec_model = self.codec_model.cpu()
        self.codec_model.vqvae.generator.remove_weight_norm()
        self.codec_model.vqvae.encoder.remove_weight_norm()
        self.codec_model.eval()
    
    def get_model_conf(self, model_path):
        model_conf = model_path + "/decoder/model.json"
        with open(model_conf, "rb") as f:
            print('reading a config file from ' + model_conf)
            confs = json.load(f)
        # for asr, tts, mt
        idim, odim, args = confs
        return argparse.Namespace(**args)

    def get_model(self, model_path):
        args_load = self.get_model_conf(model_path)
        args_load = vars(args_load)
        args = argparse.Namespace(**args_load)
        odim = args.odim
        idim = args.idim
        model = LLM2TTSCodecAR(idim, odim, args)

        # Resume from a snapshot
        snapshot_dict = torch.load(model_path + "/decoder/final.pt", map_location=lambda storage, loc: storage)
        if 'model' in snapshot_dict.keys():
            resume_model_dict = snapshot_dict['model']
        else:
            resume_model_dict = snapshot_dict
        
        model_dict = model.state_dict()
        for key in model_dict.keys():
            if key in resume_model_dict.keys():
                if model_dict[key].shape == resume_model_dict[key].shape:
                    model_dict[key] = resume_model_dict[key]
                else:
                    print('Key {} has different shape, {} VS {}'.format(key, model_dict[key].shape, 
                                                                        resume_model_dict[key].shape))
            else:
                print('Key {} has not in resume model'.format(key))
        model.load_state_dict(model_dict)
        model.eval()
        return model
    
    def find_min_sum_index(self, buffer, syn, N, threshold):
        """
        Find the index with the minimum sum of a sliding window in the given audio segment 
        and perform operations based on this index.

        Parameters:
        - buffer (torch.Tensor): The buffer containing previously processed audio segments.
        - syn (torch.Tensor): The current audio segment to be processed.
        - N (int): The size of the sliding window used to calculate the sum.
        - threshold (float): Threshold value to determine whether to concatenate buffer and current segment or not.

        Returns:
        - tuple: A tuple containing the updated buffer and the processed audio segment.

        """
        arr = syn[0, 0, :]
        L = len(arr)
        mid = L // 2
        
        kernel = torch.ones(N).to(arr.device)
        window_sums = torch.nn.functional.conv1d(arr.abs().view(1, 1, -1), kernel.view(1, 1, -1), padding=0).squeeze()
        
        start_index = mid - (N // 2)
        min_sum, min_index = torch.min(window_sums[start_index:], dim=0)

        # get the start and end index of the window
        start_index = max(0, min_index.item() + start_index)
        end_index = min(L, min_index.item() + N + start_index)
        
        # calculate the real min_sum and min_index
        min_sum_real, min_index_real = torch.min(arr[start_index: end_index].abs(), dim=0)
        min_index = min_index_real.item() + start_index

        min_sum = min_sum / N
        syn_clone = syn.clone()

        if min_sum < threshold:
            syn = torch.cat([buffer.clone(), syn[:, :, :min_index]], dim=-1)
            buffer = syn_clone[:, :, min_index:]
        else:
            buffer = torch.cat([buffer, syn_clone], dim=-1)
            syn = None
        return buffer, syn

    def run(self, hidden, top_k, prefix, codec_chunk_size=40, codec_padding_size=10, 
            penalty_window_size=-1, penalty=1.1, N=2401, seg_threshold=0.01, export_dir=None):
        """
        Run the speech decoder process.

        Parameters:
        - hidden (torch.Tensor): The output for embedding layer of the language model.
        - top_k (int): The number of top-k tokens to consider during inference.
        - prefix (str, optional): The hidden state from the language model.
        - codec_chunk_size (int, default=40): The size of each chunk to process in the codec model.
        - codec_padding_size (int, default=10): The amount of padding to add on each side of the codec chunk.
        - penalty_window_size (int, default=20): The window size for applying penalties during decoding.
        - penalty (float, default=1.1): The penalty factor.

        Yields:
        - torch.Tensor: Intermediate audio segments generated by the codec model.

        """

        codec_upsample_rate = 600
        left_padding = 0
        right_padding = codec_padding_size
        prefix = None
        buffer = torch.zeros([1, 1, 0]).to(hidden.device)
        with torch.no_grad():
            with torch.autocast(device_type="cpu", dtype=torch.float32):
                # print("Starting TTS...")
                token = torch.full((1, 0), self.model.vocab_size, dtype=torch.long, device=hidden.device)
                for next_token_id in self.infer(hidden, top_k, prefix, penalty_window_size, penalty, export_dir=export_dir):
                    token = torch.cat([token, next_token_id], dim=-1)
                    if token.size(1) == left_padding + codec_chunk_size + right_padding:
                        print(token.shape, token.dtype)
                        syn = self.codec_model.vqvae(token.unsqueeze(-1), 
                                                     torch.tensor(self.codec_model.vqvae.h.global_tokens, 
                                                     device=token.device).unsqueeze(0).unsqueeze(0))
                        print("Codec Done")
                        syn = syn[:, :, left_padding * codec_upsample_rate: -right_padding * codec_upsample_rate]
                        left_padding = codec_padding_size
                        token = token[:, -(left_padding + right_padding):]
                        buffer, syn = self.find_min_sum_index(buffer, syn, N, seg_threshold)
                        if syn is not None:
                            yield syn
                if token.size(1) > 0:
                    print("Codec Done")
                    print(token.shape)
                    syn = self.codec_model.vqvae(token.unsqueeze(-1), 
                                                 torch.tensor(self.codec_model.vqvae.h.global_tokens, 
                                                 device=token.device).unsqueeze(0).unsqueeze(0))
                    syn = syn[:, :, left_padding * codec_upsample_rate:]
                    yield torch.cat([buffer, syn], dim=-1)
