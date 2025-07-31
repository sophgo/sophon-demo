#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import torch
import numpy as np
import soundfile as sf
import re
from vita_tts.decoder.decoder import LLM2TTSCodecAR
from vita_tts.decoder.vqvae import VQVAE
# from vita_tts.decoder.ticodec.vqvae_tester import VqvaeTester

def split_into_sentences(text):
    sentence_endings = re.compile(r'[，。？\n！？、,?.!]')
    sentences = sentence_endings.split(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]
def remove_special_characters(input_str):
    # Remove special tokens
    special_tokens = ['☞', '☟', '☜', '<unk>', '<|im_end|>']
    for token in special_tokens:
        input_str = input_str.replace(token, '')
    return input_str

def replace_equation(sentence):
    special_notations = {
        "sin": " sine ",
        "cos": " cosine ",
        "tan": " tangent ",
        "cot": " cotangent ",
        "sec": " secant ",
        "csc": " cosecant ",
        "log": " logarithm ",
        "exp": "e^",
        "sqrt": "根号 ",
        "abs": "绝对值 ",
    }
    
    special_operators = {
        "+": "加",
        "-": "减",
        "*": "乘",
        "/": "除",
        "=": "等于",
        '!=': '不等于',
        '>': '大于',
        '<': '小于',
        '>=': '大于等于',
        '<=': '小于等于',
    }

    greek_letters = {
        "α": "alpha ",
        "β": "beta ",
        "γ": "gamma ",
        "δ": "delta ",
        "ε": "epsilon ",
        "ζ": "zeta ",
        "η": "eta ",
        "θ": "theta ",
        "ι": "iota ",
        "κ": "kappa ",
        "λ": "lambda ",
        "μ": "mu ",
        "ν": "nu ",
        "ξ": "xi ",
        "ο": "omicron ",
        "π": "派 ",
        "ρ": "rho ",
        "σ": "sigma ",
        "τ": "tau ",
        "υ": "upsilon ",
        "φ": "phi ",
        "χ": "chi ",
        "ψ": "psi ",
        "ω": "omega "
    }

    sentence = sentence.replace('**', ' ')

    sentence = re.sub(r'(?<![\d)])-(\d+)', r'负\1', sentence)

    for key in special_notations:
        sentence = sentence.replace(key, special_notations[key]) 
    for key in special_operators:
        sentence = sentence.replace(key, special_operators[key])
    for key in greek_letters:
        sentence = sentence.replace(key, greek_letters[key])


    sentence = re.sub(r'\(?(\d+)\)?\((\d+)\)', r'\1乘\2', sentence)
    sentence = re.sub(r'\(?(\w+)\)?\^\(?(\w+)\)?', r'\1的\2次方', sentence)
    
    return sentence

class llm2TTS():
    def __init__(self, codec_args):
        # self.model = self.get_model(model_path).cuda().to(
        #                             torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        #                             )
        self.model = LLM2TTSCodecAR(codec_args)
        self.infer = self.model.infer
        self.vqvae = VQVAE(codec_args)
        # self.codec_model = VqvaeTester(config_path=model_path + "/codec/model.json", 
        #                                 model_path=model_path + "/codec/final.pt",
        #                                 sample_rate=24000)
        # self.codec_model = self.codec_model.cpu()
        # self.codec_model.vqvae.generator.remove_weight_norm()
        # self.codec_model.vqvae.encoder.remove_weight_norm()
        # self.codec_model.eval()
    
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

    def run_OLD(self, hidden, top_k, prefix, codec_chunk_size=40, codec_padding_size=10, 
            penalty_window_size=-1, penalty=1.1, N=2401, seg_threshold=0.01):
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
        buffer = torch.zeros([1, 1, 0], device='cpu')
        with torch.no_grad():
            with torch.autocast(device_type="cpu", dtype=torch.float32):
                # print("Starting TTS...")
                token = torch.full((1, 0), self.model.vocab_size, dtype=torch.long, device='cpu')
                for next_token_id in self.infer(hidden):
                    token = torch.cat([token, torch.tensor(next_token_id)], dim=-1)
                    if token.size(1) == left_padding + codec_chunk_size + right_padding:
                        syn = self.codec_model.vqvae(token.unsqueeze(-1), 
                                                     torch.tensor(self.codec_model.vqvae.h.global_tokens, 
                                                     device=token.device).unsqueeze(0).unsqueeze(0))
                        syn = syn[:, :, left_padding * codec_upsample_rate: -right_padding * codec_upsample_rate]
                        left_padding = codec_padding_size
                        token = token[:, -(left_padding + right_padding):]
                        buffer, syn = self.find_min_sum_index(buffer, syn, N, seg_threshold)
                        if syn is not None:
                            yield syn
                if token.size(1) > 0:
                    syn = self.codec_model.vqvae(token.unsqueeze(-1), 
                                                 torch.tensor(self.codec_model.vqvae.h.global_tokens, 
                                                 device=token.device).unsqueeze(0).unsqueeze(0))
                    syn = syn[:, :, left_padding * codec_upsample_rate:]
                    yield torch.cat([buffer, syn], dim=-1)
    
    def run(self, hidden, top_k, prefix, codec_chunk_size=40, codec_padding_size=10, 
            penalty_window_size=-1, penalty=1.1, N=2401, seg_threshold=0.01):
        codec_upsample_rate = 600
        left_padding = 0
        right_padding = codec_padding_size
        buffer = torch.zeros([1, 1, 0], dtype=torch.int32)
        token = np.full((1, 0), self.model.vocab_size, dtype=np.int32)
        for next_token_id in self.infer(hidden):
            token = np.concatenate([token, np.array(next_token_id)], axis=-1)
            if token.shape[1] == left_padding + codec_chunk_size + right_padding:
                padded_token = np.pad(
                    token,
                    pad_width=((0, 0), (0, self.vqvae.feat_len - token.shape[1])),
                    mode="constant",
                    constant_values=0,
                )
                syn = self.vqvae(np.expand_dims(padded_token, -1), self.vqvae.global_tokens)
                rate = syn.shape[2] // padded_token.shape[1]
                syn = syn[:,:,:token.shape[1] * rate]
                syn = torch.tensor(syn[:, :, left_padding * codec_upsample_rate: -right_padding * codec_upsample_rate])
                left_padding = codec_padding_size
                token = token[:, -(left_padding + right_padding):]
                buffer, syn = self.find_min_sum_index(buffer, syn, N, seg_threshold)
                if syn is not None:
                    yield syn
        if token.shape[1] > 0:
            padded_token = np.pad(
                token,
                pad_width=((0, 0), (0, self.vqvae.feat_len - token.shape[1])),
                mode="constant",
                constant_values=0,
            )
            syn = self.vqvae(np.expand_dims(padded_token, -1), self.vqvae.global_tokens)
            rate = syn.shape[2] // padded_token.shape[1]
            syn = syn[:,:,:token.shape[1] * rate]
            syn = torch.tensor(syn[:, :, left_padding * codec_upsample_rate:])
            yield torch.cat([buffer, syn], dim=-1)
