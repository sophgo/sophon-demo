import random
import torch
import copy
import re

from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from vita.model.vita_tts.adapter import *

IGNORE_ID = -1

class AudioLLM(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        llm_path: str,
        freeze_llm: bool = True,
        enc_out_dim: int = 512,
        llm_embed_dim: int = 4096,
        kernel_size: int = 3,
        IGNORE_ID: int = -100,
        adpter_type: str = 'cnn',
        add_audio_bos_eos: bool = False,
        task_num: int = 10,
        add_ctc_prompt_ratio: float = 0.0,
        lang_dict: dict = None,
        ctc: torch.nn.Module = None,
        tokenize_ctc_char: bool = False,
        task_before_audio: bool = False,
        hyp_before_task: bool = False,
        prompt_finetune: bool = False,
        add_prompt_before: bool = False,
        prompt_num: int = 5,
        prefix_finetune: bool = False,
        prefix_num: int = 5,
        llm_head_num: int = 32,
        num_key_value_heads: int = None,
        task_type: str = 'prompt',
        freeze_encoder: bool = False,
        freeze_adpter: bool = False,
        activation_func: str = 'relu',
        norm: str = 'batch',
        use_lora: bool = False,
        clone_encoder: torch.nn.Module = None,
        chat_template: str = None,
        predict_usr_state: int = 0,
        chunk_size: int = -1,
    ):
        super().__init__()

        self.encoder =  encoder
        self.llm_decoder = AutoModelForCausalLM.from_pretrained(llm_path, 
                                                    torch_dtype="auto",
                                                    trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, 
                                                    trust_remote_code=True)
        self.freeze_llm =  freeze_llm
        self.enc_out_dim = enc_out_dim
        self.llm_embed_dim = llm_embed_dim
        self.IGNORE_ID = IGNORE_ID
        self.add_audio_bos_eos = add_audio_bos_eos
        self.add_ctc_prompt_ratio = add_ctc_prompt_ratio
        self.lang_dict = lang_dict
        self.tokenize_ctc_char = tokenize_ctc_char
        self.task_before_audio = task_before_audio
        self.hyp_before_task = hyp_before_task
        self.prompt_finetune = prompt_finetune
        self.add_prompt_before = add_prompt_before
        self.prompt_num = prompt_num
        self.prefix_finetune = prefix_finetune
        self.prefix_num = prefix_num
        self.llm_head_num = llm_head_num
        if num_key_value_heads is None:
            self.num_key_value_heads = llm_head_num
        else:
            self.num_key_value_heads = num_key_value_heads
        self.kv_cache_dim = llm_embed_dim // self.llm_head_num * self.num_key_value_heads
        self.task_type = task_type
        self.freeze_encoder = freeze_encoder
        self.freeze_adpter = freeze_adpter
        self.predict_usr_state = predict_usr_state
        self.chunk_size = chunk_size

        if not hasattr(self.tokenizer, "eod_id"):
            self.tokenizer.eod_id = self.tokenizer.eos_token_id
        if not hasattr(self.llm_decoder, "transformer"):
            self.llm_decoder.transformer = self.llm_decoder.model
            self.llm_decoder.transformer.h = self.llm_decoder.transformer.layers
        if not hasattr(self.llm_decoder.transformer, "wte"):
            self.llm_decoder.transformer.wte = \
                self.llm_decoder.transformer.embed_tokens

        # for chat mode
        if chat_template is not None:
            self.tokenizer.eod_id = self.tokenizer('<|im_end|>'
                                                )['input_ids'][0]
            self.chat_template = {}
            chat_template = chat_template.split('<audio>')
            chat_prefix = chat_template[0].split('<|im_end|>')
            chat_role = chat_prefix[0] + '<|im_end|>'
            self.chat_template['role'] = self.tokenizer(
                        [chat_role], return_tensors="pt")['input_ids']
            self.chat_template['prefix'] = self.tokenizer(
                        [chat_prefix[1]], return_tensors="pt")['input_ids']
            self.chat_template['suffix'] = self.tokenizer(
                        [chat_template[1]], return_tensors="pt")['input_ids']
        else:
            self.chat_template = None

        # for CTC prompt
        if self.add_ctc_prompt_ratio > 0.0:
            assert lang_dict is not None
            assert ctc is not None
            self.ctc = ctc.eval()
            if clone_encoder is None:
                self.clone_encoder = copy.deepcopy(encoder)
            else:
                self.clone_encoder = clone_encoder
            self.clone_encoder.eval()
            for (name, param) in self.clone_encoder.named_parameters():
                param.requires_grad = False
            for (name, param) in self.ctc.named_parameters():
                param.requires_grad = False
        else:
            self.clone_encoder = None

        if self.freeze_llm:
            self.llm_decoder.eval()
            for (name, param) in self.llm_decoder.named_parameters():
                param.requires_grad = False
        
        if use_lora:
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=UNET_TARGET_MODULES,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
            )

        if adpter_type == 'cnn':
            self.adpter = CNNAdapter(enc_out_dim, llm_embed_dim, kernel_size)
        elif adpter_type == 'linear':
            self.adpter = LinearAdapter(enc_out_dim, llm_embed_dim)
        elif adpter_type == 'subsampling':
            self.adpter = CNNSubsampling(enc_out_dim, llm_embed_dim, 
                                        kernel_size, activation_func, norm)
        
        self.task_embeddings = torch.nn.Embedding(task_num, llm_embed_dim)
        if task_type == 'prefix':
            self.prefix_embeddings = nn.ModuleList(
                    [
                        torch.nn.ModuleList(
                            [nn.Embedding(task_num, self.kv_cache_dim),
                            nn.Embedding(task_num, self.kv_cache_dim)]
                        )
                        for i in range(len(self.llm_decoder.transformer.h))
                    ]
                )

        if self.prompt_finetune or self.prefix_finetune:
            if self.prompt_finetune:
                self.prompt_embeddings = nn.Embedding(prompt_num, llm_embed_dim)
                self.prompt_ids = torch.Tensor([i for i in range(prompt_num)]).long()
            if self.prefix_finetune:
                self.prefix_embeddings = nn.ModuleList(
                    [
                        torch.nn.ModuleList(
                            [nn.Embedding(prefix_num, self.kv_cache_dim),
                            nn.Embedding(prefix_num, self.kv_cache_dim)]
                        )
                        for i in range(len(self.llm_decoder.transformer.h))
                    ]
                )
                self.prefix_ids = torch.Tensor([i for i in range(prefix_num)]).long()

        if self.freeze_encoder:
            self.encoder.eval()
            for (name, param) in self.encoder.named_parameters():
                param.requires_grad = False
        if self.freeze_adpter:
            self.adpter.eval()
            for (name, param) in self.adpter.named_parameters():
                param.requires_grad = False

        if self.predict_usr_state:
            self.predictor_head = torch.nn.Linear(llm_embed_dim, predict_usr_state)
        else:
            self.predictor_head = None

        # define task ids
        self.task_ids = {
            "sot": 0,
            "transcribe": 1,
            "translate": 2,
            "zh": 3,
            "en": 4,
            "audio": 5,
            "/audio": 6,
            "hyps": 7,
            "/hyps": 8,
        }
        
    def set_system_role(
        self,
        extra_inputs: Optional[dict] = None,
    ):
        # Ensure 'past_key_values' does not exist in extra_inputs, raise an exception if it does
        assert extra_inputs.get('past_key_values', None) is None, "past key values already exist!!!"
        
        # If 'role' key is present in extra_inputs, use that role as the chat prefix
        if extra_inputs.get('role', None) is not None:
            chat_prefix = self.tokenizer([extra_inputs['role']], 
                return_tensors="pt")['input_ids'].to('cuda')  # Convert role to tokens and move to CUDA device
        else:
            # If no 'role' is provided, use the default chat template and remove the last token (<|im_end|>)
            chat_prefix = self.chat_template['role'][:, :-1].to('cuda')
        
        # Use the LLM decoder's word embedding layer to convert the chat prefix into embeddings
        inputs_embeds = self.llm_decoder.transformer.wte(chat_prefix)
        
        # Create an attention mask with the same shape as the chat prefix, all values set to True
        attention_mask = torch.full(chat_prefix.shape, 
                            True).to(inputs_embeds.device) 
        
        # Prepare the input dictionary containing embeddings and attention mask
        inputs = {
                'inputs_embeds': inputs_embeds.half(),  # Convert embeddings to half precision floats
                'attention_mask': attention_mask,
            }

        # Call the _generate_one_step method to generate one step output, including past_key_values, etc.
        _, past_key_values, stat, _ = self._generate_one_step(
                                                copy.deepcopy(inputs), "sl")
        
        # Return the generated past_key_values
        return past_key_values

    def recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        extra_inputs: Optional[dict] = None,
    ):
        assert extra_inputs.get('past_key_values', None) is not None, "must set system role first!!!"

        buffer = extra_inputs.get('encoder_cache', None)
        cnn_cache = extra_inputs.get('adapter_cache', None)
        pe_index = extra_inputs.get('pe_index', 0)
        if extra_inputs['stat'] == 'sl' or extra_inputs['stat'] == 'cl':
            # Encoder
            
            if buffer is None:
                buffer = [None] * self.encoder.enc[1].num_blocks
            
            encoder_out, buffer, _, _, pe_index = self.encoder.infer(speech, buffer, 
                                                                    0, None, pe_index)

            encoder_mask = torch.full(encoder_out.shape[:2], True).unsqueeze(1
                                                            ).to(encoder_out.device)

            # adapter
            inputs_embeds, encoder_mask, cnn_cache = self.adpter(encoder_out, encoder_mask, 
                                            cache=cnn_cache, return_cache=True) # 1, T, D

            attention_mask = encoder_mask.squeeze(1) # 1, T

        # prompt
        if extra_inputs['stat'] == 'sl':
            if self.prompt_finetune:
                prompt_ids = self.prompt_ids.repeat(1, 1).to(inputs_embeds.device)
                prompt_embeds = self.prompt_embeddings(
                                    prompt_ids.to(inputs_embeds.device)) # B, 5, D
                prompt_mask = torch.full(prompt_ids.shape, 
                                    True).to(inputs_embeds.device) # B, 5
                
                if self.add_prompt_before:
                    inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), 1) # B, (T+5), D
                    attention_mask = torch.cat((prompt_mask, attention_mask), 1) # B, (T+5)

        # chat mode
        if self.chat_template is not None:
            if extra_inputs['stat'] == 'sl':
                chat_prefix = self.chat_template['prefix'].to(
                                                    inputs_embeds.device)
                chat_prefix = torch.cat((torch.tensor([[self.tokenizer.eod_id]]
                                    ).to(inputs_embeds.device), chat_prefix), 1)
                chat_prefix_embeds = self.llm_decoder.transformer.wte(chat_prefix)
                chat_prefix_mask = torch.full(chat_prefix.shape, 
                                True).to(inputs_embeds.device)
                inputs_embeds = torch.cat((chat_prefix_embeds, inputs_embeds), 1)
                attention_mask = torch.cat((chat_prefix_mask, attention_mask), 1)
            if extra_inputs['stat'] == 'ss':
                chat_suffix = self.chat_template['suffix'].to('cuda')
                chat_suffix_embeds = self.llm_decoder.transformer.wte(chat_suffix)
                chat_suffix_mask = torch.full(chat_suffix.shape, True).to('cuda')
                inputs_embeds = chat_suffix_embeds
                attention_mask = chat_suffix_mask
        
        if extra_inputs['stat'] != 'cs':
            inputs = {
                'inputs_embeds': inputs_embeds.half(),
                'attention_mask': attention_mask,
            }
        else:
            attention_mask = torch.full([1, 1], True).to('cuda')
            inputs = {
                'input_ids': extra_inputs['last_id'],
                'attention_mask': attention_mask
            }

        # add kv cache
        inputs['past_key_values'] = extra_inputs['past_key_values']
        past_mask = torch.full([1, inputs['past_key_values'][0][0].size(2)],
                                True).to('cuda')
        attention_mask = torch.cat((past_mask, attention_mask), 1)
        inputs['attention_mask'] = attention_mask

        top_p = extra_inputs.get('top_p', 1.0)
        top_k = extra_inputs.get('top_k', 0)
        temperature = extra_inputs.get('temperature', 1.0)

        last_id, past_key_values, stat, hidden_state = self._generate_one_step(copy.deepcopy(inputs), 
                                                extra_inputs['stat'],
                                                top_p=top_p, 
                                                top_k=top_k,
                                                temperature=temperature)

        return last_id, stat, past_key_values, cnn_cache, buffer, pe_index, hidden_state
    
    def _post_decode(self, output, temperature=1.0, top_k=0, top_p=0.0):
        """
        Decoding function, based on the posterior probability output, 
        uses top_k, top_p, and temperature parameters for sampling.

        Parameters:
        - output: torch.Tensor, shaped as (1, 1, D), represents the posterior probability output by the model.
        - top_k: int, indicates selecting the top k tokens with the highest probability for sampling.
                      If 0, no top_k filtering is performed.
        - top_p: float, indicates selecting tokens with cumulative probability not exceeding p for sampling.
                        If 0.0, no top_p filtering is performed.
        - temperature: float, represents the sampling temperature parameter. 
                              The higher the value, the more random the sampling; 
                            the lower the value, the more deterministic the sampling.

        Returns:
        - Selected token index.
        """
        output = output.squeeze(0).squeeze(0)

        # temperature
        if temperature != 1.0:
            output = output / temperature

        probs = torch.nn.functional.softmax(output, dim=-1)

        # top_k
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
            probs = probs / probs.sum()

        # top_p
        if top_p > 0.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            if sorted_indices_to_remove[0]:
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()

        token_index = torch.multinomial(probs, 1)
        return token_index.unsqueeze(0)
    
    def _generate_one_step(
        self,
        inputs,
        stat,
        top_p: float = 1.0,
        top_k: int = 0,
        temperature: float = 1.0,
    ):
        """
        Generates the model's next output based on the current input and state.

        Parameters:
        - inputs: The input tensor containing the model's input data.
        - stat: The current state information used to control the generation process.
        - top_p: The threshold for controlling top-p sampling.
        - top_k: The threshold for controlling top-k sampling.
        - temperature: Controls the randomness of sampling.

        Returns:
        - last_id: The index of the last generated token.
        - stat: The updated state information.
        - past_key_values: The model's historical key-value pairs, used for cross-step memory.
        - hidden_state: The model's hidden state, used to maintain cross-step contextual information.
        """
        outputs = self.llm_decoder.model(**inputs)
        if stat == 'sl' or stat == 'cl':
            state_logits = self.predictor_head(
                        outputs['last_hidden_state'])[0, :]
            prob = F.softmax(state_logits[:, :-1])
            state_prob = prob[-1].clone()
            state_1 = state_prob[1]
            state_2 = state_prob[2]
            print("State 1 prob: {:.4f}, State 2 prob: {:.4f}".format(state_1.item(), state_2.item()))
            if state_2 > 0.5:
                return None, outputs['past_key_values'], 'el', None
            if state_1 > 0.5:
                return None, outputs['past_key_values'], 'ss', None
            return None, outputs['past_key_values'], 'cl', None

        last_logit = self.llm_decoder.lm_head(outputs['last_hidden_state'][:, -1:, :])
        last_id = self._post_decode(last_logit, temperature=temperature, top_k=top_k, top_p=top_p)
        return_tts_state = outputs['last_hidden_state'][:, -1:, :]

        if last_id[0][0] == self.tokenizer.eod_id:
            return None, outputs['past_key_values'], 'sl', return_tts_state
        else:
            return last_id, outputs['past_key_values'], 'cs', return_tts_state
