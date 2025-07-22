import os
import json
import copy
import torch, torchvision, torchaudio
import re
from tqdm import tqdm
from typing import Optional, Tuple
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor, AutoFeatureExtractor
import numpy as np
from vita.model.vita_tts.decoder.llm2tts import llm2TTS
def rename_class(cls_instance):
    cls = cls_instance.__class__
    cls.__module__ = cls.__module__.replace("-", "_") # split(".")[-1]

def logging(message):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(message)
            return func(*args, **kwargs)
        return wrapper
    return decorator

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

class ModelMapper:
    def __init__(self):
        self.attrs = []
        self.mapper = dict()
        self.regist_models()
    def regist_models(self):
        self.regist_vita()
    def regist(self, model_type, model_map):
        assert('config' in model_map and
               'decoder' in model_map and
               'attention' in model_map)
        self.mapper[model_type] = model_map
    def regist_vita(self):
        vita_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_hidden_layers',
                'num_key_value_heads': 'num_key_value_heads',
                'rope_theta': 'rope_theta',
                'vocab_size': 'vocab_size'
            },
            'model': {
                'lm_': 'lm_head',
                'embed_': 'model.embed_tokens',
                'blocks_': 'model.layers',
                'final_layernorm_': 'model.norm',
                'visual_model': 'model.vision_tower',
                'mm_projector': 'model.mm_projector',
                'speech_model': 'model.audio_encoder',
            },
            'decoder': {
                'self_attn': 'self_attn',
                'mlp': 'mlp',
                'input_layernorm': 'input_layernorm',
                'post_attention_layernorm': 'post_attention_layernorm'
            },
            'attention': {
                'q_proj': 'q_proj',
                'k_proj': 'k_proj',
                'v_proj': 'v_proj',
                'o_proj': 'o_proj',
                'rotary_emb': 'rotary_emb'
            }
        }
        self.regist('vita-Qwen2', vita_map)
    def get_map(self, config):
        model_type = config.model_type
        if model_type in self.mapper:
            return self.mapper[model_type]
    def get_map_by_model_type(self, model_type):
        if model_type in self.mapper:
            return self.mapper[model_type]
    @staticmethod
    def do_map(dst, src, map):
        for dst_attr, src_attr in map.items():
            attributes = src_attr.split('.')
            obj = src
            for attr in attributes:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    obj = None
                    break
            setattr(dst, dst_attr, obj)

class OnnxRebuilder:
    def __init__(self,
				 onnx_dir: str,
                 tts_dir: str,
                 model_path: str,
				 seq_length: int,
                 model_type: str,
				 embedding_disk: bool,
				 lmhead_with_topk: bool,
                 config):
        self.onnx_model = None
        self.model_path = model_path
        self.dtype = torch.float32
        self.onnx_dir = onnx_dir
        self.tts_dir = tts_dir
        self.seq_length = seq_length
        self.model_type = model_type
        self.embedding_disk = embedding_disk
        self.lmhead_with_topk = lmhead_with_topk
        self.config = config
        self.model_mapper = ModelMapper()

    def _replace_initializer(self, old_init, new_init):
        """
        Replaces the contents of an existing initializer with new data.

        Args:
            old_init (onnx.onnx_ml_pb2.TensorProto): The existing initializer to be replaced.
            new_init (onnx.onnx_ml_pb2.TensorProto): The new initializer with updated data.
        """
        old_init.CopyFrom(new_init)

    def rebuild_config(self):
        if not hasattr(self, 'num_key_value_heads') or self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.rope_theta is None:
            self.rope_theta = 10000.0
        self.head_dim = self.hidden_size // self.num_attention_heads

    def rebuild_modules(self):
        # Embedding
        self.embed = Embedding(self.embed_, self.hidden_size)

        # Rotary
        self.rotary = Rotary(self)

        # Blocks
        self.blocks = []
        for block in self.blocks_.children():
            self.blocks.append(Decoder(block, self))

        # Lmhead
        self.lm = Lm(self)

        # Visual
        if hasattr(self, 'visual_model') and self.visual_model is not None:
            self.visual = VitaVisionEmbedding(self.visual_model, self)

        # Speech
        if hasattr(self, 'speech_model') and self.speech_model is not None:
            self.speech = VitaSpeechEmbedding(self.speech_model, self)
    @logging("export_config ...")
    def export_config(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.save_pretrained(f'{self.onnx_dir}/../config')
        config_dict = self.config.to_dict()
        with open(f'{self.onnx_dir}/../config/config.json', "w") as f:
            json.dump(config_dict, f, indent=4)
        return

    @logging("export_processor ...")
    def export_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor.save_pretrained(f'{self.onnx_dir}/../processor')
        return

    @logging("export_embed ...")
    def export_embed(self):
        if not hasattr(self, 'embed') or not isinstance(self.embed.embed, torch.nn.Embedding):
            return

        embed_model = copy.deepcopy(self.embed).float()
        if not self.embedding_disk:
            embedding_file = f'{self.onnx_dir}/embedding.pt'
            if os.path.exists(embedding_file):
                print(f"{embedding_file} already exists. Skipping export.")
                return
            input_ids = torch.tensor([range(self.seq_length)], dtype=torch.int32)
            module = torch.jit.trace(embed_model.forward, input_ids)
            torch.jit.save(module, embedding_file)
        else:
            embedding_file = f'{self.onnx_dir}/../embedding.bin'
            if os.path.exists(embedding_file):
                print(f"{embedding_file} already exists. Skipping export.")
                return
            import ctypes
            tensor_data = embed_model.embed.weight.data.to(torch.bfloat16)
            data_ptr = tensor_data.untyped_storage().data_ptr()
            buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
            with open(embedding_file, 'wb') as f:
                f.write(buffer)

    @logging("export_block ...")
    def export_block(self):
        hidden_states = torch.randn((1, self.seq_length, self
                                     .hidden_size), dtype=torch.float32)
        position_ids = torch.tensor([range(self.seq_length)], dtype=torch.long)
        attention_mask = torch.randn((1, 1, self.seq_length, self.seq_length), dtype=torch.float32)

        for i in tqdm(range(self.num_hidden_layers)):
            onnx_path = f'{self.onnx_dir}/block_{i}.onnx'
            if os.path.exists(onnx_path):
                print(f"{onnx_path} already exists. Skipping export.")
                continue

            model = self.blocks[i].float()
            torch.onnx.export(
                model,
                (hidden_states, position_ids, attention_mask),
                onnx_path,
                verbose=False,
                input_names=["input_states", "position_ids", "attention_mask"],
                output_names=["hidden_states", "past_k", "past_v"],
                do_constant_folding=False, # set False to keep original name
                opset_version=15
            )

        self.onnx_model = None

    @logging("export_block_cache ...")
    def export_block_cache(self):
        hidden_states = torch.randn((1, 1, self.hidden_size), dtype=torch.float32)
        position_ids = torch.tensor([range(1)], dtype=torch.long)
        attention_mask = torch.ones(
            (1, 1, 1, self.seq_length + 1), dtype=torch.float32)
        past_k = torch.randn((1, self.seq_length, self.num_key_value_heads, self.head_dim), dtype=torch.float32)
        past_v = torch.randn((1, self.seq_length, self.num_key_value_heads, self.head_dim), dtype=torch.float32)

        for i in tqdm(range(self.num_hidden_layers)):
            onnx_path = f'{self.onnx_dir}/block_cache_{i}.onnx'
            if os.path.exists(onnx_path):
                print(f"{onnx_path} already exists. Skipping export.")
                continue

            model = self.blocks[i].float()
            torch.onnx.export(
                model,
                (hidden_states, position_ids, attention_mask, (past_k, past_v)),
                onnx_path,
                verbose=False,
                input_names=["input_states", "position_ids", "attention_mask", "history_k", "history_v"],
                output_names=["hidden_states", "past_k", "past_v"],
                do_constant_folding=False,
                opset_version=15,
            )
        self.onnx_model = None

    @logging("export_lm_head ...")
    def export_lm_head(self):
        lmhead_file = f'{self.onnx_dir}/lm_head.pt'
        if os.path.exists(lmhead_file):
            print(f"{lmhead_file} already exists. Skipping export.")
            return
        model = self.lm.float()
        hidden_states = torch.randn((1, self.hidden_size), dtype=torch.float32)
        module = torch.jit.trace(model.forward, hidden_states)
        torch.jit.save(module, lmhead_file)

    @logging("export_greedy_head ...")
    def export_greedy_head(self):
        onnx_path = f'{self.onnx_dir}/greedy_head.onnx'
        if os.path.exists(onnx_path):
            print(f"{onnx_path} already exists. Skipping export.")
            return

        model = GreedyHead().float()
        m_logits = torch.randn(1, self.vocab_size)
        torch.onnx.export(
            model, (m_logits),
            onnx_path,
            verbose=False,
            input_names=['m_logits'],
            output_names=['token'],
            do_constant_folding=True,
            opset_version=15)

    @logging("export_penalty_sample_head ...")
    def export_penalty_sample_head(self):
        onnx_path = f'{self.onnx_dir}/penalty_sample_head.onnx'
        if os.path.exists(onnx_path):
            print(f"{onnx_path} already exists. Skipping export.")
            return

        model = PenaltySampleHead()
        m_logits = torch.randn(1, self.vocab_size)
        input_ids = torch.tensor([range(self.seq_length)])
        top_p = torch.tensor([0.8])
        temperature = torch.tensor([0.98])
        penalty = torch.tensor([0.98])

        torch.onnx.export(
            model, (m_logits, input_ids, top_p, temperature, penalty),
            onnx_path,
            verbose=False,
            input_names=[
                'm_logits', 'input_ids', 'top_p', 'temperature',
                'penalty'
            ],
            output_names=['probs', 'token'],
            do_constant_folding=True,
            opset_version=15)

    @logging("export_visual...")
    def export_visual(self):
        onnx_path = os.path.join(self.onnx_dir, "vit.onnx")
        if os.path.exists(onnx_path):
            print(f"{onnx_path} already exists. Skipping export.")
            return

        self.visual.export(onnx_path)
        return
    
    @logging("export_speech...")
    def export_speech(self):
        onnx_path = os.path.join(self.onnx_dir, "speech.onnx")
        if os.path.exists(onnx_path):
            print(f"{onnx_path} already exists. Skipping export.")
            return

        self.speech.export(onnx_path)
        return

    def test_net_with_mask_new(self): #使用简化过的image_embedding模型处理
        dtype = self.dtype
        image_token = 151655
        audio_token = 151656
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        # prompt = f'{user_prompt}decribe this image in detail.{prompt_suffix}{assistant_prompt}'
        prompt = f'{prompt_suffix}{assistant_prompt}'
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = processor
        ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        
        image = Image.open("vita_newlog.jpg").convert('RGB')
        img_processor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            ),
        ])
        image = torchvision.transforms.functional.resize(image, [448, 448])
        img_embeds = img_processor(image)
        img_set_tensor = self.visual(img_embeds.unsqueeze(0)).squeeze()

        text_ids = self.tokenizer.encode(prompt)
        user_prompt_ids = self.tokenizer.encode(user_prompt)   
        audio_processor = self.speech_model.audio_processor
        audio, audio_for_llm_lens = audio_processor.process('q1.wav')
        audio_length = audio.shape[0]
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        audio_set_tensor = self.speech(audio, audio_length).squeeze()
        ids = user_prompt_ids + [image_token] * img_set_tensor.shape[0] + [audio_token] * audio_set_tensor.shape[0] + text_ids
        # ids = user_prompt_ids + [image_token] * img_set_tensor.shape[0] + text_ids
        # ids = user_prompt_ids + text_ids
        token_len = len(ids)
        input_ids = ids + [0] * (self.seq_length - token_len)
        input_ids = torch.tensor(input_ids).view(self.seq_length)
        hidden_states = self.embed(input_ids)

        positions_tuple = torch.nonzero(input_ids == image_token, as_tuple=True)
        new_hidden_states = hidden_states.squeeze(0).index_put(
            indices=positions_tuple,
            values=img_set_tensor,
            accumulate=False
        )
        positions_tuple = torch.nonzero(input_ids == audio_token, as_tuple=True)
        new_hidden_states = new_hidden_states.squeeze(0).index_put(
            indices=positions_tuple,
            values=audio_set_tensor,
            accumulate=False
        )
        hidden_states = new_hidden_states.view(1, self.seq_length, self.hidden_size)

        out = hidden_states  # [1, seq_length, 3072]
        position_ids = list(range(token_len)) + (self.seq_length - token_len) * [0]
        position_ids = torch.tensor([position_ids])
        attention_mask = torch.ones((self.seq_length, self.seq_length)).float() * -10000.0
        for i in range(token_len):
            for j in range(token_len):
                if j <= i:
                    attention_mask[i][j] = 0.0
        attention_mask = attention_mask.view(
            1, 1, self.seq_length, self.seq_length)
        k_cache = []
        v_cache = []
        
        for i in range(self.config.num_hidden_layers):
            out, kv = self.blocks[i](out.to(dtype), position_ids, attention_mask.to(dtype))
            k, v = kv
            k[:, token_len:, :, :] = 0
            v[:, token_len:, :, :] = 0
            k_cache.append(k)
            v_cache.append(v)

        out = out[:, token_len - 1:token_len].view(1, 1, self.hidden_size)
        logits = self.lm(out.to(dtype))
        _, token = torch.topk(logits.float(), 1)
        out_ids = [int(token)]
        while int(token) not in [self.config.eos_token_id, ID_IM_END, ID_END] and token_len < self.seq_length:
            token_len += 1
            input_ids = torch.tensor([token])
            out = self.embed(input_ids).view(1, 1, self.hidden_size)
            position_ids = torch.tensor([[token_len - 1]])
            attention_mask = torch.zeros(
                (1, 1, 1, self.seq_length + 1)).float()
            attention_mask[:, :, :, token_len-1:] = -10000.0
            for i in range(self.config.num_hidden_layers):
                # block_input_dict = {"input_states": out.numpy(), "position_ids": position_ids.numpy(), "attention_mask": attention_mask.numpy(),
                #                     "history_k": k_cache[i].numpy(), "history_v": v_cache[i].numpy()}
                # np.savez("block_cache_input.npz", **block_input_dict)
                # exit(1)
                out, kv = self.blocks[i](out.to(dtype), position_ids,
                                        attention_mask.to(dtype),
                                        (k_cache[i].to(dtype), v_cache[i].to(dtype)))
                k, v = kv
                k_cache[i][:, token_len-1:token_len, :, :] = k[:, :, :, :]
                v_cache[i][:, token_len-1:token_len, :, :] = v[:, :, :, :]
            logits = self.lm(out.to(dtype))
            _, token = torch.topk(logits.float(), 1)
            out_ids.append(int(token))
        words = self.tokenizer.decode(out_ids)
        print(words)
        print("\noutput_ids:{}".format(out_ids))

        # tts
        decoder_topk = 2
        codec_chunk_size = 40
        codec_padding_size = 10
        tts = llm2TTS(os.path.join(self.model_path, 'vita_tts_ckpt/'))
        llm_resounse = replace_equation(remove_special_characters("Hello! How can I assist you today? If you have any questions or need information, feel free to ask."))
        #print('tts_text', llm_resounse)
        segs = None
        
        for idx, text in enumerate(split_into_sentences(llm_resounse)):
            tts_input_ids = self.tokenizer.encode(text)
            tts_embedding = self.embed(torch.tensor(tts_input_ids))
            # save_npy = tts_embedding.reshape(-1, 896).unsqueeze(0).numpy()
            # np.save("tts_embedding.npy", save_npy)
            for seg in tts.run(tts_embedding.reshape(-1, 896).unsqueeze(0).to("cpu"), decoder_topk,
                                None, 
                                codec_chunk_size, codec_padding_size):
                if segs is None:
                    segs = seg
                else:
                    segs = torch.cat((segs, seg), dim=2)        

        torchaudio.save("test_stream.wav", segs.squeeze(0).cpu(), sample_rate=24000)
# some wrapper class for export
class Embedding(torch.nn.Module):
    def __init__(self, embed, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = embed

    def forward(self, input_ids):
        return self.embed(input_ids).view(1, -1, self.hidden_size)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

class Attention(torch.nn.Module):
    def __init__(self, attn, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rotary = config.rotary
        ModelMapper.do_map(self, attn, config.model_map['attention'])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # rope
        if rotary_pos_emb is None:
            cos, sin = self.rotary.cos[position_ids], self.rotary.sin[position_ids]
        else:
            cos, sin = rotary_pos_emb
        cos, sin = cos.to(query_states.dtype), sin.to(query_states.dtype)
        query_states = self.rotary.apply_rotary_pos(query_states, cos, sin)
        key_states = self.rotary.apply_rotary_pos(key_states, cos, sin)
        past_kv = (key_states, value_states)
        # kv cache
        if past_key_value is not None:
            past_key, past_value = past_key_value[0], past_key_value[1]
            key_states = torch.cat((past_key, key_states), dim=1)
            value_states = torch.cat((past_value, value_states), dim=1)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        #------- attention ----------
        # query_states @ key_states
        attn_weights = torch.matmul(query_states.transpose(1, 2), key_states.transpose(1, 2).transpose(2, 3)) / (self.head_dim ** 0.5)
        # attention_mask
        attn_weights = attn_weights + attention_mask
        # upcast softmax to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights @ value_states
        attn_output = torch.matmul(attn_weights, value_states.transpose(1, 2))

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, past_kv

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class Rotary(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_length = config.seq_length
        self.rope_theta = config.rope_theta
        self.rotary_dim = config.head_dim
        self.model_type = config.model_type
        if hasattr(config, 'rotary_dim'):
            self.rotary_dim = config.rotary_dim
        self.cos, self.sin = self.init_rotary_pos_emb(self.seq_length)
        self.cos = self.cos.squeeze(0)
        self.sin = self.sin.squeeze(0)

    def init_rotary_pos_emb(self, seq_length):
        position_ids = torch.tensor([range(seq_length)], dtype=torch.long)
        theta = 1.0 / (self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        if self.model_type != 'chatglm2':
            rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
        return rotary_pos_emb

    def apply_rotary_pos(self, x, cos, sin):
        return self.llama_rotary_pos(x, cos, sin)

    def llama_rotary_pos(self, x, cos, sin):
        x = (x * cos) + (rotate_half(x) * sin)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, decoder, config):
        super().__init__()
        ModelMapper.do_map(self, decoder, config.model_map['decoder'])
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(self.self_attn, config)

        # chatglm
        self.alpha = (2 * config.num_hidden_layers) ** 0.5 if config.model_type == 'chatglm' else 1.0
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        norm_hidden_states = hidden_states
        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            rotary_pos_emb=rotary_pos_emb
        )
        # Fully Connected
        if self.alpha != 1.0:
            # chatglm-6b
            hidden_states = norm_hidden_states * self.alpha + hidden_states
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_input * self.alpha + mlp_output
        elif hasattr(self, 'post_attention_layernorm'):
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            feed_forward_hidden_states = self.mlp(norm_hidden_states)
            hidden_states = hidden_states + feed_forward_hidden_states + residual

        return hidden_states, present_key_value

class Lm(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        rename_class(config.final_layernorm_)

        self.final_layernorm = config.final_layernorm_
        self.lm = config.lm_
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states):
        hidden_states = self.final_layernorm(hidden_states)
        m_logits = self.lm(hidden_states)
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

class VitaVisionEmbedding(torch.nn.Module):
    def __init__(self, vita_image_embed, base):
        super().__init__()
        self.base = base
        self.vita_image_embed = vita_image_embed

    def forward(self, image_tensor):
        dtype = self.base.dtype
        image_tensor = image_tensor.to(dtype)
        img_features = self.vita_image_embed(image_tensor)
        img_feature_proj = self.base.mm_projector(img_features)
        return img_feature_proj

    def export(self, onnx_path):
        img_embeds = torch.rand(1, 3, 448, 448) - 0.5
        model = self.float().eval()
        torch.onnx.export(
            model, (img_embeds),
            onnx_path,
            verbose=False,
            input_names=['img_embeds'],
            output_names=['hidden_states'],
            do_constant_folding=True,
            opset_version=17
        )
        del model
        return

class VitaSpeechEmbedding(torch.nn.Module):
    def __init__(self, vita_audio_embed, base):
        super().__init__()
        self.base = base
        self.vita_audio_embed = vita_audio_embed
    def forward(self, input_embeds, lens):
        dtype = self.base.dtype
        input_embeds = input_embeds.to(dtype)
        audio_feature_proj = self.vita_audio_embed(input_embeds, lens)
        return audio_feature_proj["inputs_embeds"]

    def export(self, onnx_path):
        audio_embeds = torch.rand(1, 384, 80)
        audio_length = audio_embeds.shape[1]
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        model = self.float().eval()
        torch.onnx.export(
            model, (audio_embeds, audio_length),
            onnx_path,
            verbose=False,
            input_names=['audio_embeds', 'audio_length'],
            output_names=['hidden_states'],
            do_constant_folding=True,
            opset_version=17
        )
        del model
        return
