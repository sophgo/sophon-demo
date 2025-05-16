import os
import json
import copy
import torch, torchvision
import soundfile
from tqdm import tqdm
from typing import Optional, Tuple
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor, AutoFeatureExtractor
import numpy as np

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

class ModelMapper:
    def __init__(self):
        self.attrs = []
        self.mapper = dict()
        self.regist_models()
    def regist_models(self):
        self.regist_phi4mm()
    def regist(self, model_type, model_map):
        assert('config' in model_map and
               'decoder' in model_map and
               'attention' in model_map)
        self.mapper[model_type] = model_map
    def regist_phi4mm(self):
        phi_map = {
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
                'visual_model': 'model.embed_tokens_extend.image_embed',
                'speech_model': 'model.embed_tokens_extend.audio_embed',
            },
            'decoder': {
                'self_attn': 'self_attn',
                'mlp': 'mlp',
                'input_layernorm': 'input_layernorm',
                'post_attention_layernorm': 'post_attention_layernorm'
            },
            'attention': {
                'qkv_proj': 'qkv_proj',
                'o_proj': 'o_proj',
                'rotary_emb': 'rotary_emb'
            }
        }
        self.regist('phi4mm', phi_map)
        
    def get_map(self, config):
        model_type = config.model_type
        if model_type in self.mapper:
            return self.mapper[model_type]
        return self.default_map

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
            self.visual = Phi4mmVisionEmbedding(self.visual_model, self)

        # Speech
        if hasattr(self, 'speech_model') and self.speech_model is not None:
            self.speech = Phi4mmSpeechEmbedding(self.speech_model, self)
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
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        # prompt = f'{user_prompt}what is the answer for 1+1? Explain it.{prompt_suffix}{assistant_prompt}'
        prompt = f'{prompt_suffix}{assistant_prompt}'
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = processor.tokenizer
        ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        
        image = Image.open("australia.jpg")
        img_processor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            ),
        ])
        image = torchvision.transforms.functional.resize(image, [448, 448])
        img_embeds = img_processor(image)
        img_set_tensor = self.visual(img_embeds.unsqueeze(0)).squeeze()
        
        audio = soundfile.read("what_is_the_traffic_sign_in_the_image.wav")
        audio_processor = AutoFeatureExtractor.from_pretrained(self.model_path)
        audio_embeds = audio_processor([audio], return_tensors='pt')['input_audio_embeds']
        audio_set_tensor = self.speech(audio_embeds).squeeze()
        text_ids = self.tokenizer.encode(prompt)
        user_prompt_ids = self.tokenizer.encode(user_prompt)
        ids = user_prompt_ids + [200010] * img_set_tensor.shape[0] + [200011] * audio_set_tensor.shape[0] + text_ids
        token_len = len(ids)
        input_ids = ids + [0] * (self.seq_length - token_len)
        input_ids = torch.tensor(input_ids).view(self.seq_length)
        hidden_states = self.embed(input_ids)

        positions_tuple = torch.nonzero(input_ids == 200010, as_tuple=True)
        new_hidden_states = hidden_states.squeeze(0).index_put(
            indices=positions_tuple,
            values=img_set_tensor,
            accumulate=False
        )
        positions_tuple = torch.nonzero(input_ids == 200011, as_tuple=True)
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
            k[:, :, token_len:, :] = 0
            v[:, :, token_len:, :] = 0
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
            attention_mask[:, :, :, token_len-1:self.seq_length] = -10000.0
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
    def test_net_with_mask(self): #使用原始image_embedding模型处理
        dtype = self.dtype
        
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        # prompt = f'{user_prompt}what is the answer for 1+1? Explain it.{prompt_suffix}{assistant_prompt}'
        prompt = f'{user_prompt}<|image_1|>What is shown in this image?{prompt_suffix}{assistant_prompt}'
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = processor.tokenizer
        ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        
        image = Image.open("australia.jpg")
        inputs = processor(text=prompt, images=image, return_tensors='pt')
        ids = inputs['input_ids'].squeeze(0).tolist()
        token_len = len(ids)
        input_ids = ids + [0] * (self.seq_length - token_len)
        input_ids = torch.tensor(input_ids).view(self.seq_length)
        hidden_states = self.embed(input_ids)

        image_attention_mask = inputs["image_attention_mask"]
        img_embeds = inputs['input_image_embeds']
        img_sizes = inputs['image_sizes']
        img_set_tensor = self.visual(img_embeds, image_attention_mask, img_sizes)
        merged_img_set_tensor = torch.cat(img_set_tensor, dim=1).squeeze(0)
        positions_tuple = torch.nonzero(input_ids == 200010, as_tuple=True)
        new_hidden_states = hidden_states.squeeze(0).index_put(
            indices=positions_tuple,
            values=merged_img_set_tensor,
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
            k[:, :, token_len:, :] = 0
            v[:, :, token_len:, :] = 0
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
            attention_mask[:, :, :, token_len-1:self.seq_length] = -10000.0
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
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rotary = config.rotary
        ModelMapper.do_map(self, attn, config.model_map['attention'])
        if hasattr(self, 'qkv_proj') and self.qkv_proj is not None:
            # split qkv linear to q, k, v
            split_sizes = [self.hidden_size] * 3
            if self.qkv_proj.weight.shape[0] != self.hidden_size * 3:
                # M/GQA
                split_sizes = [
                    self.num_heads * self.head_dim,           # q_size
                    self.num_key_value_heads * self.head_dim, # k_size
                    self.num_key_value_heads * self.head_dim  # v_size
                ]
            self.q_proj = torch.nn.Linear(self.hidden_size, split_sizes[0])
            self.k_proj = torch.nn.Linear(self.hidden_size, split_sizes[1])
            self.v_proj = torch.nn.Linear(self.hidden_size, split_sizes[2])
            if config.model_type == 'chatglm':
                # chatglm-6b
                qkv_weight = self.qkv_proj.weight.data.view(self.num_heads, 3, self.head_dim, self.hidden_size)
                self.q_proj.weight.data = qkv_weight[:, 0, :, :].reshape(self.hidden_size, self.hidden_size)
                self.k_proj.weight.data = qkv_weight[:, 1, :, :].reshape(self.hidden_size, self.hidden_size)
                self.v_proj.weight.data = qkv_weight[:, 2, :, :].reshape(self.hidden_size, self.hidden_size)
                qkv_bias = self.qkv_proj.bias.data.view(self.num_heads, 3, self.head_dim)
                self.q_proj.bias.data = qkv_bias[:, 0, :].reshape(self.hidden_size)
                self.k_proj.bias.data = qkv_bias[:, 1, :].reshape(self.hidden_size)
                self.v_proj.bias.data = qkv_bias[:, 2, :].reshape(self.hidden_size)
            else:
                # other
                qw, kw, vw = torch.split(self.qkv_proj.weight, split_sizes)
                self.q_proj.weight.data = qw
                self.k_proj.weight.data = kw
                self.v_proj.weight.data = vw
                if self.qkv_proj.bias is not None:
                    qb, kb, vb = torch.split(self.qkv_proj.bias, split_sizes)
                    self.q_proj.bias.data = qb
                    self.k_proj.bias.data = kb
                    self.v_proj.bias.data = vb
                else:
                    self.q_proj.bias.data = torch.zeros(split_sizes[0]).to(self.dtype)
                    self.k_proj.bias.data = torch.zeros(split_sizes[1]).to(self.dtype)
                    self.v_proj.bias.data = torch.zeros(split_sizes[2]).to(self.dtype) # + 0.0001

            self.q_proj.weight.requires_grad = False
            self.k_proj.weight.requires_grad = False
            self.v_proj.weight.requires_grad = False
            self.q_proj.bias.requires_grad = False
            self.k_proj.bias.requires_grad = False
            self.v_proj.bias.requires_grad = False
            del self.qkv_proj

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
        if self.model_type == 'chatglm':
            self.rotary_dim = config.head_dim // 2
        elif self.model_type == 'phi4mm':
            self.rotary_dim *= config.config.partial_rotary_factor
            self.rotary_dim = int(self.rotary_dim)

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
        if self.model_type == 'chatglm':
            return self.chatglm_rotary_pos(x, cos, sin)
        if self.model_type == 'chatglm2':
            return self.chatglm2_rotary_pos(x, cos, sin)
        if self.model_type == 'phi-msft':
            return self.phi_rotary_pos(x, cos, sin)
        if self.model_type == 'phi4mm':
           return self.phi4mm_rotary_pos(x, cos, sin)
        return self.llama_rotary_pos(x, cos, sin)

    def phi4mm_rotary_pos(self, x, cos, sin):
        x_rot, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        x_embed = torch.cat([(x_rot * cos) + (rotate_half(x_rot) * sin), x_pass], dim=-1)
        return x_embed
    def llama_rotary_pos(self, x, cos, sin):
        x = (x * cos) + (rotate_half(x) * sin)
        return x

    def phi_rotary_pos(self, x, cos, sin):
        x, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        x = (x * cos) + (rotate_half(x) * sin)
        return torch.cat((x, x_pass), dim=-1)

    def chatglm2_rotary_pos(self, x, cos, sin):
        x, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        b, s, n, h = x.shape
        xshaped = x.view(b, s, n, h//2, 2)
        x = torch.concat(
            [
                xshaped[..., 0] * cos - xshaped[..., 1] * sin,
                xshaped[..., 1] * cos + xshaped[..., 0] * sin,
            ],
            -1,
        )
        return torch.cat((x, x_pass), dim=-1)

    def chatglm_rotary_pos(self, x, cos, sin):
        seq = x.shape[1]
        x1, x2 = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        cos1, sin1 = cos[:, :seq, ...], sin[:, :seq, ...]
        cos2, sin2 = cos[:, seq:, ...], sin[:, seq:, ...]
        x1 = (x1 * cos1) + (rotate_half(x1) * sin1)
        x2 = (x2 * cos2) + (rotate_half(x2) * sin2)
        return torch.cat((x1, x2), dim=-1)

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
            # phi
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
        self.lmhead_with_topk = config.lmhead_with_topk

    def forward(self, hidden_states):
        hidden_states = self.final_layernorm(hidden_states)
        m_logits = self.lm(hidden_states)
        if self.lmhead_with_topk:
            _, token = torch.topk(m_logits.float(), 1)
            return token
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

class Phi4mmVisionEmbedding(torch.nn.Module):
    def __init__(self, phi4mm_image_embed, base):
        super().__init__()
        self.base = base
        self.phi4mm_image_embed = phi4mm_image_embed
        self.image_dim_out = self.phi4mm_image_embed.image_dim_out
        self.img_sizes = self.phi4mm_image_embed.img_sizes
        self.image_attention_mask = self.phi4mm_image_embed.image_attention_mask
        self.num_img_tokens = self.phi4mm_image_embed.num_img_tokens
        self.base_feat_height_target = self.phi4mm_image_embed.base_feat_height_target
        
        # global_gn and sub_gn for hd transform, serves as line separator
        self.use_hd_transform = self.phi4mm_image_embed.use_hd_transform
        self.with_learnable_separator = self.phi4mm_image_embed.with_learnable_separator
        self.hd_transform_order = self.phi4mm_image_embed.hd_transform_order
        self.freeze_img_processor = self.phi4mm_image_embed.freeze_img_processor
        self.crop_size = self.phi4mm_image_embed.crop_size
        
        # image token compression
        self.image_token_compression_cls = self.phi4mm_image_embed.image_token_compression_cls
        if self.image_token_compression_cls == 'avg_pool_2d':
            self.image_token_compression = self.phi4mm_image_embed.image_token_compression
            self.base_feat_height_reduction = self.phi4mm_image_embed.base_feat_height_reduction
            self.base_feat_height_target = self.phi4mm_image_embed.base_feat_height_target
        elif self.image_token_compression_cls is None:
            self.image_token_compression = None
            self.base_feat_height_reduction = 2
        else:
            raise NotImplementedError(f'image_token_compression_cls = {self.image_token_compression_cls}, not implemented')
        if self.with_learnable_separator:
            assert self.use_hd_transform, 'learnable separator is only for hd transform'
            # 1024 * 4, merge spatial to channel dimension
            self.glb_GN = self.phi4mm_image_embed.glb_GN
            self.sub_GN = self.phi4mm_image_embed.sub_GN

    def forward(self, image_tensor):
        dtype = self.base.dtype
        image_tensor = image_tensor.to(dtype)
        img_features = self.phi4mm_image_embed.get_img_features(image_tensor)
        img_feature_proj = self.phi4mm_image_embed.img_projection(img_features)
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
    def forward_origin(self, image_tensor, image_attention_mask, img_sizes):
        bs = 1
        dtype = self.phi4mm_image_embed.img_processor.embeddings.position_embedding.weight.dtype
        image_tensor = image_tensor.to(dtype)
        img_features = self.phi4mm_image_embed.get_img_features(image_tensor.flatten(0,1), image_attention_mask.type(torch.BoolTensor).flatten(0,1))
        base_feat_height_target = self.base_feat_height_target
        base_resolution = self.crop_size
        base_feat_height_reduction = self.base_feat_height_reduction
        base_feat_height = base_feat_width = int(np.sqrt(img_features.shape[1]))
        assert base_feat_height == base_feat_height_target and base_feat_width == base_feat_height_target, f'base_feat_height: {base_feat_height}, base_feat_width: {base_feat_width}, expect {base_feat_height_target} features for hd transform'
        # bs x max_num_crops x (24x24) x C
        img_features = img_features.view(bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
        C = self.image_dim_out
        H = base_feat_height

        output_imgs = []
        output_len = []
        # training is tensor, inference is list
        if isinstance(img_sizes, torch.Tensor):
            img_sizes = img_sizes.view(-1, 2)
        
        for _bs in range(bs):
            h, w = img_sizes[_bs]
            h = h // base_resolution
            w = w // base_resolution
            B_ = h * w
            # 1 x (24x24) x 1024
            global_img_feature = img_features[_bs, :1]

            # 1 x 12 x 12 x 4096
            glb_img = global_img_feature.reshape(1,H,H,C).reshape(1,H//base_feat_height_reduction,base_feat_height_reduction,H//base_feat_height_reduction,base_feat_height_reduction,C).contiguous().permute(0,1,3,2,4,5).reshape(1,H//base_feat_height_reduction,H//base_feat_height_reduction,base_feat_height_reduction*base_feat_height_reduction*C).contiguous()
            temp_glb_GN = self.sub_GN.repeat(1, H//base_feat_height_reduction, 1, 1)
            # 1 x 156 x 4096
            glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1,-1,base_feat_height_reduction*base_feat_height_reduction*C)

            # (max_num_crops-1) x (12x12) x C
            sub_img = img_features[_bs, 1:]
            
            # 16x574x1024
            # get rid of padding sub_img
            sub_img = sub_img[:B_]

            # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
            sub_img = sub_img.reshape(B_,H,H,C).reshape(B_,H//base_feat_height_reduction,base_feat_height_reduction,H//base_feat_height_reduction,base_feat_height_reduction,C).contiguous().permute(0,1,3,2,4,5).reshape(B_,-1,base_feat_height_reduction*base_feat_height_reduction*C).contiguous()
            sub_img = sub_img.reshape(1, h, w, base_feat_height // base_feat_height_reduction, base_feat_width // base_feat_height_reduction, -1).permute(0,1,3,2,4,5).reshape(1,h*base_feat_height//base_feat_height_reduction,w*base_feat_width//base_feat_height_reduction,base_feat_height_reduction*base_feat_height_reduction*C)
            if image_attention_mask is not None and len(image_attention_mask) > 0:
                reshaped_image_attention_mask = image_attention_mask[_bs,1:B_+1,0::2,0::2].reshape(1, h, w, base_feat_height // base_feat_height_reduction, base_feat_width // base_feat_height_reduction).permute(0,1,3,2,4).reshape(1,h*base_feat_height//base_feat_height_reduction,w*base_feat_width//base_feat_height_reduction)
                useful_height = int(reshaped_image_attention_mask[0,:,0].sum().item())
                useful_width = int(reshaped_image_attention_mask[0,0,:].sum().item())
                sub_img = sub_img[:,:useful_height, :useful_width]
                temp_sub_GN = self.sub_GN.repeat(1, useful_height, 1, 1)
                temp_len = int(image_attention_mask[_bs,:B_+1,0::2,0::2].sum().item()) + (useful_height+1) + base_feat_height//base_feat_height_reduction
            else:
                temp_sub_GN = self.sub_GN.repeat(1, h*base_feat_height//base_feat_height_reduction, 1, 1)
                temp_len = int((h*w+1)*self.num_img_tokens+ 1 + (h+1)*base_feat_height//base_feat_height_reduction)

            sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1,-1,base_feat_height_reduction*base_feat_height_reduction*C)
            # (1, num_img_tokens, 1024*4)

            # glb + sub
            if self.hd_transform_order == 'glb_sub':
                output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
            elif self.hd_transform_order == 'sub_glb':
                output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
            else:
                raise NotImplementedError(f'hd_transform_order = {self.hd_transform_order}, not implemented')

            #temp_len = int((h*w+1)*144 + 1 + (h+1)*12)
            assert temp_len == output_imgs[-1].shape[1], f'temp_len: {temp_len}, output_imgs[-1].shape[1]: {output_imgs[-1].shape[1]}'
            output_len.append(temp_len)

        img_set_tensor = []
        for _output_img in output_imgs:
            print(_output_img.shape)
            img_feature_proj = self.phi4mm_image_embed.img_projection(_output_img)
            print(img_feature_proj.shape)
            img_set_tensor.append(img_feature_proj)

        return img_set_tensor

    def export_origin(self, onnx_path):
        img_processor = AutoImageProcessor.from_pretrained(self.base.model_path, trust_remote_code=True)
        image = Image.open("australia.jpg")
        inputs = img_processor.preprocess(image)
        image_attention_mask = inputs["image_attention_mask"]
        img_embeds = inputs['input_image_embeds']
        img_sizes = inputs['image_sizes']
        print(img_embeds.dtype, image_attention_mask.dtype, img_sizes.dtype)
        model = copy.deepcopy(self).float()
        torch.onnx.export(
            model, (img_embeds, image_attention_mask, img_sizes),
            onnx_path,
            verbose=False,
            input_names=['img_embeds', 'image_attention_mask', 'img_sizes'],
            output_names=['hidden_states'],
            do_constant_folding=True,
            opset_version=17
        )
        del model
        return

class Phi4mmSpeechEmbedding(torch.nn.Module):
    def __init__(self, phi4mm_audio_embed, base):
        super().__init__()
        self.base = base
        self.phi4mm_audio_embed = phi4mm_audio_embed
    def forward(self, input_embeds):
        dtype = self.base.dtype
        input_embeds = input_embeds.to(dtype)
        audio_feature_proj = self.phi4mm_audio_embed.get_audio_features(input_embeds, None)
        return audio_feature_proj

    def export(self, onnx_path):
        audio_embeds = torch.rand(1, 384, 80)
        model = self.float().eval()
        torch.onnx.export(
            model, (audio_embeds),
            onnx_path,
            verbose=False,
            input_names=['audio_embeds'],
            output_names=['hidden_states'],
            do_constant_folding=True,
            opset_version=17
        )
        del model
        return
