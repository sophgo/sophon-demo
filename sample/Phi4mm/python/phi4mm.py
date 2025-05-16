import time
import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from transformers import AutoProcessor, AutoFeatureExtractor
from torchvision.transforms.functional import InterpolationMode
import os
import soundfile
import numpy as np
import sophon.sail as sail
sail.set_loglevel(sail.LogLevel.ERROR)

# Preprocess the images
IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    if max_num <= 0:
        return [image.resize((image_size, image_size))]
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, max_num=max_num, image_size=input_size, use_thumbnail=False)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class Phi4():
    def __init__(self, args):
        self.version = "1.0.0"
        # devid
        self.dev_ids = [args.devid]
        self.handles = {dev: sail.Handle(dev) for dev in self.dev_ids}

        # load tokenizer
        print("Load " + args.processor + " ...")
        self.processor = AutoProcessor.from_pretrained(
            args.processor, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

        # load model
        start_time = time.time()
        self.model = sail.EngineLLM(args.model_path, self.dev_ids)
        self.graph_names = self.model.get_graph_names()
        load_model_time = time.time() - start_time
        print(f"sail.EngineLLM init cost: {load_model_time:.3f} s")
        
        # initialize parameters
        self.EOS = [self.tokenizer.convert_tokens_to_ids("<|end|>"), self.tokenizer.convert_tokens_to_ids("<|im_end|>")]
        self.token_length = 0

        self.target = sail.Handle(self.dev_ids[0]).get_target()
        self.tensors = {}
        start_time = time.time()
        self.io_alone = 0
        if self.target in ["BM1688", "CV186AH"]:
            for net in self.graph_names:
                self.tensors[net] = {}
                self.tensors[net]["addr_mode"] = self.model.get_addr_mode(net)
                if self.tensors[net]["addr_mode"] == 0:
                    self.tensors[net]['input'] = self.model.create_max_input_tensors(net)
                    self.tensors[net]['output'] = self.model.create_max_output_tensors(net)
                elif self.tensors[net]["addr_mode"] == 1:
                    self.io_alone = 1
                    self.tensors[net]['input'] = self.model.get_input_tensors(net)
                    self.tensors[net]['output'] = self.model.get_output_tensors(net)
        else:
            for net in self.graph_names:
                self.tensors[net] = {}
                self.tensors[net]["addr_mode"] = self.model.get_addr_mode(net)
                if self.tensors[net]["addr_mode"] == 0:
                    self.tensors[net]['input'] = self.model.get_input_tensors_addrmode0(net)
                    self.tensors[net]['output'] = self.model.get_output_tensors_addrmode0(net)
                elif self.tensors[net]["addr_mode"] == 1:
                    self.io_alone = 1
                    self.tensors[net]['input'] = self.model.get_input_tensors(net)
                    self.tensors[net]['output'] = self.model.get_output_tensors(net)
        init_tensor_time = time.time() - start_time
        print(f"io tensors init cost: {init_tensor_time:.3f} s")

        # initialize params
        self.is_dynamic = self.model.get_is_dynamic("block_0")
        print("dynamic: ", self.is_dynamic)
        self.token_length = 0
        _, self.SEQLEN, self.HIDDEN_SIZE = self.tensors["block_0"]["input"][0].shape()
        _, _, self.ATTEN_HEAD, self.ATTEN_DIM = self.tensors["block_cache_0"]["input"][3].shape()

        self.ATTENTION_MASK = -10000.0
        if self.tensors["block_0"]["input"][2].dtype() == sail.Dtype.BM_BFLOAT16:
            self.ATTENTION_MASK = 50716

        self.is_sample = False
        if ("greedy_head" in self.graph_names):
            self.is_sample = True
        self.NUM_LAYERS = 32
        self.token_length = 0

        # initialize net name
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_lm = "lm_head"
        self.greedy = "greedy_head"
        self.penalty = "penalty_sample_head"
        self.name_vit = "phi4mm_vit"
        self.name_speech = "phi4mm_speech"
        self.past_k = {}
        self.past_v = {}
        # not io_alone 
        if self.io_alone == 0 or self.is_dynamic:
            print("no io_alone")
            for j in range(self.NUM_LAYERS):
                self.past_k[j] = {}
                self.past_v[j] = {}
                for i in range(len(self.dev_ids)):
                    self.past_k[j][i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 3])
                    self.past_v[j][i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 4])
        else:
            for j in range(self.NUM_LAYERS):
                self.past_k[j] = {}
                self.past_v[j] = {}
                for i in range(len(self.dev_ids)):
                    self.past_k[j][i] = self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 3]
                    self.past_v[j][i] = self.tensors[self.name_blocks_cache[j]]["input"][5 * i + 4]
    

        self.first_embed_input = self.model.create_max_input_tensors(self.name_embed)
        self.first_hidden_state = self.model.create_max_output_tensors(self.name_embed)
        self.next_embed_input = self.model.create_max_input_tensors(self.name_embed_cache)
        self.next_hidden_state = self.model.create_max_output_tensors(self.name_embed_cache)
        self.first_pid = {}
        self.next_pid = {}
        self.first_attention_mask = {}
        self.next_attention_mask = {}
        self.lm_input = self.model.create_max_input_tensors(self.name_lm)
        self.lm_output = self.model.create_max_output_tensors(self.name_lm)
        for i in range(len(self.dev_ids)):
            self.first_pid[i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks[0]]["input"][1])
            self.first_attention_mask[i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks[0]]["input"][2])
            self.next_pid[i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks_cache[0]]["input"][1])
            self.next_attention_mask[i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks_cache[0]]["input"][2])
    
    def init_input_tensor(self, dev_id, net, index):
        shape = self.model.get_input_shape(net, index)
        type = self.model.get_input_dtype(net, index)
        return sail.Tensor(self.handles[dev_id], shape, type, False, True) 
    
    def init_output_tensor(self, dev_id, net, index):
        shape = self.model.get_output_shape(net, index)
        type = self.model.get_output_dtype(net, index)
        return sail.Tensor(self.handles[dev_id], shape, type, False, True)
    
    def init_tensor(self, dev_id, shape, type):
        return sail.Tensor(self.handles[dev_id], shape, type, False, True) 
    
    def init_tensor(self, dev_id, tensor):
        return sail.Tensor(self.handles[dev_id], tensor.shape(), tensor.dtype(), False, True) 
    
    def type_convert(self, sail_dtype):
        if sail_dtype == sail.Dtype.BM_FLOAT32:
            return np.float32
        if sail_dtype == sail.Dtype.BM_FLOAT16:
            return np.float16
        if sail_dtype == sail.Dtype.BM_INT32:
            return np.int32
        if sail_dtype == sail.Dtype.BM_BFLOAT16: 
            return np.uint16
    
    def get_first_input(self, length, token):
        input_ids = np.zeros(length, self.type_convert(self.tensors[self.name_embed]["input"][0].dtype()))
        input_ids[:len(token)] = token

        position_id = np.zeros(length, self.type_convert(self.tensors[self.name_blocks[0]]["input"][1].dtype()))
        for i in range(self.token_length):
            position_id[i] = i

        attention_mask = np.ones(length*length, self.type_convert(self.tensors[self.name_blocks[0]]["input"][2].dtype())) * self.ATTENTION_MASK
        for i in range(len(token)):
            for j in range(length):
                if (j <= i):
                    attention_mask[i*length + j] = 0

        return input_ids, position_id, attention_mask
    
    def forward_first(self, tokens, pixel_values, img_offset, audio_embeds, audio_offset):
        self.token_length = len(tokens)
        if self.token_length > self.SEQLEN:
            print("warining, input seq len too large")
        length = self.token_length + 1 if self.is_dynamic else self.SEQLEN
        input_ids, position_id, attention_mask = self.get_first_input(length, tokens)
        
        for i in range(len(self.dev_ids)):
            self.tensors[self.name_embed]["input"][i] = sail.Tensor(self.first_embed_input[i], [1, length], 0)
            self.tensors[self.name_embed]["output"][i] = sail.Tensor(self.first_hidden_state[i], [1, length, self.HIDDEN_SIZE], 0)
            self.tensors[self.name_embed]["input"][i].update_data(input_ids.reshape(self.tensors[self.name_embed]["input"][i].shape()))
        self.model.process(self.name_embed, self.tensors[self.name_embed]["input"], self.tensors[self.name_embed]["output"])
        
        # ViT Inference
        if pixel_values is not None:
            for i in range(pixel_values.shape[0]):
                self.vit_input = self.tensors[self.name_vit]["input"][0]
                self.vit_output = self.tensors[self.name_vit]["output"][0]
                assert self.vit_input.shape()[0] == 1, "vit only support bs=1"
                if img_offset > 0 and pixel_values[i].numel() == np.prod(self.vit_input.shape()):
                    self.vit_input.update_data(np.expand_dims(pixel_values[i], axis=0))
                    self.vit_input.sync_s2d()
                    input_vit_tensors = {0: self.vit_input}
                    output_vit_tensors = {0: self.vit_output}
                    self.model.process(self.name_vit, input_vit_tensors, output_vit_tensors)
                    self.tensors[self.name_embed]["output"][0].sync_d2d(self.vit_output, 0, int((img_offset + i * self.vit_output.shape()[1]) * self.HIDDEN_SIZE), np.prod(self.vit_output.shape()))
                else:
                    print("No image found or invalid vit data, skip vit inference.")
                    
        # Speech Inference
        if audio_embeds is not None:
            for i in range(audio_embeds.shape[0]):
                self.speech_input = self.tensors[self.name_speech]["input"][0]
                self.speech_output = self.tensors[self.name_speech]["output"][0]
                assert self.speech_input.shape()[0] == 1, "speech only support bs=1"
                if audio_offset > 0 and audio_embeds[i].numel() == np.prod(self.speech_input.shape()):
                    self.speech_input.update_data(np.expand_dims(audio_embeds[i], axis=0))
                    self.speech_input.sync_s2d()
                    input_speech_tensors = {0: self.speech_input}
                    output_speech_tensors = {0: self.speech_output}
                    self.model.process(self.name_speech, input_speech_tensors, output_speech_tensors)
                    audio_data_size = self.audio_embeds_sizes[i] * self.HIDDEN_SIZE
                    self.tensors[self.name_embed]["output"][0].sync_d2d(self.speech_output, 0, int((audio_offset + i * self.speech_output.shape()[1]) * self.HIDDEN_SIZE), audio_data_size)
                else:
                    print("No speech found or invalid speech data, skip speech inference.")
        # blocks
        for i in range(len(self.dev_ids)):
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1] = sail.Tensor(self.first_pid[i], [1, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 2] = sail.Tensor(self.first_attention_mask[i], [1, 1, length, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1].update_data(position_id.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 1].shape()))
            self.tensors[self.name_blocks[0]]["input"][3 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 2].shape()).view(self.type_convert(self.tensors[self.name_blocks[0]]["input"][3 * i + 2].dtype())))
        for i in range(self.NUM_LAYERS):
            for j in range(len(self.dev_ids)):
                self.tensors[self.name_blocks[i]]["input"][3 * j] = sail.Tensor(self.first_hidden_state[j], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j] = sail.Tensor(self.first_hidden_state[j], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j + 1] = sail.Tensor(self.past_k[i][j], [1, length, self.ATTEN_HEAD, self.ATTEN_DIM], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j + 2] = sail.Tensor(self.past_v[i][j], [1, length, self.ATTEN_HEAD, self.ATTEN_DIM], 0)
            if i > 0:
                for j in range(len(self.dev_ids)):
                    self.tensors[self.name_blocks[i]]["input"][3 * j + 1] = self.tensors[self.name_blocks[0]]["input"][3 * j + 1]
                    self.tensors[self.name_blocks[i]]["input"][3 * j + 2] = self.tensors[self.name_blocks[0]]["input"][3 * j + 2]
            self.model.process(self.name_blocks[i], self.tensors[self.name_blocks[i]]["input"], self.tensors[self.name_blocks[i]]["output"])
            
        # lm_head
        self.tensors[self.name_lm]["input"][0] = sail.Tensor(self.first_hidden_state[0], [1, 1, self.HIDDEN_SIZE], (self.token_length - 1) * self.HIDDEN_SIZE)
        self.tensors[self.name_lm]["output"][0] = self.lm_output[0]
        
        self.model.process(self.name_lm, self.tensors[self.name_lm]["input"], self.tensors[self.name_lm]["output"])
        if not self.is_sample:
            return int(np.squeeze(self.tensors[self.name_lm]["output"][0].asnumpy()))

        # sample
        self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
        self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])

        return int(self.tensors[self.greedy]["output"][0].asnumpy().squeeze())
 
    def forward_next(self):
        self.token_length += 1
        position_id = np.array(self.token_length - 1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][1].dtype()))
        attention_mask = np.zeros(self.SEQLEN+1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][2].dtype()))
        for i in range(self.token_length - 1, self.SEQLEN):
            attention_mask[i] = self.ATTENTION_MASK

        # embedding_cache
        if len(self.dev_ids) > 1:
            # breakpoint()
            input_ids = np.array(int(self.tensors[self.name_lm]["output"][0].asnumpy()), self.type_convert(self.tensors[self.name_embed_cache]["input"][0].dtype()))
            for i in range(len(self.dev_ids)):
                self.next_embed_input[i].update_data(input_ids.reshape(self.tensors[self.name_embed_cache]["input"][i].shape()))
                self.tensors[self.name_embed_cache]["input"][i] = self.next_embed_input[i]
        else:
            self.tensors[self.name_embed_cache]["input"][0] = self.tensors[self.name_lm]["output"][0]
            if self.is_sample:
                self.tensors[self.name_embed_cache]["input"][0] = self.tensors[self.greedy]["output"][0]

        for i in range(len(self.dev_ids)):
            self.tensors[self.name_embed_cache]["output"][i] = self.next_hidden_state[i] 

        self.model.process(self.name_embed_cache, self.tensors[self.name_embed_cache]["input"], self.tensors[self.name_embed_cache]["output"])

        # block_cache
        for i in range(len(self.dev_ids)):
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1] = self.next_pid[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2] = self.next_attention_mask[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].update_data(position_id.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].shape()))
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].shape()).view(self.type_convert(self.tensors[self.name_blocks[0]]["input"][3 * i + 2].dtype())))


        for i in range(self.NUM_LAYERS):
            for j in range(len(self.dev_ids)):
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j] = self.next_hidden_state[j]
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j] = self.next_hidden_state[j]
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 3] = self.past_k[i][j]
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 4] = self.past_v[i][j]
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j + 1] = sail.Tensor(self.past_k[i][j], [1, 1, self.ATTEN_HEAD, self.ATTEN_DIM], (self.token_length-1) * (self.ATTEN_HEAD * self.ATTEN_DIM))
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j + 2] = sail.Tensor(self.past_v[i][j], [1, 1, self.ATTEN_HEAD, self.ATTEN_DIM], (self.token_length-1) * (self.ATTEN_HEAD * self.ATTEN_DIM))
            if i > 0:
                for j in range(len(self.dev_ids)):
                    self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 1] = self.tensors[self.name_blocks_cache[0]]["input"][5 * j + 1]
                    self.tensors[self.name_blocks_cache[i]]["input"][5 * j + 2] = self.tensors[self.name_blocks_cache[0]]["input"][5 * j + 2]
            self.model.process(self.name_blocks_cache[i], self.tensors[self.name_blocks_cache[i]]["input"], self.tensors[self.name_blocks_cache[i]]["output"])
        
        #lm_head
        self.tensors[self.name_lm]["input"][0] = self.next_hidden_state[0]
        self.tensors[self.name_lm]["output"][0] = self.lm_output[0]
        self.model.process(self.name_lm, self.tensors[self.name_lm]["input"], self.tensors[self.name_lm]["output"])
        if not self.is_sample:
            return int(self.tensors[self.name_lm]["output"][0].asnumpy())

        # sample
        self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
        self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])

        return int(self.tensors[self.greedy]["output"][0].asnumpy().squeeze())

    def chunk_and_pad(self, input_tensor, chunk_size=384):
        """
        input_tensor: shape (1, N, 80)
        return: shape (M, 384, 80)
        """
        N = input_tensor.shape[1]
        M = (N + chunk_size - 1) // chunk_size  # 向上取整
        batches = []
        for i in range(M):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, N)
            chunk = input_tensor[:, start:end, :]  # shape: (1, chunk_len, 80)
            # padding if needed
            chunk_len = end - start
            if chunk_len < chunk_size:
                # pad on dim=1 (sequence length)
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, chunk_size - chunk_len))
            batches.append(chunk)
        # 现在 batches 是 M 个 (1, 384, 80)，拼接成 (M, 384, 80)
        output = torch.cat(batches, dim=0)
        return output
    
    def encode(self):
        self.prefix_offset = 0
        self.pixel_values = None
        self.image_offset = 0
        self.audio_offset = 0
        self.audio_embeds = None
        prefix_tokens = self.tokenizer.encode('<|user|>')
        self.prefix_offset = len(prefix_tokens)
        postfix_tokens = self.tokenizer.encode('<|end|><|assistant|>\n')
        text_tokens = []
        image_tokens = []
        audio_tokens = []
        if self.input_str:
            text_tokens = self.tokenizer.encode(self.input_str)
        if self.image_str:
            max_num = self.SEQLEN / self.tensors[self.name_vit]["output"][0].shape()[1] - 1 # 512 / 256 - 1 = 1
            self.pixel_values = load_image(self.image_str, max_num=int(max_num)) # 1,3,448,448
            image_tokens = [200010] * self.tensors[self.name_vit]["output"][0].shape()[1] * self.pixel_values.size(0)
            self.image_offset = self.tensors[self.name_vit]["output"][0].shape()[1]
        if self.audio_str:
            audio = soundfile.read(self.audio_str)
            audio_inputs = self.processor.audio_processor([audio], return_tensors='pt')
            self.audio_embeds = audio_inputs['input_audio_embeds']
            self.audio_embeds = self.chunk_and_pad(self.audio_embeds, self.tensors[self.name_speech]["input"][0].shape()[1])
            audio_embeds_size = audio_inputs['audio_embed_sizes'].tolist()[0]
            self.audio_embeds_sizes = []
            while audio_embeds_size > self.tensors[self.name_speech]["output"][0].shape()[1]:
                self.audio_embeds_sizes.append(self.tensors[self.name_speech]["output"][0].shape()[1])
                audio_embeds_size -= self.tensors[self.name_speech]["output"][0].shape()[1]
            self.audio_embeds_sizes.append(audio_embeds_size)
            self.audio_offset = self.prefix_offset + self.image_offset
            audio_tokens = [200011] * self.tensors[self.name_speech]["output"][0].shape()[1] * self.audio_embeds.size(0)
        self.input_ids = prefix_tokens + image_tokens + audio_tokens + text_tokens + postfix_tokens
        
    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
================================================================="""
        )
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break
            self.image_str = input("\nImage Path: ")
            if self.image_str:
                if not os.path.exists(self.image_str):
                    print("Can't find image: {}".format(self.image_str))
                    continue
            self.audio_str = input("\nAudio Path: ")
            if self.audio_str:
                if not os.path.exists(self.audio_str):
                    print("Can't find audio: {}".format(self.audio_str))
                    continue
            self.encode()
            # Chat
            first_start = time.time()
            token = self.forward_first(
                self.input_ids, self.pixel_values, self.prefix_offset, self.audio_embeds, self.audio_offset)
            first_end = time.time()
            tok_num = 1
            # Following tokens
            full_word_tokens = []
            print("\nAnswer:")
            while token not in self.EOS and self.token_length < self.SEQLEN:
                full_word_tokens.append(token)
                word = self.tokenizer.decode(
                    full_word_tokens, skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode([token, token], skip_special_tokens=True)[
                            len(pre_word):]
                    print(word, flush=True, end="")
                    full_word_tokens = []
                tok_num += 1
                token = self.forward_next()
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")

def main(args):
    model = Phi4(args)
    model.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str,
                        required=True, help='path to the bmodel file')
    parser.add_argument('-t', '--processor', type=str,
                        default="./processor", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=int,
                        default=0, help='device ID to use')
    args = parser.parse_args()
    main(args)
