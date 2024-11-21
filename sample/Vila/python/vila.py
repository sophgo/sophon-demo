#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

from transformers import SiglipImageProcessor, AutoTokenizer
from PIL import Image
import numpy as np
import cv2
import sophon.sail as sail
import time
import argparse

def opencv_extract_frames(video_file, num_frames):
    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    frame_interval = frame_count // num_frames
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    images = []
    count = 0
    success = True
    while success:
        # print("frame_count:", frame_count, "count:", count, "num_frames:", num_frames, "frame_interval:", frame_interval)
        if frame_count >= num_frames:
            # breakpoint()
            # vidcap.set(cv2.CAP_PROP_POS_FRAMES, 200)
            success, frame = vidcap.read()
            if count in frame_indices:
                try:
                    # breakpoint()
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    images.append(im_pil)
                except BaseException:
                    continue
                if len(images) >= num_frames:
                    return images, num_frames
            count += 1
        else:
            # Left padding frames if the video is not long enough
            success, frame = vidcap.read()
            if success:
                try:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    images.append(im_pil)
                except BaseException:
                    continue
                count += 1
            else:
                break

def process_images(images, image_processor):
    new_images = []
    for image in images:
        image = image.convert("RGB")
        crop_size = image_processor.size
        image = image.resize((crop_size["height"], crop_size["width"]))
        image = image_processor.preprocess(image, return_tensors="np")["pixel_values"][0]
        new_images.append(image)
    return np.stack(new_images)

def tokenizer_image_token(prompt, tokenizer, image_token_index=-200, lstrip=False):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if lstrip:
        offset = 1
    else:
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

    for chunk_id, x in enumerate(insert_separator(prompt_chunks, [image_token_index] * (offset + 1))):
        if chunk_id == 0 and lstrip:
            input_ids.extend(x)
        else:
            input_ids.extend(x[offset:])
    return input_ids

class Vila:
    def __init__(self, video_path, llm_bmodel_path, vision_bmodel_path, dev_id) -> None:
        self.model = sail.EngineLLM(llm_bmodel_path, [dev_id])
        self.vision_model = sail.EngineLLM(vision_bmodel_path, [dev_id])
        self.handle = sail.Handle(dev_id)
        image_processor = SiglipImageProcessor.from_pretrained("./python/config/image_processer")
        self.tokenizer = AutoTokenizer.from_pretrained("./python/config/llm_token")
        self.graph_names = self.model.get_graph_names()
        self.NUM_LAYERS = (len(self.graph_names) - 3) // 2
        self.input_tensors = {}
        self.output_tensors = {}
        self.name_vision_embed = "vision_embedding"
        self.name_llm_embed = "embedding"
        self.name_llm_embed_cache = "embedding_cache" 
        self.name_lm = "lm_head"
        self.name_block = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_block_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.input_tensors[self.name_vision_embed] = self.vision_model.create_max_input_tensors(self.name_vision_embed)
        self.output_tensors[self.name_vision_embed] = self.vision_model.create_max_output_tensors(self.name_vision_embed)
        self.num_frames, self.vision_token_length, _ = self.output_tensors[self.name_vision_embed][0].shape()
        start = time.time()
        images, _ = opencv_extract_frames(video_path, self.num_frames)
        end = time.time()
        print(f"opencv_extract_frames cost: {end -  start}")
        images_np = process_images(images, image_processor).astype(np.float16)
        self.input_tensors[self.name_vision_embed][0].update_data(images_np.view(np.uint16))
        vision_start = time.time()
        self.vision_model.process(self.name_vision_embed, self.input_tensors[self.name_vision_embed], self.output_tensors[self.name_vision_embed])
        vision_end = time.time()
        print(f"vision_embedding: {vision_end -  vision_start}")

        _, self.SEQ_LEN, self.HIDDEN_SIZE = self.model.get_input_shape(self.name_block[0], 0)
        _, _, self.NUM_HEADS, self.HEAD_DIM = self.model.get_output_shape(self.name_block[0], 1)

        hidden_states, position_ids, attention_mask = self.model.create_max_input_tensors(self.name_block[0]).values()
        self.io_alone = 1
        if self.io_alone == 0:
            past_k = [self.init_input_tensor(self.name_block_cache[i], 3) for i in range(self.NUM_LAYERS)]
            past_v= [self.init_input_tensor(self.name_block_cache[i], 4) for i in range(self.NUM_LAYERS)]
        else:
            past_k = [self.model.get_input_tensors(self.name_block_cache[i])[3] for i in range(self.NUM_LAYERS)]
            past_v = [self.model.get_input_tensors(self.name_block_cache[i])[4] for i in range(self.NUM_LAYERS)]


        self.input_tensors[self.name_llm_embed] = self.model.create_max_input_tensors(self.name_llm_embed)
        self.output_tensors[self.name_llm_embed] = self.model.create_max_output_tensors(self.name_llm_embed)
        for i in range(self.NUM_LAYERS):
            self.input_tensors[self.name_block[i]] = {}
            self.output_tensors[self.name_block[i]] = {}
            self.input_tensors[self.name_block[i]][0] = hidden_states
            self.input_tensors[self.name_block[i]][1] = position_ids
            self.input_tensors[self.name_block[i]][2] = attention_mask
            self.output_tensors[self.name_block[i]][0] = hidden_states
            self.output_tensors[self.name_block[i]][1] = past_k[i]
            self.output_tensors[self.name_block[i]][2] = past_v[i]

        self.input_tensors[self.name_lm] = self.model.create_max_input_tensors(self.name_lm)
        self.output_tensors[self.name_lm] = self.model.create_max_output_tensors(self.name_lm)

        self.input_tensors[self.name_llm_embed_cache] = self.output_tensors[self.name_lm]
        self.output_tensors[self.name_llm_embed_cache] = self.model.create_max_output_tensors(self.name_llm_embed_cache)
        position_ids_next = self.init_input_tensor(self.name_block_cache[i], 1)
        attention_mask_next = self.init_input_tensor(self.name_block_cache[i], 2)
        past_k_cache = [self.init_output_tensor(self.name_block_cache[i], 1) for i in range(self.NUM_LAYERS)]
        past_v_cache = [self.init_output_tensor(self.name_block_cache[i], 2) for i in range(self.NUM_LAYERS)]

        for i in range(self.NUM_LAYERS):
            self.input_tensors[self.name_block_cache[i]] = {}
            self.output_tensors[self.name_block_cache[i]] = {}
            self.input_tensors[self.name_block_cache[i]][0] = self.output_tensors[self.name_llm_embed_cache][0]
            self.input_tensors[self.name_block_cache[i]][1] = position_ids_next
            self.input_tensors[self.name_block_cache[i]][2] = attention_mask_next
            self.input_tensors[self.name_block_cache[i]][3] = past_k[i]
            self.input_tensors[self.name_block_cache[i]][4] = past_v[i]
            self.output_tensors[self.name_block_cache[i]][0] = self.output_tensors[self.name_llm_embed_cache][0]
            self.output_tensors[self.name_block_cache[i]][1] = past_k_cache[i]
            self.output_tensors[self.name_block_cache[i]][2] = past_v_cache[i]

    def type_convert(self, sail_dtype):
        if sail_dtype == sail.Dtype.BM_FLOAT32:
            return np.float32
        if sail_dtype == sail.Dtype.BM_FLOAT16:
            return np.float16
        if sail_dtype == sail.Dtype.BM_INT32:
            return np.int32
        if sail_dtype == sail.Dtype.BM_BFLOAT16: 
            return np.uint16

    def init_tensor(self, shape, type):
        return sail.Tensor(self.handle, shape, type, False, True) 
    
    def init_input_tensor(self, net, index):
        shape = self.model.get_input_shape(net, index)
        type = self.model.get_input_dtype(net, index)
        return sail.Tensor(self.handle, shape, type, False, True) 
    
    def init_output_tensor(self, net, index):
        shape = self.model.get_output_shape(net, index)
        type = self.model.get_output_dtype(net, index)
        return sail.Tensor(self.handle, shape, type, False, True)
    
    def forward_first(self, token):
        self.token_length = len(token) + (self.vision_token_length - 1) * self.num_frames
        token_np = np.array(token)
        image_index = np.where(token_np == -200)[0]
        image_index = np.append(np.insert(image_index, 0, -1), len(token))
        token_np[token_np==-200] = 0
        input_ids = np.zeros(self.SEQ_LEN, self.type_convert(self.input_tensors[self.name_llm_embed][0].dtype()))
        input_ids[:len(token)] = token_np

        self.input_tensors[self.name_llm_embed][0].update_data(input_ids.reshape(self.input_tensors[self.name_llm_embed][0].shape()))
        self.model.process(self.name_llm_embed, self.input_tensors[self.name_llm_embed], self.output_tensors[self.name_llm_embed])

        #image d2d
        offset = 0
        for i in range(len(image_index) - 1):
            self.input_tensors[self.name_block[0]][0].sync_d2d(
                self.output_tensors[self.name_llm_embed][0], 
                (image_index[i] + 1) * self.HIDDEN_SIZE,
                offset,
                (image_index[i + 1] - image_index[i] - 1) * self.HIDDEN_SIZE)
            offset += (image_index[i + 1] - image_index[i] - 1) * self.HIDDEN_SIZE
            if i < self.num_frames:
                self.input_tensors[self.name_block[0]][0].sync_d2d(
                    self.output_tensors[self.name_vision_embed][0], 
                    i * self.vision_token_length * self.HIDDEN_SIZE,
                    offset,
                    self.vision_token_length * self.HIDDEN_SIZE)
                offset += self.vision_token_length * self.HIDDEN_SIZE


        position_id = np.zeros(self.SEQ_LEN, self.type_convert(self.input_tensors[self.name_block[0]][1].dtype()))
        position_id[:self.token_length] = np.arange(self.token_length)
        attention_mask = np.full((self.SEQ_LEN, self.SEQ_LEN), -10000.0, dtype=self.type_convert(self.input_tensors[self.name_block[0]][2].dtype()))
        for i in range(self.token_length):
            attention_mask[i, :i+1] = 0

        self.input_tensors[self.name_block[0]][1].update_data(position_id.reshape(self.input_tensors[self.name_block[0]][1].shape()))
        self.input_tensors[self.name_block[0]][2].update_data(attention_mask.reshape(self.input_tensors[self.name_block[0]][2].shape()).view(np.uint16))

        for i in range(self.NUM_LAYERS):
            self.model.process(self.name_block[i], self.input_tensors[self.name_block[i]], self.output_tensors[self.name_block[i]])
        self.input_tensors[self.name_lm][0].sync_d2d(self.output_tensors[self.name_block[self.NUM_LAYERS - 1]][0], (self.token_length - 1) * self.HIDDEN_SIZE, 0, self.HIDDEN_SIZE)
        self.model.process(self.name_lm, self.input_tensors[self.name_lm], self.output_tensors[self.name_lm])

        return int(self.output_tensors[self.name_lm][0].asnumpy())


    def forward_next(self,):
        self.token_length += 1
        position_id = np.array(self.token_length - 1, self.type_convert(self.input_tensors[self.name_block_cache[0]][1].dtype()))
        attention_mask = np.zeros(self.SEQ_LEN + 1, self.type_convert(self.input_tensors[self.name_block_cache[0]][2].dtype()))
        for i in range(self.token_length - 1, self.SEQ_LEN):
            attention_mask[i] = -10000.0
        self.model.process(self.name_llm_embed_cache, self.input_tensors[self.name_llm_embed_cache], self.output_tensors[self.name_llm_embed_cache])
        # breakpoint()
        self.input_tensors[self.name_block_cache[0]][1].update_data(position_id.reshape(self.input_tensors[self.name_block_cache[0]][1].shape()))
        self.input_tensors[self.name_block_cache[0]][2].update_data(attention_mask.reshape(self.input_tensors[self.name_block_cache[0]][2].shape()).view(np.uint16))
        for i in range(self.NUM_LAYERS):
            self.model.process(self.name_block_cache[i], self.input_tensors[self.name_block_cache[i]], self.output_tensors[self.name_block_cache[i]])
            self.input_tensors[self.name_block_cache[i]][3].sync_d2d(self.output_tensors[self.name_block_cache[i]][1], 0, (self.token_length - 1) * self.NUM_HEADS * self.HEAD_DIM, self.NUM_HEADS * self.HEAD_DIM)
            self.input_tensors[self.name_block_cache[i]][4].sync_d2d(self.output_tensors[self.name_block_cache[i]][2], 0, (self.token_length - 1) * self.NUM_HEADS * self.HEAD_DIM, self.NUM_HEADS * self.HEAD_DIM)
        self.input_tensors[self.name_lm][0] = self.output_tensors[self.name_block_cache[self.NUM_LAYERS - 1]][0]
        self.model.process(self.name_lm, self.input_tensors[self.name_lm], self.output_tensors[self.name_lm])

        # breakpoint()
        return int(self.output_tensors[self.name_lm][0].asnumpy())


def argsparser():

    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--llm', type=str, default='./models/BM1684X/llama_int4_seq512.bmodel', help='path of llm model')
    parser.add_argument('--vision', type=str, default='./models/BM1684X/vision_embedding_1batch.bmodel', help='path of vision_embedding model')
    parser.add_argument('--video', type=str, default='./demo.mp4', help='path of video')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    vila = Vila(args.video, args.llm, args.vision, args.dev_id)
    while True:
        input_str = input("\nQuestion for this video: ")
        if input_str == "exit":
            break
        print("\nAnswer: ", end = '')
        images_prompt = ""
        for i in range(vila.num_frames):
            images_prompt += "<image>\n"
        prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {images_prompt}<video>\\n {input_str}. ASSISTANT:"
        tokens = tokenizer_image_token(prompt, vila.tokenizer)
        first_start = time.time()
        token = vila.forward_first(tokens)
        token_num = 0
        start = time.time()
        ftl = start - first_start
        while(token != 2):
            token_num += 1
            print(vila.tokenizer.decode([29871, token]), end="", flush=True)
            token = vila.forward_next()        

        end = time.time()
        print(f"\nftl: {ftl}")
        print(f"\ntps: {token_num / (end - start)}")

