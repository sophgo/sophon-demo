#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sophon.sail as sail
import argparse
import time
from transformers import AutoTokenizer
import numpy as np

class Baichuan2:

    def __init__(self, handle, engine, tokenizer):
        # load tokenizer
        self.sp = tokenizer
        self.handle = handle
        self.EOS = self.sp.eos_token_id

        # load bmodel
        # 这里devio，后面都没有创建系统内存的tensor
        self.net = engine
        self.graph_name = self.net.get_graph_names()
        
        self.NUM_LAYERS = 32
        self.MAX_LEN = 512
        self.ATTENTION_MASK = -1000.
        _, _, self.HIDDEN_SIZE = self.net.get_input_shape("block_0", self.net.get_input_names("block_0")[0])
        
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]

        # tensors:
        # forward_first: embedding_tensor
        self.first_embed_input = self.init_sail_tensor(self.name_embed, 0, [1, self.MAX_LEN])
        self.first_embed_output = self.init_sail_tensor(self.name_embed, 0, [1, self.MAX_LEN, self.HIDDEN_SIZE], False)

        # forward_next: embedding_tensor
        self.next_embed_input = self.init_sail_tensor(self.name_embed_cache, 0, [1, 1])
        self.next_embed_output = self.init_sail_tensor(self.name_embed_cache, 0, [1, 1, self.HIDDEN_SIZE], False)

        # forward_first: hidden_state
        self.first_hidden_input = self.init_sail_tensor(self.name_blocks[0], 0)
        self.first_hidden_output = self.init_sail_tensor(self.name_blocks[0], 0, None, False)

        # forward_next: hidden_state
        self.next_hidden_input = self.init_sail_tensor(self.name_blocks_cache[0], 0)
        self.next_hidden_output = self.init_sail_tensor(self.name_blocks_cache[0], 0, None, False)

        # forward_first: position_id_tensor and attention_mask_tensor
        self.first_pid = self.init_sail_tensor(self.name_blocks[0], 1)
        self.first_attention = self.init_sail_tensor(self.name_blocks[0], 2)
       
        # forward_next: position_id_tensor and attention_mask_tensor
        self.next_pid = self.init_sail_tensor(self.name_blocks_cache[0], 1)
        self.next_attention = self.init_sail_tensor(self.name_blocks_cache[0], 2)

        # forward_next: present_key / present_value (for update kv_cache)
        self.present_key = self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False)
        self.present_value = self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False)

        # forward_first: key_tensor 和 value_tensor
        self.past_key_output = []
        self.past_value_output = []

        # forward_next: cache block的kv tensor名
        self.cache_key_input = []
        self.cache_key_output = []
        self.cache_value_input = []
        self.cache_value_output = []

        for _ in range(self.NUM_LAYERS):
            
            self.past_key_output.append(self.init_sail_tensor(self.name_blocks[0], 1, None, False))
            self.past_value_output.append(self.init_sail_tensor(self.name_blocks[0], 2, None, False))

            self.cache_key_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 3))
            self.cache_key_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False))

            self.cache_value_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 4))
            self.cache_value_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False))

        # lm_head tensor
        self.lm_input = self.init_sail_tensor(self.name_lm, 0)
        self.lm_output = self.init_sail_tensor(self.name_lm, 0, None, False)

        self.token_length = 0
        self.round = 0

    def init_sail_tensor(self, name, input_idx, shape=None, input_type=True):
        """
        init a sail tensor of sail.engine.
        parameters:
        input:
            name: str, graph_name/net_name
            tensor_idx: int, input/output tensor id
            shape: list[int], shape of tensor
            is_input: bool, is input tensor or not
        return:
            dict
        """
        tensor = {}
        if input_type:
            tensor["name"] = self.net.get_input_names(name)[input_idx]
            tensor["shape"] = self.net.get_input_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_input_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True)
        else:
            tensor["name"] = self.net.get_output_names(name)[input_idx]
            tensor["shape"] = self.net.get_output_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_output_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True) 
        return tensor

    def _make_context(self, input_str, history=[], role="user"):
        B_INST, E_INST = "<reserved_106>", "<reserved_107>"
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "user":
                encoded = self.sp.encode(f"{B_INST}{content.strip()}{E_INST}")
            elif item["role"] == "assistant" or item["role"] == "system":
                encoded = self.sp.encode(content.strip())
            else:
                raise ValueError(f"role should be in {{'system', 'user', 'assistant'}} but we get {item['role']}")
            # 添加编码后的消息，移除首个token（假设为bos token）
            input_ids.extend(encoded[1:])
        if role == "user" or role == "assistant" or item["role"] == "system":
            input_ids.extend(self.sp.encode(f"{B_INST}{input_str.strip()}{E_INST}")[1:])
        else:
            raise ValueError(f"role should be in {{'system', 'user', 'assistant'}} but we get {role}")
        return input_ids


    def forward_first(self, token):
        input_ids = np.zeros(self.MAX_LEN, dtype=np.int32)  # Initialize input_ids with zeros
        position_id = np.arange(self.MAX_LEN, dtype=np.int32)  # Use arange for position_id
        attention_mask = np.full((self.MAX_LEN, self.MAX_LEN), self.ATTENTION_MASK, dtype=np.float16)  # Initialize attention_mask with self.ATTENTION_MASK

        self.token_length = len(token)  
        input_ids[:self.token_length] = token  # Set the first part of input_ids to the token IDs
        position_id[self.token_length:] = 0
        lower_tri_indices = np.tril_indices(self.token_length)
        attention_mask[lower_tri_indices] = 0

        # embedding
        input_ids = input_ids.reshape(1, -1)
        self.first_embed_input["data"].update_data(input_ids)
        input_embed_tensors = {(self.first_embed_input["name"], 0): self.first_embed_input["data"]}
        output_embed_tensors = {(self.first_embed_output["name"], 0): self.first_embed_output["data"]}
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)

        # blocks
        self.first_hidden_tensor = self.first_embed_output["data"]
        self.first_hidden_tensor.reshape(self.first_hidden_input["shape"])
        self.first_pid["data"].update_data(position_id.reshape(self.first_pid["shape"]))
        self.first_attention["data"].update_data(attention_mask.reshape(self.first_attention["shape"]))

        input_blocks_tensors = {(self.first_hidden_input["name"], 0): self.first_hidden_tensor, 
                                (self.first_pid["name"], 0): self.first_pid["data"], 
                                (self.first_attention["name"], 0): self.first_attention["data"]}

        for i in range(self.NUM_LAYERS):        

            output_blocks_tensors = {(self.first_hidden_output["name"], 0): self.first_hidden_tensor,
                                    (self.past_key_output[i]["name"], 0): self.past_key_output[i]["data"],
                                    (self.past_value_output[i]["name"], 0): self.past_value_output[i]["data"]}
            
            self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)
        
        # lm_head
        # hidden_states 的最后一个位置的元素取出来作为 lm_head的输入
        copy_len = self.first_hidden_tensor.shape()[-1]
        self.lm_input["data"].sync_d2d(self.first_hidden_tensor, (self.token_length-1)* copy_len, 0, copy_len)
        
        input_lm_tensors = {(self.lm_input["name"], 0): self.lm_input["data"]}
        output_lm_tensors = {(self.lm_output["name"], 0): self.lm_output["data"]}
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return int(self.lm_output["data"].asnumpy())

    def forward_next(self, ):
        attention_mask = np.zeros(self.MAX_LEN+1, np.float16)
        attention_mask[self.token_length-1:self.MAX_LEN] = self.ATTENTION_MASK

        position_id = np.array(self.token_length - 1, dtype=np.int32)
        # embedding
        self.next_embed_input["data"] = self.lm_output["data"]
        self.next_embed_input["data"].reshape(self.next_embed_input["shape"])
        input_embed_tensors = {(self.next_embed_input["name"], 0): self.next_embed_input["data"]}
        output_embed_tensors = {(self.next_embed_output["name"], 0): self.next_embed_output["data"]}
        self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)

        # blocks
        self.next_pid["data"].update_data(position_id.reshape(self.next_pid["shape"]))
        self.next_attention["data"].update_data(attention_mask.reshape(self.next_attention["shape"]))

        self.next_hidden_tensor = self.next_embed_output["data"]
        self.next_hidden_tensor.reshape(self.next_hidden_input["shape"])

        for i in range(self.NUM_LAYERS):
            inputs_block_cache_tensors = {(self.next_hidden_input["name"], 0): self.next_hidden_tensor, 
                                        (self.next_pid["name"], 0): self.next_pid["data"], 
                                        (self.next_attention["name"], 0): self.next_attention["data"], 
                                        (self.cache_key_input[i]["name"], 0): self.past_key_output[i]["data"], 
                                        (self.cache_value_input[i]["name"], 0): self.past_value_output[i]["data"]}
            outputs_block_cache_tensors = {(self.next_hidden_output["name"], 0): self.next_hidden_tensor,
                                        (self.cache_key_output[i]["name"], 0): self.present_key["data"],
                                        (self.cache_value_output[i]["name"], 0): self.present_value["data"]}
            self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)

            # update kv_cache()
            unit_size = self.present_key["shape"][-1]*self.present_key["shape"][-2]
            self.past_key_output[i]["data"].sync_d2d(self.present_key["data"], 0, (self.token_length-1)*unit_size, unit_size)
            self.past_value_output[i]["data"].sync_d2d(self.present_value["data"], 0, (self.token_length-1)*unit_size, unit_size)

        self.lm_input_tensor = self.next_hidden_tensor
        self.lm_input_tensor.reshape(self.lm_input["shape"])
        
        input_lm_tensors = {(self.lm_input["name"], 0): self.lm_input_tensor}
        output_lm_tensors = {(self.lm_output["name"], 0): self.lm_output["data"]}
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return int(self.lm_output["data"].asnumpy())

    def chat_stream(self, input, history):
        input_tokens = self._make_context(input, history=[], role="user")
        if (len(input_tokens) > self.MAX_LEN - 10):
            yield 'input length is too long'
            return
        tok_num = 0
        tokens = self._make_context(input, history=history, role="user")
        while (len(tokens) > self.MAX_LEN / 2):
            history = history[1:]
            tokens = self._make_context(input, history=history, role="user")
        first_start = time.time()
        pre_token = self.forward_first(tokens)
        first_end = time.time()
        token = pre_token
        is_emoji = False
        emoji_token = []
        while token != self.EOS and self.token_length < self.MAX_LEN:
            if (token == 243 or is_emoji):
                # 处理emoji表情
                is_emoji = True
                emoji_token += [token]
                if (len(emoji_token) == 4):
                    yield self.sp.decode(emoji_token)
                    emoji_token = []
                    is_emoji = False
            else: 
                yield self.sp.decode([token])

            self.token_length += 1
            tok_num += 1
            token = self.forward_next()

        next_end = time.time()
        first_duration = first_end-first_start
        next_duration = next_end-first_end
        tps = tok_num / next_duration
        print('\n\n')
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")
    

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/baichuan2-7b_int8_1dev.bmodel', help='path of bmodel')
    parser.add_argument('--token', type=str, default='./token_config/', help='path of tokenizer')
    parser.add_argument('--dev_ids', nargs='+', type=int, default=[0], help='dev id list, split by space')
    args = parser.parse_args()
    return args

def app(client):
    history = []
    while True:
        input_str = input("\nQuestion: ")
        if input_str == "exit":
            break
        if input_str == "clear":
            history = []
            continue
        print("\nAnswer: ")
        assistant_msg = ''
        for response in client.chat_stream(input_str, history):
            assistant_msg = response
            print(response, flush=True, end='')
        history.append({"role": "user", "content": input_str})
        history.append({"role": "assistant", "content": assistant_msg})

def main(args):
    handle = sail.Handle(args.dev_ids[0])    # 传入list，只用第一个
    tokenizer = AutoTokenizer.from_pretrained(args.token, trust_remote_code=True)
    engine = sail.EngineLLM(args.bmodel, args.dev_ids)
    client = Baichuan2(handle, engine, tokenizer)
    app(client)

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done')
