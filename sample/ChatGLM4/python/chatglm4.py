#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sophon.sail as sail
import time
import argparse
from transformers import AutoTokenizer
import numpy as np

#convert sail_dtype to numpy dtype
def type_convert(sail_dtype):
    if sail_dtype == sail.Dtype.BM_FLOAT32:
        return np.float32
    if sail_dtype == sail.Dtype.BM_FLOAT16:
        return np.float16
    if sail_dtype == sail.Dtype.BM_INT32:
        return np.int32
    
    raise TypeError("only support float32 and int32 right now")

def fp16_cast(arr:np.ndarray): #这个接口的作用在于把np.float16假冒成np.uint16传进Tensor，sail update_data如果能接收传输二进制，那就不需要这个了。
    """
    reinterpret an array with int16 instead of float16, because pybind11 do not support float16.
    """
    if arr.dtype == np.float16:
        return arr.view(np.uint16)
    else:
        return arr
    
class ChatGLM4:
    def __init__(self, handle, engine, tokenizer):
        # load tokenizer
        self.sp = tokenizer
        self.handle = handle
        # warm up
        self.sp.decode([0]) 
        self.EOS = [self.sp.eos_token_id, self.sp.convert_tokens_to_ids("<|user|>"), self.sp.convert_tokens_to_ids("<|observation|>")]

        # load bmodel
        # 这里devio，后面都没有创建系统内存的tensor
        self.net = engine

        self.graph_names = self.net.get_graph_names()
        
        # initialize glm parameters
        self.NUM_LAYERS = (len(self.graph_names) - 5) // 2
        self.first_hidden_input_shape = self.net.get_input_shape("block_0", self.net.get_input_names("block_0")[0])
        batch, self.SEQLEN, self.HIDDEN_SIZE = self.first_hidden_input_shape
        self.is_greedy_sample = True
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_sample = "greedy_head" if self.is_greedy_sample else "penalty_sample_head"

        # tensors:
        # forward_first: embedding_tensor
        self.first_embed_input = self.init_sail_tensor(self.name_embed, 0, [1, self.SEQLEN])
        self.first_embed_output = self.init_sail_tensor(self.name_embed, 0, [1, self.SEQLEN, self.HIDDEN_SIZE], False)

        # forward_next: embedding_tensor
        self.next_embed_input = self.init_sail_tensor(self.name_embed_cache, 0, [1, 1])
        self.next_embed_output = self.init_sail_tensor(self.name_embed_cache, 0, [1,  1, self.HIDDEN_SIZE], False)

        # forward_first: hidden_state
        self.first_hidden_input = self.init_sail_tensor(self.name_blocks[0], 0)
        self.first_hidden_output = self.init_sail_tensor(self.name_blocks[0], 0, None, False)

        # forward_next: hidden_state
        self.next_hidden_input = self.init_sail_tensor(self.name_blocks_cache[0], 0)
        self.next_hidden_output = self.init_sail_tensor(self.name_blocks_cache[0], 0, None, False)

        # forward_first: position_id_tensor 和 attention_mask_tensor
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

        for i in range(self.NUM_LAYERS):
            self.past_key_output.append(self.init_sail_tensor(self.name_blocks[0], 1, None, False))
            self.past_value_output.append(self.init_sail_tensor(self.name_blocks[0], 2, None, False))
            self.past_key_output[i]["data"].memory_set(0)
            self.past_value_output[i]["data"].memory_set(0)

            self.cache_key_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 3))
            self.cache_key_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False))

            self.cache_value_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 4))
            self.cache_value_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False))

        # lm_head tensor
        self.lm_input = self.init_sail_tensor(self.name_lm, 0)
        self.lm_output = self.init_sail_tensor(self.name_lm, 0, None, False)

        # sample tensor
        self.sample_input = self.init_sail_tensor(self.name_sample, 0)
        self.sample_output = self.init_sail_tensor(self.name_sample, 0, None, False)

        self.token_length = 0
        self.round = 0

    def init_sail_tensor(self, name, tensor_idx, shape=None, is_input=True):
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
        if is_input:
            tensor["name"] = self.net.get_input_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_input_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_input_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True)
        else:
            tensor["name"] = self.net.get_output_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_output_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_output_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True) 
        return tensor
        
    def forward_first(self, token):
        # Keep

        input_ids = np.zeros(self.SEQLEN, type_convert(self.first_embed_input["dtype"]))

        # token = token[-512:] if len(token) > 512 else token
        input_ids[:min(self.SEQLEN, len(token))] = token
        self.token_length = len(token)
        input_ids = input_ids.reshape(1, -1)

        position_id = np.zeros(self.SEQLEN, type_convert(self.first_pid["dtype"])) 
        position_id[:self.token_length] = range(self.token_length)
            
        attention_mask = np.zeros(self.SEQLEN*self.SEQLEN, type_convert(self.first_attention["dtype"])) -1000.0#这里的type要从模型获取。
        for i in range(self.token_length):
            for j in range(i+1):
                attention_mask[i*self.SEQLEN + j] = 0

        # embedding
        self.first_embed_input["data"].update_data(fp16_cast(input_ids))
        input_embed_tensors = {self.first_embed_input["name"]: self.first_embed_input["data"]}
        output_embed_tensors = {self.first_embed_output["name"]: self.first_embed_output["data"]}
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)

        # blocks
        self.first_hidden_tensor = self.first_embed_output["data"]
        self.first_hidden_tensor.reshape(self.first_hidden_input["shape"])
        self.first_pid["data"].update_data(fp16_cast(position_id.reshape(self.first_pid["shape"])))
        self.first_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.first_attention["shape"])))
        
        input_blocks_tensors = {self.first_hidden_input["name"]: self.first_hidden_tensor, 
                                self.first_pid["name"]: self.first_pid["data"], 
                                self.first_attention["name"]: self.first_attention["data"]}

        for i in range(self.NUM_LAYERS):        
            output_blocks_tensors = {self.first_hidden_output["name"]: self.first_hidden_tensor,
                                    self.past_key_output[i]["name"]: self.past_key_output[i]["data"],
                                    self.past_value_output[i]["name"]: self.past_value_output[i]["data"],}
            self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)
        # lm_head
        # hidden_states 的最后一个位置的元素取出来作为 lm_head的输入
        copy_len = self.first_hidden_tensor.shape()[-1]
        self.lm_input["data"].sync_d2d(self.first_hidden_tensor,
                                      (self.token_length-1)* copy_len,  
                                      0, 
                                      copy_len)
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input["data"]}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}

        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)

        # sample
        input_sample_tensor = {self.sample_input["name"]: self.lm_output["data"]}
        output_sample_tensor = {self.sample_output["name"]: self.sample_output["data"]}
        self.net.process(self.name_sample, input_sample_tensor, output_sample_tensor)
        return int(self.sample_output["data"].asnumpy())

    def forward_next(self, ):
        attention_mask = np.zeros(self.SEQLEN+1, type_convert(self.next_attention["dtype"]))
        for i in range(self.SEQLEN - self.token_length + 1):
            attention_mask[i] = 1
        position_id = np.array(self.token_length - 1, type_convert(self.next_pid["dtype"]))

        # embedding
        self.next_embed_input["data"] = self.sample_output["data"]
        self.next_embed_input["data"].reshape(self.next_embed_input["shape"])

        input_embed_tensors = {self.next_embed_input["name"]: self.next_embed_input["data"]}
        output_embed_tensors = {self.next_embed_output["name"]: self.next_embed_output["data"]}
        self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)

        # blocks
        self.next_pid["data"].update_data(fp16_cast(position_id.reshape(self.next_pid["shape"])))
        self.next_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.next_attention["shape"])))

        self.next_hidden_tensor = self.next_embed_output["data"]
        self.next_hidden_tensor.reshape(self.next_hidden_input["shape"])

        for i in range(self.NUM_LAYERS):
            inputs_block_cache_tensors = {self.next_hidden_input["name"]: self.next_hidden_tensor, 
                                        self.next_pid["name"]: self.next_pid["data"], 
                                        self.next_attention["name"]: self.next_attention["data"], 
                                        self.cache_key_input[i]["name"]: self.past_key_output[i]["data"], 
                                        self.cache_value_input[i]["name"]: self.past_value_output[i]["data"]}
            outputs_block_cache_tensors = {self.next_hidden_output["name"]: self.next_hidden_tensor,
                                        self.cache_key_output[i]["name"]: self.present_key["data"],
                                        self.cache_value_output[i]["name"]: self.present_value["data"]}
            self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)
            unit_size = self.present_key["shape"][-1]
            for j in range(self.present_key["shape"][1]):
                self.past_key_output[i]["data"].sync_d2d(self.present_key["data"], j*unit_size, (j*self.SEQLEN + self.token_length-1)*unit_size, unit_size)
                self.past_value_output[i]["data"].sync_d2d(self.present_value["data"], j*unit_size, (j*self.SEQLEN + self.token_length-1)*unit_size, unit_size)


        self.lm_input_tensor = self.next_hidden_tensor
        self.lm_input_tensor.reshape(self.lm_input["shape"])
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input_tensor}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        # sample
        input_sample_tensor = {self.sample_input["name"]: self.lm_output["data"]}
        output_sample_tensor = {self.sample_output["name"]: self.sample_output["data"]}
        self.net.process(self.name_sample, input_sample_tensor, output_sample_tensor)
        return int(self.sample_output["data"].asnumpy())
            
    def chat_stream(self, input, history):
        history.append({"role": "user", "content": input})
        input_tokens = self.sp.apply_chat_template([{"role": "user", "content": input}], add_generation_prompt=True)[0]
        if (len(input_tokens) > self.SEQLEN / 3):
            yield 'input length is too long'
            return
        tok_num = 0
        tokens = self.sp.apply_chat_template(history, add_generation_prompt=True)[0]
        while (len(tokens) > self.SEQLEN / 2):
            history = history[1:]
            tokens = self.sp.apply_chat_template(history, add_generation_prompt=True)[0]
        first_start = time.time()
        pre_token = self.forward_first(tokens)
        first_end = time.time()
        token = pre_token
        emoji_token = []
        while token not in self.EOS and self.token_length < self.SEQLEN:
            if ('�' in self.sp.decode([token])):
                # 处理emoji表情和生僻字
                emoji_token.append(token)
            else:
                if emoji_token:
                    yield self.sp.decode(emoji_token)
                    emoji_token = []
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



def app(client):
    history = []
    while True:
        input_str = input("\nQuestion: ")
        if input_str == "exit":
            break
        print("\nAnswer: ")
        assistant_msg = ''
        for response in client.chat_stream(input_str, history):
            assistant_msg = response
            print(response, flush=True, end='')
        history.append({"role": "user", "content": input_str})
        history.append({"role": "assistant", "content": assistant_msg})

def main(args):
    handle = sail.Handle(args.dev_id)
    tokenizer = AutoTokenizer.from_pretrained(args.token, trust_remote_code=True)
    engine = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.DEVIO)
    client = ChatGLM4(handle, engine, tokenizer)
    app(client)

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str, default='./models/BM1684X/glm4-9b_int4_1dev.bmodel', help='path of bmodel')
    parser.add_argument('--token', type=str, default='./python/token_config/', help='path of tokenizer')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done')