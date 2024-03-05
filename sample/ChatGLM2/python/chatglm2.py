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
import sentencepiece as spm
import numpy as np

def type_convert(sail_dtype):
    if sail_dtype == sail.Dtype.BM_FLOAT32:
        return np.float32
    if sail_dtype == sail.Dtype.BM_INT32:
        return np.int32
    
    raise TypeError("only support float32 and int32 right now")

class ChatGLM2:
    def __init__(self, args):
        # self.dev_id = args.dev_id

        self.NUM_LAYERS = 28
        self.MAX_LEN = 512
        self.HIDDEN_SIZE = 4096

        # load tokenizer
        print("Load " + args.token + " ...")
        self.sp = spm.SentencePieceProcessor(model_file=args.token)
        self.EOS = self.sp.eos_id()
        print("Done!")

        
        # load bmodel
        # 这里devio，后面都没有创建系统内存的tensor
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.DEVIO)
        self.handle = sail.Handle(args.dev_id)
        self.graph_name = self.net.get_graph_names()

        self.name_embed = "embedding"
        self.name_lm = "lm_head"
        self.name_blocks = ["glm_block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["glm_block_cache_"+str(i) for i in range(self.NUM_LAYERS)]

        # tensors:
        # forward_first: embedding_tensor
        self.first_embed_input_name = self.net.get_input_names(self.name_embed)[0]
        self.first_embed_input_shape = [self.MAX_LEN]
        self.first_embed_input_dtype = self.net.get_input_dtype(self.name_embed, self.first_embed_input_name)
        self.first_embed_input_tensor = sail.Tensor(self.handle, self.first_embed_input_shape, self.first_embed_input_dtype, False, True)

        self.first_embed_output_name = self.net.get_output_names(self.name_embed)[0]
        self.first_embed_output_shape = [self.MAX_LEN, self.HIDDEN_SIZE]
        self.first_embed_output_dtype = self.net.get_output_dtype(self.name_embed, self.first_embed_output_name)
        self.first_embed_output_tensor = sail.Tensor(self.handle, self.first_embed_output_shape, self.first_embed_output_dtype, False, True)

        # forward_next: embedding_tensor
        self.next_embed_input_name = self.net.get_input_names(self.name_embed)[0]
        self.next_embed_input_shape = [1]
        self.next_embed_input_dtype = self.net.get_input_dtype(self.name_embed, self.next_embed_input_name)
        self.next_embed_input_tensor = sail.Tensor(self.handle, self.next_embed_input_shape, self.next_embed_input_dtype, False, True)

        self.next_embed_output_name = self.net.get_output_names(self.name_embed)[0]
        self.next_embed_output_shape = [1, self.HIDDEN_SIZE]
        self.next_embed_output_dtype = self.net.get_output_dtype(self.name_embed, self.next_embed_output_name)
        self.next_embed_output_tensor = sail.Tensor(self.handle, self.next_embed_output_shape, self.next_embed_output_dtype, False, True)

        # forward_first: hidden_state
        self.first_hidden_input_name = self.net.get_input_names(self.name_blocks[0])[0]
        self.first_hidden_input_shape = self.net.get_input_shape(self.name_blocks[0], self.first_hidden_input_name)

        self.first_hidden_output_name = self.net.get_output_names(self.name_blocks[0])[0]

        # forward_next: hidden_state
        self.next_hidden_input_name = self.net.get_input_names(self.name_blocks_cache[0])[0]
        self.next_hidden_input_shape = self.net.get_input_shape(self.name_blocks_cache[0], self.next_hidden_input_name)

        self.next_hidden_output_name = self.net.get_output_names(self.name_blocks_cache[0])[0]

        # forward_first: position_id_tensor 和 attention_mask_tensor
        self.first_pid_name = self.net.get_input_names(self.name_blocks[0])[1]
        self.first_pid_shape = self.net.get_input_shape(self.name_blocks[0], self.first_pid_name)
        self.first_pid_dtype = self.net.get_input_dtype(self.name_blocks[0], self.first_pid_name)
        self.first_pid_tensor = sail.Tensor(self.handle, self.first_pid_shape, self.first_pid_dtype, False, True)

        self.first_attention_name = self.net.get_input_names(self.name_blocks[0])[2]
        self.first_attention_shape = self.net.get_input_shape(self.name_blocks[0], self.first_attention_name)
        self.first_attention_dtype = self.net.get_input_dtype(self.name_blocks[0], self.first_attention_name)
        self.first_attention_tensor = sail.Tensor(self.handle, self.first_attention_shape, self.first_attention_dtype, False, True)

        # forward_next: position_id_tensor and attention_mask_tensor
        self.next_pid_name = self.net.get_input_names(self.name_blocks_cache[0])[1]
        self.next_pid_shape = self.net.get_input_shape(self.name_blocks_cache[0], self.next_pid_name)
        self.next_pid_dtype = self.net.get_input_dtype(self.name_blocks_cache[0], self.next_pid_name)
        self.next_pid_tensor = sail.Tensor(self.handle, self.next_pid_shape, self.next_pid_dtype, False, True)

        self.next_attention_name = self.net.get_input_names(self.name_blocks_cache[0])[2]
        self.next_attention_shape = self.net.get_input_shape(self.name_blocks_cache[0], self.next_attention_name)
        self.next_attention_dtype = self.net.get_input_dtype(self.name_blocks_cache[0], self.next_attention_name)
        self.next_attention_tensor = sail.Tensor(self.handle, self.next_attention_shape, self.next_attention_dtype, False, True)

        # forward_first: key_tensor 和 value_tensor
        self.past_key_tensor = []
        self.past_key_name = []
        self.past_key_shape = []
        self.past_key_dtype = []
        self.past_value_tensor = []
        self.past_value_name = []
        self.past_value_shape = []
        self.past_value_dtype = []
        # forward_next: cache block的kv tensor名
        self.cache_key_input_name = []
        self.cache_value_input_name = []
        self.cache_key_output_name = []
        self.cache_value_output_name = []

        for i in range(self.NUM_LAYERS):
            
            self.past_key_name.append(self.net.get_output_names(self.name_blocks[0])[1])
            self.past_key_shape.append(self.net.get_output_shape(self.name_blocks[0], self.past_key_name[i]))
            self.past_key_dtype.append(self.net.get_output_dtype(self.name_blocks[0], self.past_key_name[i]))
            self.past_key_tensor.append(sail.Tensor(self.handle, self.past_key_shape[i], self.past_key_dtype[i], False, True))

            self.past_value_name.append(self.net.get_output_names(self.name_blocks[0])[2])
            self.past_value_shape.append(self.net.get_output_shape(self.name_blocks[0], self.past_value_name[i]))
            self.past_value_dtype.append(self.net.get_output_dtype(self.name_blocks[0], self.past_value_name[i]))
            self.past_value_tensor.append(sail.Tensor(self.handle, self.past_value_shape[i], self.past_value_dtype[i], False, True))
            
            self.cache_key_input_name.append(self.net.get_input_names(self.name_blocks_cache[0])[3])
            self.cache_value_input_name.append(self.net.get_input_names(self.name_blocks_cache[0])[4])

            self.cache_key_output_name.append(self.net.get_output_names(self.name_blocks_cache[0])[1])
            self.cache_value_output_name.append(self.net.get_output_names(self.name_blocks_cache[0])[2])
        
        # lm_head tensor
        self.lm_input_name = self.net.get_input_names(self.name_lm)[0]
        self.lm_input_shape = self.net.get_input_shape(self.name_lm, self.lm_input_name)
        self.lm_input_dtype = self.net.get_input_dtype(self.name_lm, self.lm_input_name)
        self.lm_input_tensor = sail.Tensor(self.handle, self.lm_input_shape, self.lm_input_dtype, False, True)

        self.lm_output_name = self.net.get_output_names(self.name_lm)[0]
        self.lm_output_shape = self.net.get_output_shape(self.name_lm, self.lm_output_name)
        self.lm_output_dtype = self.net.get_output_dtype(self.name_lm, self.lm_output_name)
        self.lm_output_tensor = sail.Tensor(self.handle, self.lm_output_shape, self.lm_output_dtype, False, True)
        
        self.history = ""
        self.token_length = 0
        self.round = 0


    def move2end(self, kv:sail.Tensor):
        if self.token_length >= self.MAX_LEN:
            return
        arr = kv.asnumpy()
        shape = arr.shape
        total = np.prod(shape)
        real = total//self.MAX_LEN*self.token_length
        arr = arr.reshape(-1)
        arr[total-real:] = arr[:real]
        arr[:total-real] = 0
        arr = arr.reshape(shape)
        kv.update_data(arr)


    def forward_first(self, token):
        input_ids = np.zeros(self.MAX_LEN, np.int32)
        input_ids[0], input_ids[1] = 64790, 64792
        position_id = np.zeros(self.MAX_LEN, np.int32)
        attention_mask = np.zeros(self.MAX_LEN*self.MAX_LEN, np.float32)
        input_ids[2:len(token)+2] = token
        self.token_length = len(token)+2
        for i in range(self.token_length):
            position_id[i] = i
        for i in range(self.MAX_LEN):
            for j in range(self.MAX_LEN):
                if not (j <= i and i < self.token_length):
                    attention_mask[i*self.MAX_LEN + j] = 1
        

        # embedding
        self.first_embed_input_tensor.update_data(input_ids)

        input_embed_tensors = {self.first_embed_input_name: self.first_embed_input_tensor}
        output_embed_tensors = {self.first_embed_output_name: self.first_embed_output_tensor}
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)

        # blocks
        self.first_hidden_tensor = self.first_embed_output_tensor
        self.first_hidden_tensor.reshape(self.first_hidden_input_shape)

        self.first_pid_tensor.update_data(position_id.reshape(self.first_pid_shape))
        self.first_attention_tensor.update_data(attention_mask.reshape(self.first_attention_shape))

        input_blocks_tensors = {self.first_hidden_input_name: self.first_hidden_tensor, 
                                self.first_pid_name: self.first_pid_tensor, 
                                self.first_attention_name: self.first_attention_tensor}

        for i in range(self.NUM_LAYERS):        

            output_blocks_tensors = {self.first_hidden_output_name: self.first_hidden_tensor,
                                    self.past_key_name[i]: self.past_key_tensor[i],
                                    self.past_value_name[i]: self.past_value_tensor[i]}
            
            self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)

            self.move2end(self.past_key_tensor[i])
            self.move2end(self.past_value_tensor[i])


        # lm_head
        first_hidden_new_shape = [self.first_hidden_tensor.shape()[0], self.first_hidden_tensor.shape()[-1]]
        self.first_hidden_tensor.reshape(first_hidden_new_shape)
        self.lm_input_tensor = sail.Tensor(self.first_hidden_tensor,[(self.token_length-1, self.token_length), (0, first_hidden_new_shape[1])], True)

        input_lm_tensors = {self.lm_input_name: self.lm_input_tensor}
        output_lm_tensors = {self.lm_output_name: self.lm_output_tensor}

        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)

        return int(self.lm_output_tensor.asnumpy())


    def forward_next(self, ):
        attention_mask = np.zeros(self.MAX_LEN+1, np.float32)
        for i in range(self.MAX_LEN - self.token_length):
            attention_mask[i] = 1
        position_id = np.array(self.token_length - 1, dtype=np.int32)

        # embedding
        self.next_embed_input_tensor = self.lm_output_tensor
        self.next_embed_input_tensor.reshape(self.next_embed_input_shape)

        input_embed_tensors = {self.next_embed_input_name: self.next_embed_input_tensor}
        output_embed_tensors = {self.next_embed_output_name: self.next_embed_output_tensor}
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)

        # blocks
        self.next_pid_tensor.update_data(position_id.reshape(self.next_pid_shape))
        self.next_attention_tensor.update_data(attention_mask.reshape(self.next_attention_shape))

        self.next_hidden_tensor = self.next_embed_output_tensor
        self.next_hidden_tensor.reshape(self.next_hidden_input_shape)


        for i in range(self.NUM_LAYERS):
            inputs_block_cache_tensors = {self.next_hidden_input_name: self.next_hidden_tensor, 
                                        self.next_pid_name: self.next_pid_tensor, 
                                        self.next_attention_name: self.next_attention_tensor, 
                                        self.cache_key_input_name[i]: self.past_key_tensor[i], 
                                        self.cache_value_input_name[i]: self.past_value_tensor[i]}
            outputs_block_cache_tensors = {self.next_hidden_output_name: self.next_hidden_tensor,
                                        self.cache_key_output_name[i]: self.past_key_tensor[i],
                                        self.cache_value_output_name[i]: self.past_value_tensor[i]}
            self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)

        self.lm_input_tensor = self.next_hidden_tensor
        self.lm_input_tensor.reshape(self.lm_input_shape)
        
        input_lm_tensors = {self.lm_input_name: self.lm_input_tensor}
        output_lm_tensors = {self.lm_output_name: self.lm_output_tensor}
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)

        return int(self.lm_output_tensor.asnumpy())
        

    def chat(self,):
        while True:
            input_str = input("\nQuestion: ")
            # input_str = "hi"
            if input_str == "exit":
                break
            print("\nAnswer: ")
            self.answer(input_str)
            print()


    def answer(self, input_str):
        self.history += "[Round " + str(self.round + 1) + "]\n\n问：" + input_str + "\n\n答："
        tok_num = 1
        tokens = self.sp.Encode(self.history)
        # tokens.empty()不知道是不是一致
        if not tokens:
            print("Sorry: your question is too wierd!!")
            self.history = ""
            self.round = 0
            return
        if len(tokens) > self.MAX_LEN - 10:
            if self.round == 0:
                print("Error: your question is too large!")
                return
            self.round = 0
            self.history = ""
            self.answer(input_str)
            return
        
        # sentencepiece不接受numpy做输入，但tensor那些得用numpy生成，
        # 原仓库也用过类似tolist的方式来回转
        pre_token = 0
        first_start = time.time()
        token = self.forward_first(tokens)
        first_end = time.time()
        
        while token != self.EOS and self.token_length < self.MAX_LEN:
            pre_ids = [pre_token]
            ids = [pre_token, token]
            pre_word = self.sp.Decode(pre_ids)
            word = self.sp.Decode(ids)
            diff = word[len(pre_word):]
            self.history += diff
            print(diff, flush=True, end='')
            if self.token_length < self.MAX_LEN:
                self.token_length += 1
            tok_num += 1
            token = self.forward_next()
        
        # 计时
        next_end = time.time()
        first_duration = first_end-first_start
        next_duration = next_end-first_end
        tps = tok_num / next_duration
        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

        if self.token_length >= self.MAX_LEN:
            self.round = 0
            self.history = self.history[:len(self.history)//2]
        else:
            self.history += "\n\n"
            self.round += 1
            

def main(args):
    chatglm2 = ChatGLM2(args)
    chatglm2.chat()
    pass


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/chatglm2-6b_fp16.bmodel', help='path of bmodel')
    parser.add_argument('--token', type=str, default='../models/BM1684X/tokenizer.model', help='path of tokenizer')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done')