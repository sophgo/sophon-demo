#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import time
import numpy as np
import sophon.sail as sail
sail.set_loglevel(sail.LogLevel.ERROR)

class LLM2TTSCodecAR():
    def __init__(self, args):
        self.version = "1.0.0"
        # dev_id
        self.dev_ids = [args.dev_id]
        self.handles = {dev: sail.Handle(dev) for dev in self.dev_ids}
        self.vocab_size = 1024
        # load model
        start_time = time.time()
        self.model = sail.EngineLLM(args.codec_model_path, self.dev_ids)
        self.graph_names = self.model.get_graph_names()
        load_model_time = time.time() - start_time
        print(f"sail.EngineLLM init cost: {load_model_time:.3f} s")
        
        # initialize parameters
        self.NUM_LAYERS = 4
        self.NUM_PREE_LAYERS = 2
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
        else:
            print("Ensure your bmodel has greedy head.")

        # initialize net name
        self.name_embed_tts = "tts_embedding"
        self.name_pres = ["pre_"+str(i) for i in range(self.NUM_PREE_LAYERS)]
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_lm = "lm_head"
        self.greedy = "greedy_head"
        self.penalty = "penalty_sample_head"
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
    
        self.tts_embed_input = self.model.create_max_input_tensors(self.name_embed_tts)
        self.tts_embed_output = self.model.create_max_output_tensors(self.name_embed_tts)
        self.pre_pid = {}
        self.first_hidden_state = {} #self.model.create_max_input_tensors(self.name_blocks[0])
        self.first_pid = {}
        self.next_pid = {}
        self.first_attention_mask = {}
        self.next_attention_mask = {}
        self.lm_input = self.model.create_max_input_tensors(self.name_lm)
        self.lm_output = self.model.create_max_output_tensors(self.name_lm)
        for i in range(len(self.dev_ids)):
            self.first_hidden_state[i] = self.init_tensor(self.dev_ids[i], self.tensors[self.name_blocks[0]]["input"][0])
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

    def pre_nn_forward(self, inputs_embeds, position_ids):
        seq_len = self.token_length + 1 if self.is_dynamic else self.SEQLEN
        hidden_states = inputs_embeds
        attention_mask = np.ones((seq_len, seq_len), 
                                 self.type_convert(self.tensors[self.name_pres[0]]["input"][2].dtype())) * self.ATTENTION_MASK
        for i in range(1, self.token_length):
            for j in range(1, self.token_length):
                attention_mask[i][j] = 0.0
        # blocks
        for i in range(len(self.dev_ids)):
            self.first_hidden_state[i].update_data(hidden_states.view(np.uint16))
            self.first_hidden_state[i].sync_s2d()
            self.tensors[self.name_pres[0]]["input"][3 * i + 1] = sail.Tensor(self.first_pid[i], [1, seq_len], 0)
            self.tensors[self.name_pres[0]]["input"][3 * i + 2] = sail.Tensor(self.first_attention_mask[i], [1, 1, seq_len, seq_len], 0)
            self.tensors[self.name_pres[0]]["input"][3 * i + 1].update_data(position_ids.reshape(self.tensors[self.name_pres[0]]["input"][3 * i + 1].shape()))
            self.tensors[self.name_pres[0]]["input"][3 * i + 1].sync_s2d()
            self.tensors[self.name_pres[0]]["input"][3 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_pres[0]]["input"][3 * i + 2].shape()).view(np.uint16))
            self.tensors[self.name_pres[0]]["input"][3 * i + 2].sync_s2d()
        for i in range(self.NUM_PREE_LAYERS):
            for j in range(len(self.dev_ids)):
                self.tensors[self.name_pres[i]]["input"][3 * j] = sail.Tensor(self.first_hidden_state[j], [1, seq_len, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_pres[i]]["output"][3 * j] = sail.Tensor(self.first_hidden_state[j], [1, seq_len, self.HIDDEN_SIZE], 0)
                # self.tensors[self.name_pres[i]]["output"][3 * j + 1] = sail.Tensor(self.past_k[i][j], [1, seq_len, self.ATTEN_HEAD, self.ATTEN_DIM], 0)
                # self.tensors[self.name_pres[i]]["output"][3 * j + 2] = sail.Tensor(self.past_v[i][j], [1, seq_len, self.ATTEN_HEAD, self.ATTEN_DIM], 0)
            if i > 0:
                for j in range(len(self.dev_ids)):
                    self.tensors[self.name_pres[i]]["input"][3 * j + 1] = self.tensors[self.name_pres[0]]["input"][3 * j + 1]
                    self.tensors[self.name_pres[i]]["input"][3 * j + 2] = self.tensors[self.name_pres[0]]["input"][3 * j + 2]
            self.model.process(self.name_pres[i], self.tensors[self.name_pres[i]]["input"], self.tensors[self.name_pres[i]]["output"])
    def forward_first(self):
        seq_len = self.token_length + 1 if self.is_dynamic else self.SEQLEN
        attention_mask = np.ones((seq_len, seq_len), 
                                 self.type_convert(self.tensors[self.name_blocks[0]]["input"][2].dtype())) * self.ATTENTION_MASK
        for i in range(self.token_length):
            for j in range(self.token_length):
                if j <= i:
                    attention_mask[i][j] = 0.0
        position_ids = list(range(self.token_length))
        if not self.is_dynamic:
            position_ids += (self.SEQLEN - self.token_length) * [0]
        position_ids = np.array(position_ids, np.int32)
        bos_id = np.array([[self.vocab_size]], self.type_convert(self.tensors[self.name_embed_tts]["input"][0].dtype()))
        for i in range(len(self.dev_ids)):
            self.tts_embed_input[i].update_data(bos_id.reshape(self.tensors[self.name_embed_tts]["input"][i].shape()))
            self.tts_embed_input[i].sync_s2d()
            self.tensors[self.name_embed_tts]["input"][i] = self.tts_embed_input[i]

        for i in range(len(self.dev_ids)):
            self.tensors[self.name_embed_tts]["output"][i] = sail.Tensor(self.first_hidden_state[i], 
                                                                         self.tensors[self.name_embed_tts]["output"][i].shape(), 
                                                                         0)

        self.model.process(self.name_embed_tts, self.tensors[self.name_embed_tts]["input"], self.tensors[self.name_embed_tts]["output"])

        # blocks
        for i in range(len(self.dev_ids)):
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1] = sail.Tensor(self.first_pid[i], [1, seq_len], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 2] = sail.Tensor(self.first_attention_mask[i], [1, 1, seq_len, seq_len], 0)
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1].update_data(position_ids.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 1].shape()))
            self.tensors[self.name_blocks[0]]["input"][3 * i + 1].sync_s2d()
            self.tensors[self.name_blocks[0]]["input"][3 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks[0]]["input"][3 * i + 2].shape()).view(np.uint16))
            self.tensors[self.name_blocks[0]]["input"][3 * i + 2].sync_s2d()
        for i in range(self.NUM_LAYERS):
            for j in range(len(self.dev_ids)):
                self.tensors[self.name_blocks[i]]["input"][3 * j] = sail.Tensor(self.first_hidden_state[j], [1, seq_len, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j] = sail.Tensor(self.first_hidden_state[j], [1, seq_len, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j + 1] = sail.Tensor(self.past_k[i][j], [1, seq_len, self.ATTEN_HEAD, self.ATTEN_DIM], 0)
                self.tensors[self.name_blocks[i]]["output"][3 * j + 2] = sail.Tensor(self.past_v[i][j], [1, seq_len, self.ATTEN_HEAD, self.ATTEN_DIM], 0)
            if i > 0:
                for j in range(len(self.dev_ids)):
                    self.tensors[self.name_blocks[i]]["input"][3 * j + 1] = self.tensors[self.name_blocks[0]]["input"][3 * j + 1]
                    self.tensors[self.name_blocks[i]]["input"][3 * j + 2] = self.tensors[self.name_blocks[0]]["input"][3 * j + 2]
            self.model.process(self.name_blocks[i], self.tensors[self.name_blocks[i]]["input"], self.tensors[self.name_blocks[i]]["output"])

    def forward_next(self, cur_token):
        seq_len = self.token_length + 1 if self.is_dynamic else self.SEQLEN
        position_id = np.array([[self.token_length - 1]], self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][1].dtype()))
        attention_mask = np.zeros(seq_len + 1, self.type_convert(self.tensors[self.name_blocks_cache[0]]["input"][2].dtype()))
        attention_mask[self.token_length - 1:] = self.ATTENTION_MASK

        for i in range(len(self.dev_ids)):
            self.tts_embed_input[i].update_data(cur_token.reshape(self.tensors[self.name_embed_tts]["input"][i].shape()))
            self.tts_embed_input[i].sync_s2d()
            self.tensors[self.name_embed_tts]["input"][i] = self.tts_embed_input[i]

        for i in range(len(self.dev_ids)):
            self.tensors[self.name_embed_tts]["output"][i] = self.tts_embed_output[i]

        self.model.process(self.name_embed_tts, self.tensors[self.name_embed_tts]["input"], self.tensors[self.name_embed_tts]["output"])

        # block_cache
        for i in range(len(self.dev_ids)):
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1] = self.next_pid[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2] = self.next_attention_mask[i]
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].update_data(position_id.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].shape()))
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 1].sync_s2d()
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].shape()).view(np.uint16))
            self.tensors[self.name_blocks_cache[0]]["input"][5 * i + 2].sync_s2d()

        for i in range(self.NUM_LAYERS):
            for j in range(len(self.dev_ids)):
                self.tensors[self.name_blocks_cache[i]]["input"][5 * j] = self.tts_embed_output[j]
                self.tensors[self.name_blocks_cache[i]]["output"][3 * j] = self.tts_embed_output[j]
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
        self.tensors[self.name_lm]["input"][0] = self.tts_embed_output[0]
        self.tensors[self.name_lm]["output"][0] = self.lm_output[0]
        print("lm input:", np.average(self.tensors[self.name_lm]["input"][0].asnumpy()))
        print("lm output:", np.average(self.tensors[self.name_lm]["output"][0].asnumpy()))
        self.model.process(self.name_lm, self.tensors[self.name_lm]["input"], self.tensors[self.name_lm]["output"])
        # sample
        self.tensors[self.greedy]["input"][0] = self.tensors[self.name_lm]["output"][0]
        self.model.process(self.greedy, self.tensors[self.greedy]["input"], self.tensors[self.greedy]["output"])
        self.tensors[self.greedy]["output"][0].sync_d2s()
        return self.tensors[self.greedy]["output"][0].asnumpy().squeeze(0)
        
    def infer(self, hidden, max_tokens=1000):
        self.token_length = hidden.shape[1] + 1
        position_ids = [0] + list(range(self.token_length)) + (self.SEQLEN - self.token_length - 1) * [0]
        position_ids = np.array([position_ids], dtype=np.int32)
        padded_hidden = np.pad(
            hidden,
            pad_width=((0, 0), (1, self.SEQLEN - hidden.shape[1] - 1), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        self.pre_nn_forward(padded_hidden, position_ids)
        self.forward_first()
        cur_token = np.array([[self.vocab_size + 1]], self.type_convert(self.tensors[self.name_embed_tts]["input"][0].dtype()))
        for i in range(max_tokens):
            self.token_length += 1
            if self.token_length == self.SEQLEN:
                print("exceed max seq length...")
                break
            cur_token = self.forward_next(cur_token)
            if cur_token == self.vocab_size + 2:
                break
            yield cur_token