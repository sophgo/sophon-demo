#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
import cv2
import os
import time

import sophon.sail as sail
import logging
logging.basicConfig(level=logging.INFO)



class CLIP:
    def __init__(self, text_model, dev_id):
    
        # text bmodel
        self.text_net = sail.Engine(text_model, dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(text_model))
        self.text_net_graph_name = self.text_net.get_graph_names()[0]
        self.text_net_input_name = self.text_net.get_input_names(self.text_net_graph_name)[0]
        self.text_net_output_name = self.text_net.get_output_names(self.text_net_graph_name)[0]
        self.text_net_input_shape = self.text_net.get_input_shape(self.text_net_graph_name, self.text_net_input_name)
        self.text_net_batch_size = self.text_net_input_shape[0]

        self.top_k = 5 # 前5个相似数据
        # 使用转onnx时保存的固定数据
        # 获取当前脚本文件的绝对路径
        script_path = os.path.abspath(__file__)
        # 获取当前脚本所在的目录
        script_dir = os.path.dirname(script_path)
        self.text_projection = np.load(os.path.join(script_dir, '../../models/text_projection_512_512.npy'))

        # self.logit_scale = torch.tensor(4.605170249938965)

        # init preprocess
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

        self.encode_image_time = 0.0
        self.encode_text_time = 0.0
        self.preprocess_time = 0.0

    def softmax(self, x, axis=None):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def topk(self, x, k):
        indices = np.argpartition(x, -k)[-k:]
        indices = indices[np.argsort(-x[indices])]
        return x[indices], indices

    def encode_text(self, text):
        start_time = time.time()
        text_batch = text.shape[0]
        processed_outputs = []
        if text_batch > self.text_net_batch_size:
            for start_idx in range(0, text_batch, self.text_net_batch_size):
                end_idx = min(start_idx + self.text_net_batch_size, text_batch)  # Ensure end_idx does not exceed text_batch
                batch_slice = text[start_idx:end_idx]
                if batch_slice.shape[0] < self.text_net_batch_size:
                    padding_size = self.text_net_batch_size - batch_slice.shape[0]
                    batch_slice = np.concatenate([batch_slice, np.zeros((padding_size, *batch_slice.shape[1:]), dtype=batch_slice.dtype)], axis=0)
                input_data = {self.text_net_input_name: batch_slice}
                results = self.text_net.process(self.text_net_graph_name, input_data)[self.text_net_output_name]
                processed_outputs.append(results)
        else:
            padding_text = None
            if text_batch < self.text_net_batch_size:
                padding_size = self.text_net_batch_size - text_batch
                padding_text = np.concatenate([text, np.zeros((padding_size, *text.shape[1:]), dtype=text.dtype)], axis=0)
            else:
                padding_text = text
            input_data = {self.text_net_input_name: padding_text}
            results = self.text_net.process(self.text_net_graph_name, input_data)[self.text_net_output_name]
            processed_outputs.append(results)

        processed_outputs = np.concatenate(processed_outputs, axis=0)[:text_batch]  # Trim padding off the final output if it was padded
        processed_outputs = np.dot(processed_outputs[np.arange(processed_outputs.shape[0]), text.argmax(axis=-1)], self.text_projection)
        self.encode_text_time += time.time() - start_time
        return processed_outputs
