import os
import logging
from transformers import  BertConfig, BertTokenizer
import sophon.sail as sail
import numpy as np


class TTSProsody(object):
    def __init__(self, bert_model = "./bmodel/bert_1684x_f32.bmodel", dev_id = 0):
        # use dynamic bert
        self.net = sail.Engine(bert_model, dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(bert_model))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_names[0])
        self.max_length = self.input_shape[1]

        self.tokenizer = BertTokenizer.from_pretrained("./python/bert")
        self.bert_config = BertConfig.from_pretrained("./python/bert")

    def text2Token(self, text):
        token = self.tokenizer.tokenize(text)
        txtid = self.tokenizer.convert_tokens_to_ids(token)
        return txtid

    def get_char_embeds(self, text):
        input_ids = self.text2Token(text)
        assert len(input_ids) <= self.max_length
        input_masks = [1] * len(input_ids)
        type_ids = [0] * len(input_ids)
        input_ids_array = np.expand_dims(np.array(input_ids, dtype=np.int32), axis=0)
        input_masks_array = np.expand_dims(np.array(input_masks, dtype=np.int32), axis=0)
        type_ids_array = np.expand_dims(np.array(type_ids, dtype=np.int32), axis=0)
        

        # 填充函数
        def pad_to_length(array, max_length):
            return np.pad(array, ((0, 0), (0, self.max_length - array.shape[1])), mode='constant', constant_values=0)

        # 填充到 max_length 长度
        padded_input_ids = pad_to_length(input_ids_array, self.max_length)
        padded_input_masks = pad_to_length(input_masks_array, self.max_length)
        padded_type_ids = pad_to_length(type_ids_array, self.max_length)

        input_data = {self.input_names[0]: padded_input_ids,
                      self.input_names[1]: padded_input_masks,
                      self.input_names[2]: padded_type_ids}

        # np.savez("bert.npz", **input_data)
        output_data = self.net.process(self.graph_name, input_data)
        char_embeds = output_data[self.output_names[0]].squeeze(0)

        return char_embeds

    def expand_for_phone(self, char_embeds: np.ndarray, length):  # length of phones for char
        assert char_embeds.shape[0] >= len(length)

        expand_vecs = list()
        # while(sum(length) < self.max_length * 2):
        #     length.append(1)

        for vec, leng in zip(char_embeds, length):
            vec = np.repeat(vec[np.newaxis, ...], leng, axis=0)
            expand_vecs.append(vec)

        expand_embeds = np.concatenate(expand_vecs, 0)
        # Calculate the padding length, self.max_length * 2
        padding_length = self.max_length * 2 - expand_embeds.shape[0]
        padding = np.zeros((padding_length, expand_embeds.shape[1]))
        # Concatenate the original array with the padding
        padded_expand_embeds = np.vstack((expand_embeds, padding))
        assert padded_expand_embeds.shape[0] == self.max_length * 2
        return padded_expand_embeds
