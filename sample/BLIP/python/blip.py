#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
#!/bin/python3
import sophon.sail as sail
import cv2
import numpy as np
import time

from transformers import BertTokenizerFast

def init_tokenizer(tokenizer_path):
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer

def preprocess(ori_image, image_size):
    raw_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(raw_image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    image_array = np.array(resized_image, dtype=np.float32)

    mean = np.array([0.48145466, 0.4578275, 0.40821073],dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711],dtype=np.float32)
    image_array = (image_array / float(255.0) - mean) / std
    image_array = np.transpose(image_array, (2, 0, 1))
    image = np.expand_dims(image_array, axis=0)
    return image

class blip_itm:
    def __init__(self, args):
        self.net = sail.Engine(args.bmodel_path, args.dev_id, sail.IOMode.SYSIO)
        self.predict_time = 0.0

    def predict(self, pixel_values, text_inputs):
        start_time = time.time()
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        inputs = {
                "pixel_values": pixel_values,
                "input_ids": "",
                "attention_mask": ""
        }
        itm=[]
        item_num = input_ids.shape[0]
        for i in range(item_num):
            inputs["input_ids"] = input_ids[i:i+1]
            inputs["attention_mask"] = attention_mask[i:i+1]
            output = self.net.process("blip_itm", inputs)
            if 'output_Gather_f32' in output:
                itm.append(output['output_Gather_f32'])
            else:
                itm.append(output['output_Gather'])
        self.predict_time += (time.time() - start_time) / item_num
        return itm

class blip_vqa:
    def __init__(self, args):
        self.net_venc = sail.Engine(args.venc_bmodel_path, args.dev_id, sail.IOMode.SYSIO)
        self.net_tenc = sail.Engine(args.tenc_bmodel_path, args.dev_id, sail.IOMode.SYSIO)
        self.net_tdec = sail.Engine(args.tdec_bmodel_path, args.dev_id, sail.IOMode.SYSIO)
        self.image_encode_time = 0.0
        self.predict_time = 0.0

    def image_process(self, pixel_values):
        start_time = 0.0
        venc_inputs = {
                "pixel_values": pixel_values,
        }
        image_embeds = self.net_venc.process("blip_vqa_venc", venc_inputs)
        self.image_encode_time += time.time() - start_time
        if "output_LayerNormalization_f32" in image_embeds:
            return image_embeds['output_LayerNormalization_f32']
        else:
            return image_embeds['output_LayerNormalization']

    def predict(self, image_embeds, text_inputs):
        start_time = time.time()
        tenc_inputs = {
            "image_embeds": image_embeds,
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask
        }
        question_states = self.net_tenc.process("blip_vqa_tenc", tenc_inputs)
        if 'output_LayerNormalization_f32' in question_states:
            question_states_in = question_states['output_LayerNormalization_f32']
        else:
            question_states_in = question_states['output_LayerNormalization']
        tdec_inputs = {
                "question_states":question_states_in,
        }
        outputs = self.net_tdec.process("blip_vqa_tdec", tdec_inputs)
        self.predict_time += time.time() - start_time
        if 'output_Concat_f32' in outputs:
            return outputs['output_Concat_f32']
        else:
            return outputs['output_Concat']

class blip_cap:
    def __init__(self, args):
        self.net = sail.Engine(args.bmodel_path, args.dev_id, sail.IOMode.SYSIO)
        self.predict_time = 0.0

    def predict(self, pixel_values):
        start_time = time.time()
        inputs = {
                "pixel_values": pixel_values,
        }
        output = self.net.process("blip_cap", inputs)
        if "output_Concat_f32" in output:
            outputs = output["output_Concat_f32"]
        else:
            outputs = output["output_Concat"]
        self.predict_time += time.time() - start_time
        return outputs
