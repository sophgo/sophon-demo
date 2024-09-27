# -*- coding: utf-8 -*-
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import logging
import sophon.sail as sail
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class EngineOV:
    def __init__(self, model_path="./models/bert_model/bge_large_512_fp16.bmodel", device_id=0) :
        self.net = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
        logging.info("load {} success, dev_id {}".format(model_path, device_id))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)


    def __call__(self, input_ids, attention_mask, token_type_ids=None):
        input_data = {self.input_names[0]: input_ids,
                      self.input_names[1]: attention_mask}
        if token_type_ids is not None:
            input_data[self.input_names[2]] = token_type_ids
        outputs = self.net.process(self.graph_name, input_data)
        return  outputs[self.output_names[0]]

