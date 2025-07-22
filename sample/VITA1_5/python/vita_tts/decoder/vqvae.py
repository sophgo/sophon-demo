#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sophon.sail as sail
import numpy as np
class VQVAE:
    def __init__(self, args):
        self.dev_id = args.dev_id
        self.handle = sail.Handle(self.dev_id)
        self.engine = sail.Engine(args.vqvae_model_path, self.dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_names = self.engine.get_input_names(self.graph_name)
        self.feat_len = self.engine.get_input_shape(self.graph_name, self.input_names[0])[1]
        self.output_names = self.engine.get_output_names(self.graph_name)
        self.global_tokens = np.array([[[473,975,419,219,565,121,550,616]]], dtype=np.int32)
    def __call__(self, x, global_style_token):
        input_tensors = {self.input_names[0]: x,
                         self.input_names[1]: global_style_token}
        outputs = self.engine.process(self.graph_name, input_tensors)
        return outputs[self.output_names[0]]