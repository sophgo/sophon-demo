import sophon.sail as sail
import numpy as np 
import time 
import os

class EngineOV:
    def __init__(self, model_path="", batch=1, device_id=0) :
        # 如果环境变量中没有设置device_id，则使用默认值
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            print(">>>> device_id is in os.environ. and device_id = ",device_id)
        self.model_path = model_path
        # self.model = SGInfer(model_path , batch=batch, devices=[device_id])
        self.model = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
        self.device_id = device_id
        self.graph_name = self.model.get_graph_names()[0]
        self.input_names = self.model.get_input_names(self.graph_name)
        self.input_shape = self.model.get_input_shape(self.graph_name, self.input_names[0])
    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)
        
    def __call__(self, args):
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
        input_tensors = {}
        for i in range(len(self.input_names)):
            input_tensors[self.input_names[i]] = values[i]
            
        output_tensors = self.model.process(self.graph_name, input_tensors)
        results = list(output_tensors.values())
        return results
