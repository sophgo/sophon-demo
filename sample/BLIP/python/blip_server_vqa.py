#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
from queue import Queue
import sophon.sail as sail
import os

import threading
from blip import init_tokenizer, preprocess


# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录
script_dir = os.path.dirname(script_path)
# 设置参数
config_default = {
    "venc_bmodel_path":os.path.join(script_dir,"../models/BM1684X/blip_vqa_venc_bm1684x_f32_1b.bmodel"),
    "tenc_bmodel_path":os.path.join(script_dir,"../models/BM1684X/blip_vqa_tenc_bm1684x_f32_1b.bmodel"),
    "tdec_bmodel_path":os.path.join(script_dir,"../models/BM1684X/blip_vqa_tdec_bm1684x_f32_1b.bmodel"),
    "tokenizer_path":  os.path.join(script_dir,"../models/bert-base-uncased"),
    "dev_id":0,
    "input_image_queue_size": 20,
    "result_queue_size": 20
}

class BlipServer:
    def __init__(self, config=config_default):
        """
        初始化blip_engine, input_image_queue
        """
        # 初始化blip
        self.net_venc = sail.Engine(config['venc_bmodel_path'], config['dev_id'], sail.IOMode.SYSIO)
        self.net_tenc = sail.Engine(config['tenc_bmodel_path'], config['dev_id'], sail.IOMode.SYSIO)
        self.net_tdec = sail.Engine(config['tdec_bmodel_path'], config['dev_id'], sail.IOMode.SYSIO)
        self.tokenizer = init_tokenizer(config['tokenizer_path'])
        """
        input_image_queue内部是多个dict, 用id确保每个数据项在队列中都有唯一的标识符
        {
            "id": int,
            "image": numpy.ndarray,
            "text": list[str]
        }
        """
        self.input_image_queue = Queue(config["input_image_queue_size"])    # 后端服务器调用此接口，传入图片队列
        
        """
        result_queue内部是多个dict, 用id确保每个数据项在队列中都有唯一的标识符
        TODO: 这里不涉及到batch_size, 后续可以考虑添加
        {
            "id": int,
            "image": numpy.ndarray,
            "texts": list[str],
            "similarity": numpy.ndarray
        }
        """
        self.result_queue = Queue(config["result_queue_size"])    # 后端服务器调用此接口，传入结果队列
        
        # 线程标志位
        self.running = True
        self.existing_ids = set()  # 用于存储已存在的id
        # 启动线程
        self.predict_thread = threading.Thread(target=self.predict, args=())
        self.predict_thread.daemon = True # 设置为守护线程，主线程结束时，子线程也结束
        self.predict_thread.start()


    # 这个函数不需要放入线程中，因为已经推理是异步的了
    def push_image(self, data: dict):
        """
        后端服务器调用此接口，传入数据队列
        data: dict
            {
                "id": int,
                "image": numpy.ndarray,
                "text": list[str]
            }
        """
        if data["id"] in self.existing_ids:
            raise ValueError(f"ID {data['id']} already exists in the queue.")
        
        self.existing_ids.add(data["id"])
        self.input_image_queue.put(data)    # 非pipeline预测，直接调用demo
        print("==================================================")
        print("push success,id:",data["id"],data["texts"])
        print("==================================================")

    def predict(self):
        """
        这个推理函数将被当前class的init线程调用,修改input_image_queue、result_queue, 执行预测
        """
        while self.running:
            try:
                # get data ，队列为空则线程阻塞
                data = self.input_image_queue.get()
                print("==================================================")
                print("predict success,id:",data["id"],data["texts"])
                print("==================================================")
                # self.existing_ids.remove(data["id"])  # 从集合中移除已处理的id
                # Tokenize
                text_inputs = self.tokenizer(data["texts"], padding='max_length', truncation=True, max_length=35,return_tensors="np")
                # Preprocess
                image_input = preprocess(data["image"], image_size = 480)
		        # image encoding
                image_output = self.net_venc.process("blip_vqa_venc", {"pixel_values":image_input})
                if "output_LayerNormalization_f32" in image_output:
                        image_embeds = image_output['output_LayerNormalization_f32']
                else:
                        image_embeds = image_output['output_LayerNormalization']
		        # text encoding
                all_texts = []
                for i in range(text_inputs['input_ids'].shape[0]):
                    tenc_inputs = {
                            "image_embeds": image_embeds,
                            "input_ids": text_inputs['input_ids'][i:i+1],
                            "attention_mask": text_inputs['attention_mask'][i:i+1]
                    }
                    question_states = self.net_tenc.process("blip_vqa_tenc", tenc_inputs)
                    if 'output_LayerNormalization_f32' in question_states:
                            question_states_in = question_states['output_LayerNormalization_f32']
                    else:
                            question_states_in = question_states['output_LayerNormalization']
                    # Predict
                    outputs = self.net_tdec.process("blip_vqa_tdec", {"question_states":question_states_in})
                    if 'output_Concat_f32' in outputs:
                            answer = outputs['output_Concat_f32']
                    else:
                            answer = outputs['output_Concat']
                    texts = self.tokenizer.decode(answer[0].astype('int64'), skip_special_tokens=True)
                    all_texts.append(texts)
                # push result 
                self.result_queue.put({
                    "id": data["id"],
                    "question": data["texts"],
                    # "image": data["image"],
                    "texts": all_texts,
                })
                print("predict success,id:",data["id"])
            except Exception as e:
                print("predict error,id:",data["id"])
                print(e)

    # 返回结果
    def get_batch_result(self):
        if self.result_queue.empty():
            return None
        return self.result_queue.get()

    def stop(self):
        self.running = False
        self.predict_thread.join()
