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

import clip
from zeroshot_predict import *
import threading


# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录
script_dir = os.path.dirname(script_path)
# 设置参数
config_default = {
    "image_model":os.path.join(script_dir,"../models/BM1684X/clip_image_vitb32_bm1684x_f16_1b.bmodel"),
    "text_model":os.path.join(script_dir,"../models/BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel"),
    "dev_id":0,
    "input_image_queue_size": 20,
    "result_queue_size": 20
}

class ClipServer:
    def __init__(self, config=config_default):
        """
        初始化clip_engine, input_image_queue
        """
        # 初始化clip_engine, load bmodel, self.clip_engine is (model, preprocess) 
        self.clip_engine, self.preprocess = clip.load(config["image_model"], config["text_model"], config["dev_id"]) 
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
                text_inputs = clip.tokenize(data["texts"])
                # Preprocess
                image_input = np.expand_dims(self.preprocess(data["image"]), axis=0)
                # Predict
                values, indices = self.clip_engine.predict(image_input, text_inputs)
                # Text: {text[indices[i]]}, Similarity: {values[i].item()}
                texts = [data["texts"][index] for index in indices]
                # 将 similarity 转换为 Python 的原生 float 类型
                similarity = [float(value) for value in values]
                print("indices",indices)
                # push result 
                self.result_queue.put({
                    "id": data["id"],
                    # "image": data["image"],
                    "texts": texts,
                    "similarity": similarity
                })
                print("predict success,id:",data["id"])
            except Exception as e:
                print("predict error,id:",data["id"])
                print(e)

    def pipeline_predict(self, image_path, texts):
        # TODO:
        return self.clip.predict(image_path, texts)

    # 返回结果
    def get_batch_result(self):
        if self.result_queue.empty():
            return None
        return self.result_queue.get()

    def stop(self):
        self.running = False
        self.predict_thread.join()
