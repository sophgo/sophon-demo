#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
from queue import Queue
import sys
import os 
script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录
script_dir = os.path.dirname(script_path)

sys.path.append(os.path.join(script_dir,"sam"))
sys.path.append(os.path.join(script_dir,"groundingdino"))

import numpy as np
from transformers import BertTokenizerFast
from PostProcess import PostProcess as dino_PostProcess
from sam_model import Sam
from PIL import Image
import sophon.sail as sail
import cv2
import threading
import gdsam_util as util
import custom_model


# 设置参数
config_default = {
    "SAM":"../models/BM1684X/decode_bmodel/SAM-ViT-B_decoder_single_mask_fp16_1b.bmodel",
    "SAM_encoder":"../models/BM1684X/embedding_bmodel/SAM-ViT-B_embedding_fp16_1b.bmodel",
    "groudingdion":"../models/BM1684X/groundingdino/groundingdino_bm1684x_fp16.bmodel",
    "tokenizer_path":"../models/bert-base-uncased",
    "dev_id":0,
    "input_image_queue_size": 20,
    "result_queue_size": 20
}

class gdsamServer:
    def __init__(self, config=config_default):
        """
        初始化engine, input_image_queue
        """
        self.dino_engine = custom_model.create_grounding_dino(config_default["groudingdion"], "person", 
                                                   config_default["tokenizer_path"], 0.2, 0.3, 
                                                   config_default["dev_id"], token_spans=None)
        self.sam_vit_b = custom_model.create_SAM_b("0,0,0,0", config_default["SAM"], config_default["dev_id"])
        self.sam = Sam()
        self.sam_encoder = custom_model.create_SamEncoder(config_default["SAM_encoder"], 
                                                          config_default["SAM"], 
                                                          config_default["dev_id"])
        """
        input_image_queue内部是多个dict, 用id确保每个数据项在队列中都有唯一的标识符
        {
            "id": int,
            "image": numpy.ndarray,
            "text_prompt": str
            "box_threshold": float
            "text_threshold": float
        }
        """
        self.input_image_queue = Queue(config["input_image_queue_size"])    # 后端服务器调用此接口，传入图片队列
        
        """
        result_queue内部是多个dict, 用id确保每个数据项在队列中都有唯一的标识符
        TODO: 这里不涉及到batch_size, 后续可以考虑添加
        {
            "id": int,
            "image": numpy.ndarray,
            "pred_phrases": pred_phrases,
            "pred_image": pred_image
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
                "text_prompt": str
                "box_threshold": float
                "text_threshold": float
            }
        """
        if data["id"] in self.existing_ids:
            raise ValueError(f"ID {data['id']} already exists in the queue.")
        
        self.existing_ids.add(data["id"])
        self.input_image_queue.put(data)    # 非pipeline预测，直接调用demo
        print("==================================================")
        print("push success,id:",data["id"])
        print("==================================================")

    def predict(self):
        """
        这个推理函数将被当前class的init线程调用,修改input_image_queue、result_queue, 执行预测
        """
        while self.running:
            try:
                # get data 
                data = self.input_image_queue.get()
                print("==================================================")
                print("predict start,id:",data["id"])
                print("==================================================")
                # self.existing_ids.remove(data["id"])  # 从集合中移除已处理的id

                # Tokenize
                tokenizer = self.dino_engine.tokenizer
                # Preprocess
                resized_image = cv2.resize(data["image"], (800, 800))
                pre_data = self.dino_engine.preprocess(tokenizer, data["text_prompt"], np_image=resized_image)
                
                # dino Inference
                output = self.dino_engine(pre_data)
                # Postprocess
                
                Dino_PostProcess = dino_PostProcess(
                    caption = data["text_prompt"],
                    token_spans = None,
                    tokenizer = tokenizer,
                    box_threshold = data["box_threshold"], 
                    text_threshold = data["text_threshold"]
                )
                boxes_filt, pred_phrases = Dino_PostProcess(output)
                # 坐标绝对值化
                boxes_filt = util.get_grounding_output(data["image"], boxes_filt, pred_phrases)
                
                # SAM Predict
                results = [','.join(map(str, box)) for box in boxes_filt]
                # 循环预测多个box
                masks = []
                mask = np.empty((0,0))
                for result in results:
                    self.sam_vit_b = custom_model.create_SAM_b(result, config_default["SAM"], config_default["dev_id"])
                    # preprocess
                    img = self.sam_vit_b.preprocess(data["image"], self.sam_encoder, self.sam, result)
                    # 预测
                    outputs_0 = self.sam_vit_b.predict(img)
                    sam_results = self.sam_vit_b.postprocess(outputs_0)
                   
                    # 把预测结果mask合并
                    mask = sam_results[0][0]
                    masks.append(mask)
                masks = np.array(masks).reshape(len(results), 1, mask.shape[0], mask.shape[1])       
                pred_image = util.draw_output_image(data["image"], masks, boxes_filt, pred_phrases)
                
                self.result_queue.put({
                        "id": data["id"],
                        "text_prompt": data["text_prompt"],
                        "pred_phrases": pred_phrases,
                        "pred_image":pred_image
                    })
                print("predict success,id:",data["id"])
                
            except Exception as e:
                print("predict error,id:",data["id"])
                print(e)

    def pipeline_predict(self, image_path, texts):
        # TODO:
        return self.gdsam.predict(image_path, texts)

    # 返回结果
    def get_batch_result(self):
        if self.result_queue.empty():
            return None
        return self.result_queue.get()

    def stop(self):
        self.running = False
        self.predict_thread.join()