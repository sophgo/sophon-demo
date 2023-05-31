#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# -*- coding: utf-8 -*- 
import os
import cv2
import numpy as np
import argparse
import sophon.sail as sail
import math
import logging
import time
logging.basicConfig(level=logging.DEBUG)

class PPOCRv2Cls(object):
    def __init__(self, args):
        self.cls_thresh = args.cls_thresh
        self.label_list = args.label_list
        # load bmodel
        model_path = args.bmodel_cls
        logging.info("using model {}".format(model_path))
        self.net = sail.Engine(model_path, args.dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        logging.info("load bmodel success!")
        self.input_shape = self.net.get_max_input_shapes(self.graph_name)[self.input_name]
        self.cls_batch_size = self.input_shape[0] # Max batch size in model stages.
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        
    def preprocess(self, img):
        h, w, _ = img.shape
        ratio = w / float(h)
        resized_h = self.input_shape[2]
        new_w = math.ceil(ratio * resized_h)
        if new_w > self.input_shape[3]:
            resized_w = self.input_shape[3]
        else:
            resized_w = new_w
            
        if h != resized_h or w != resized_w:
            img = cv2.resize(img, (resized_w, resized_h))
        img = img.astype('float32')
        img = np.transpose(img, (2, 0, 1))
        img -= 127.5
        img *= 0.0078125

        padding_im = np.zeros((3, resized_h, self.input_shape[3]), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = img
        
        # CHW to NCHW format
        #padding_im = np.expand_dims(padding_im, axis=0)
        # Convert the img to row-major order, also known as "C order":
        #img = np.ascontiguousarray(img)

        return padding_im

    def predict(self, tensor):
        input_data = {self.input_name: np.array(tensor, dtype=np.float32)}
        outputs = self.net.process(self.graph_name, input_data)
        return outputs['softmax_0.tmp_0_Softmax']

    def postprocess(self, outputs):
        #outputs = list(outputs.values())[0]
        pred_idxs = outputs.argmax(axis=1)
        #outputs = np.argmax(outputs, axis = 1)
        res = [(self.label_list[idx], outputs[i, idx])
                      for i, idx in enumerate(pred_idxs)]

        return res

    def __call__(self, img_list):
        img_num = len(img_list)
        img_input_list = []
        
        start_prep = time.time()
        for img in img_list:
            img = self.preprocess(img)
            img_input_list.append(img)
        self.preprocess_time += time.time() - start_prep
        
        start_infer = time.time()
        cls_res = []
        for beg_img_no in range(0, img_num, self.cls_batch_size):
            end_img_no = min(img_num, beg_img_no + self.cls_batch_size)
            if beg_img_no + self.cls_batch_size > img_num:
                for ino in range(beg_img_no, end_img_no):
                    img_input = np.expand_dims(img_input_list[ino], axis=0)
                    outputs = self.predict(img_input)
                    res = self.postprocess(outputs)
                    cls_res.extend(res)
            else:
                img_input = np.stack(img_input_list[beg_img_no:end_img_no])
                outputs = self.predict(img_input)
                res = self.postprocess(outputs)
                cls_res.extend(res)
        self.inference_time += time.time() - start_infer

        start_post = time.time()
        for id, res in enumerate(cls_res):
            if res[0] == '180' and res[1] > self.cls_thresh:
                img_list[id] = cv2.rotate(img_list[id], 1)
        self.postprocess_time += time.time() - start_post
            
        return img_list, cls_res

def main(opt):
    ppocrv2_cls = PPOCRv2Cls(opt)
    Tp = 0
    img_list = []
    for img_name in os.listdir(opt.input):
        #print(file_name)
        #label = img_name.split('.')[0]
        img_file = os.path.join(opt.input, img_name)
        #print(img_file, label)
        src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
        img_list.append(src_img)
    
    img_list, res = ppocrv2_cls(img_list)
    for id, img_name in enumerate(os.listdir(opt.input)):
        logging.info("img_name:{}, pred:{}, conf:{}".format(img_name, res[id][0], res[id][1]))

def list_type(arg):
    return arg.split(',')
def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--dev_id', type=int, default=0, help='tpu card id')
    parser.add_argument('--input', type=str, default='../datasets/cali_set_rec', help='input image directory path')
    parser.add_argument('--bmodel_cls', type=str, default='../models/BM1684X/ch_PP-OCRv3_cls_fp32_1b.bmodel', help='classifier bmodel path')
    parser.add_argument("--cls_thresh", type=float, default=0.9)
    parser.add_argument("--label_list", type=list_type, default="0,180")
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
