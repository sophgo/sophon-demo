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
import time
import numpy as np
import argparse
import glob
import sophon.sail as sail
import logging
logging.basicConfig(level=logging.INFO)
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from PIL import Image
    
class RAM(object):
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.input_shapes = [self.net.get_input_shape(self.graph_name, name) for name in self.input_names]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_shapes = [self.net.get_output_shape(self.graph_name, name) for name in self.output_names]
        logging.debug("load {} success!".format(args.bmodel))
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_names, self.input_shapes)))
        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_names, self.output_shapes)))
        self.input_name = self.input_names[0]
        self.input_shape = self.input_shapes[0]

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        def convert_to_rgb(image):
            return image.convert("RGB")
        def get_transform(net_w, net_h):
            return Compose([
                convert_to_rgb,
                Resize((net_w, net_h)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.transform = get_transform(self.net_w, self.net_h)
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

        self.handle = self.net.get_handle()
        self.bmcv = sail.Bmcv(self.handle)
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_names[0])
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        
        self.tag_list = self.load_tag_list(args.tag_list)
        self.tag_list_chinese = self.load_tag_list(args.tag_list_chinese)
        self.num_class = len(self.tag_list)
        self.class_threshold = np.ones(self.num_class)
        with open(args.tag_list_threshold, 'r', encoding='utf-8') as f:
            ram_class_threshold = [float(s.strip()) for s in f]
        for key,value in enumerate(ram_class_threshold):
            self.class_threshold[key] = value
            
    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, 'r', encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list
    
    def preprocess(self, img):
        image = self.transform(img).numpy()
        return image

    def predict(self, input_img):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)
        return list(outputs.values())[0]

    def postprocess(self, outputs):
        sigmoid_logits = 1 / (1 + np.exp(-outputs))
        tag = np.where(
            sigmoid_logits > self.class_threshold,
            1.0,
            np.zeros(self.num_class))
        tag_output = []
        tag_output_chinese = []
        for b in range(self.batch_size):
            index = np.argwhere(tag[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_output.append(' | '.join(token))
            token_chinese = self.tag_list_chinese[index].squeeze(axis=1)
            tag_output_chinese.append(' | '.join(token_chinese))

        return tag_output, tag_output_chinese

    def __call__(self, img_list):
        img_num = len(img_list)
        img_input_list = []
        for img in img_list:
            start_time = time.time()
            img = self.preprocess(img)
            self.preprocess_time += time.time() - start_time
            img_input_list.append(img)
        
        if img_num == self.batch_size:
            input_img = np.stack(img_input_list)
            start_time = time.time()
            outputs = self.predict(input_img)
            self.inference_time += time.time() - start_time
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(img_input_list)
            start_time = time.time()
            outputs = self.predict(input_img)[:img_num]
            self.inference_time += time.time() - start_time
        
        start_time = time.time()
        res = self.postprocess(outputs)
        self.postprocess_time += time.time() - start_time

        return res

    def get_time(self):
        return self.dt

def main(args):
    ram = RAM(args)
    batch_size = ram.batch_size
    if not os.path.isdir(args.input):
        # logging.error("input must be an image directory.")
        # return 0
        raise Exception('{} is not a directory.'.format(args.input))

    img_list = []
    filename_list = []
    decode_time = 0.0
    cn = 0
    for filename in glob.glob(args.input+'/*'):
        if os.path.splitext(filename)[-1] not in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            continue
        cn += 1
        start_time = time.time()
        src_img = Image.open(filename)
        if src_img is None:
            logging.error("{} imread is None.".format(filename))
            continue
        decode_time += time.time() - start_time
        img_list.append(src_img)
        filename_list.append(filename)
        if (len(img_list) == batch_size or cn == len(filename_list)) and len(img_list):
            tags, tags_ch = ram(img_list)
            for i in range(len(img_list)):
                print(filename_list[i])
                print(tags[i])
                print(tags_ch[i])
            img_list = []
            filename_list = []

    # calculate speed  
    logging.info("------------------ Inference Time Info ----------------------")
    print(cn)
    decode_time = decode_time / cn
    preprocess_time = ram.preprocess_time / cn
    inference_time = ram.inference_time / cn
    postprocess_time = ram.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
        
def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input, must be image directory')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/ram_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    parser.add_argument('--tag_list', type=str, default='../datasets/ram_tag_list.txt', help='path of tag_list')
    parser.add_argument('--tag_list_chinese', type=str, default='../datasets/ram_tag_list_chinese.txt', help='path of tag_list_chinese')
    parser.add_argument('--tag_list_threshold', type=str, default='../datasets/ram_tag_list_threshold.txt', help='path of tag_list_threshold')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)
