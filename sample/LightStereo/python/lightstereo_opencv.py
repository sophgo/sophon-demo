#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import json
import time
import cv2
import argparse
import numpy as np
import sophon.sail as sail
import logging
logging.basicConfig(level=logging.INFO)
# sail.set_print_flag(1)

# import sys
# sys.path.append("../tools/")
# from core.lightstereo import LightStereo
# import core.common_utils as common_utils
# import torch
# torch.cuda.set_device(0)
# common_utils.set_random_seed(0)

class LightStereoInfer:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name_left = self.net.get_input_names(self.graph_name)[0]
        self.input_name_right = self.net.get_input_names(self.graph_name)[1]
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name_left)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        # self.model = LightStereo(MAX_DISP=192, LEFT_ATT=True, AGGREGATION_BLOCKS=(1,2,4), EXPANSE_RATIO=4)
        # self.model = self.model.to(0)
        # common_utils.load_params_from_file(
        #         self.model, "../models/ckpt/LightStereo-S-SceneFlow.ckpt", device='cuda:0',
        #         dist_mode=False, logger=None, strict=False)
        # self.model.eval()
        
    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
    
    def rightTopPad(self, image):
        h, w = image.shape[:2]
        th, tw = self.net_h, self.net_w
        h = min(h, th)  # ensure h is within the bounds of the image
        w = min(w, tw)  # ensure w is within the bounds of the image

        pad_left = 0
        pad_right = tw - w
        pad_top = th - h
        pad_bottom = 0
        pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        image = np.pad(image, pad_width, 'edge')

        return image
    
    def preprocess(self, ori_img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        if ori_img.shape[0] <= self.net_h and ori_img.shape[1] <= self.net_w:
            img = self.rightTopPad(ori_img)
        else:
            img = cv2.resize(ori_img, (self.net_w, self.net_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img = (img/255-self.mean)/self.std
        img = np.transpose(img, (2, 0, 1))
        return img
    
    def predict(self, left_imgs, right_imgs, img_num):
        input_data = {self.input_name_left: left_imgs,
                      self.input_name_right: right_imgs}
        outputs = self.net.process(self.graph_name, input_data)[self.output_name]
        # left_tensor = torch.Tensor(left_imgs).to(0)
        # right_tensor = torch.Tensor(right_imgs).to(0)
        # with torch.cuda.amp.autocast(enabled=True):
        #     outputs = self.model(left_tensor, right_tensor)
        #     outputs = outputs.cpu().detach().numpy()
        out = outputs[:img_num]
        return out
    
    def __call__(self, left_imgs, right_imgs):
        assert len(left_imgs) == len(right_imgs), "left and right images must have the same amount!"
        img_num = len(left_imgs)
        preprocessed_left_img_list = []
        preprocessed_right_img_list = []
        start_time = time.time()
        for left_img, right_img in zip(left_imgs, right_imgs):
            assert left_img.shape == right_img.shape, "left and right image must have the same shape!"
            preprocessed_left_img = self.preprocess(left_img)
            preprocessed_right_img = self.preprocess(right_img)
            preprocessed_left_img_list.append(preprocessed_left_img)
            preprocessed_right_img_list.append(preprocessed_right_img)
        if img_num == self.batch_size:
            input_left_imgs = np.stack(preprocessed_left_img_list)
            input_right_imgs = np.stack(preprocessed_right_img_list)
        else:
            input_left_imgs = np.zeros(self.input_shape, dtype='float32')
            input_right_imgs = np.zeros(self.input_shape, dtype='float32')
            input_left_imgs[:img_num] = np.stack(preprocessed_left_img_list)
            input_right_imgs[:img_num] = np.stack(preprocessed_right_img_list)

        self.preprocess_time += time.time() - start_time
            
        start_time = time.time()
        outputs = self.predict(input_left_imgs, input_right_imgs, img_num)
        self.inference_time += time.time() - start_time
        results = []
        start_time = time.time()
        for i in range(img_num):
            ori_h = left_imgs[i].shape[0]
            ori_w = left_imgs[i].shape[1]
            if ori_h <= self.net_h and ori_w <= self.net_w:
                #crop
                th = self.net_h - ori_h
                results.append(outputs[i][th:self.net_h,:ori_w])
            else:
                #resize
                results.append(cv2.resize(outputs[i], (ori_w, ori_h)))
        self.postprocess_time += time.time() - start_time

        return results
   
def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 
    
    # initialize net
    lightstereo = LightStereoInfer(args)
    batch_size = lightstereo.batch_size
    
    # read image paths
    left_img_paths = []
    right_img_paths = []
    with open(args.input, 'r') as f:
        for line in f.readlines():
            dir_path = os.path.dirname(args.input)
            left_img_path = os.path.join(dir_path, line.split(" ")[0])
            right_img_path = os.path.join(dir_path, line.split(" ")[1])
            left_img_paths.append(left_img_path)
            right_img_paths.append(right_img_path)
    
    decode_time = 0.0
    # test images
    cn = 0
    left_imgs_list = []
    right_imgs_list = []
    left_filename_list = []
    right_filename_list = []
    for left_img_path, right_img_path in zip(left_img_paths, right_img_paths):
        cn += 1
        left_img_name = os.path.split(left_img_path)[-1]
        right_img_name = os.path.split(right_img_path)[-1]
        logging.info("{}, left_image: {}, right_image: {}.".format(cn, left_img_name, right_img_name))
        
        # decode
        start_time = time.time()
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)

        if left_img is None or right_img is None:
            logging.error("Meet an error when decoding, skipping this pair of images.")
            continue
        if len(left_img.shape) != 3:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        if len(right_img.shape) != 3:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
        decode_time += time.time() - start_time
                
        left_imgs_list.append(left_img)
        right_imgs_list.append(right_img)
        left_filename_list.append(left_img_name)
        right_filename_list.append(right_img_name)
        if (len(left_imgs_list) == batch_size or cn == len(left_img_paths)) and len(left_imgs_list):
            # predict
            results = lightstereo(left_imgs_list, right_imgs_list)
            for i in range(len(left_imgs_list)):
                disp_img = results[i]

                # save image
                cv2.imwrite(os.path.join(output_img_dir, left_filename_list[i]), disp_img)
            
            left_imgs_list.clear()
            right_imgs_list.clear()
            left_filename_list.clear()
            right_filename_list.clear()
        
    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = lightstereo.preprocess_time / cn
    inference_time = lightstereo.inference_time / cn
    postprocess_time = lightstereo.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    # average_latency = decode_time + preprocess_time + inference_time + postprocess_time
    # qps = 1 / average_latency
    # logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/KITTI12/kitti12_train194.txt', help='path of input txt')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/LightStereo-S-SceneFlow_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')