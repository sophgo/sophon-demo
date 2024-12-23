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
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name_left = self.net.get_input_names(self.graph_name)[0]
        self.input_name_right = self.net.get_input_names(self.graph_name)[1]
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name_left)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name_left)
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_name)
        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)

        # init bmcv for preprocess
        self.handle = self.net.get_handle()
        self.bmcv = sail.Bmcv(self.handle)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)

        self.input_scale_left = self.net.get_input_scale(self.graph_name, self.input_name_left)
        self.input_scale_right = self.net.get_input_scale(self.graph_name, self.input_name_left)
        self.output_scale = self.net.get_output_scale(self.graph_name, self.output_name)

        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

        self.a = [1/(255.*x) for x in self.std]
        self.b = [-x/y for x,y in zip(self.mean, self.std)]

        self.ab_left = []
        self.ab_right = []
        for i in range(3):
            self.ab_left.append(self.a[i]*self.input_scale_left)
            self.ab_left.append(self.b[i]*self.input_scale_left)
            self.ab_right.append(self.a[i]*self.input_scale_right)
            self.ab_right.append(self.b[i]*self.input_scale_right)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

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
        
    def preprocess_bmcv(self, input_bmimg, output_bmimg, is_left=True):
        if input_bmimg.format()==sail.Format.FORMAT_YUV420P:
            input_bmimg_bgr = self.bmcv.yuv2bgr(input_bmimg)
        else:
            input_bmimg_bgr = input_bmimg
        if input_bmimg_bgr.height() <= self.net_h and input_bmimg_bgr.width() <= self.net_w:
            resize_bmimg = self.rightTopPad(input_bmimg_bgr)
        else:
            resize_bmimg = self.bmcv.resize(input_bmimg_bgr, self.net_w, self.net_h, sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)
        resize_bmimg_rgb = sail.BMImage(self.handle, self.net_h, self.net_w, \
                        sail.Format.FORMAT_RGB_PLANAR, resize_bmimg.dtype())
        self.bmcv.convert_format(resize_bmimg, resize_bmimg_rgb)
        
        if is_left:
            ab = self.ab_left
        else:
            ab = self.ab_right
        self.bmcv.convert_to(resize_bmimg_rgb, output_bmimg, ((ab[0], ab[1]), \
                                                             (ab[2], ab[3]), \
                                                             (ab[4], ab[5])))
    
    def rightTopPad(self, image):
        h, w = image.height(), image.width()
        th, tw = self.net_h, self.net_w
        h = min(h, th)  # ensure h is within the bounds of the image
        w = min(w, tw)  # ensure w is within the bounds of the image

        pad_top = th - h
        image_padded = sail.BMImage(self.handle, self.net_h, self.net_w, image.format(), image.dtype())
        self.bmcv.image_copy_to_padding(image, image_padded, 0, 0, 0, 0, pad_top)
        return image_padded
    
    def predict(self, input_tensor_left, input_tensor_right, img_num):
        input_tensors = {self.input_name_left: input_tensor_left,
                         self.input_name_right: input_tensor_right}

        output_tensor = sail.Tensor(self.handle, self.output_shape, self.output_dtype, True, True)
        output_tensors = {self.output_name: output_tensor}

        self.net.process(self.graph_name, input_tensors, output_tensors)
        outputs = output_tensor.asnumpy() * self.output_scale
        
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
        input_tensor_left = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False) #let tensor to attach BMImage's device memory
        input_tensor_right = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False) #let tensor to attach BMImage's device memory
        start_time = time.time()
        if self.batch_size == 1:
            left_img = left_imgs[0]
            right_img = right_imgs[0]
            l_shape = [left_img.width(), left_img.height()]
            r_shape = [right_img.width(), right_img.height()]
            assert l_shape == r_shape, "left and right image must have the same shape!"
            preprocessed_left_img = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
            self.preprocess_bmcv(left_img, preprocessed_left_img, True)
            preprocessed_right_img = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
            self.preprocess_bmcv(right_img, preprocessed_right_img, False)
            self.bmcv.bm_image_to_tensor(preprocessed_left_img, input_tensor_left)
            self.bmcv.bm_image_to_tensor(preprocessed_right_img, input_tensor_right)
        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            left_bmimgs = BMImageArray()
            right_bmimgs = BMImageArray()
            for left_img, right_img in zip(left_imgs, right_imgs):
                l_shape = [left_img.width(), left_img.height()]
                r_shape = [right_img.width(), right_img.height()]
                assert l_shape == r_shape, "left and right image must have the same shape!"
                preprocessed_left_img = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
                self.preprocess_bmcv(left_img, preprocessed_left_img, True)
                preprocessed_right_img = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
                self.preprocess_bmcv(right_img, preprocessed_right_img, False)
                left_bmimgs[i] = preprocessed_left_img.data()
                right_bmimgs[i] = preprocessed_right_img.data()
            self.bmcv.bm_image_to_tensor(left_bmimgs, input_tensor_left)
            self.bmcv.bm_image_to_tensor(right_bmimgs, input_tensor_right)
        self.preprocess_time += time.time() - start_time
            
        start_time = time.time()
        outputs = self.predict(input_tensor_left, input_tensor_right, img_num)
        self.inference_time += time.time() - start_time
        results = []
        start_time = time.time()
        for i in range(img_num):
            ori_h = left_imgs[i].height()
            ori_w = left_imgs[i].width()
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
        decoder = sail.Decoder(left_img_path, True, args.dev_id)
        left_img = sail.BMImage()
        ret = decoder.read(lightstereo.handle, left_img)    
        if ret != 0:
            logging.error("{} decode failure.".format(left_img_path))
            continue
        decoder = sail.Decoder(right_img_path, True, args.dev_id)
        right_img = sail.BMImage()
        ret = decoder.read(lightstereo.handle, right_img)    
        if ret != 0:
            logging.error("{} decode failure.".format(right_img_path))
            continue
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