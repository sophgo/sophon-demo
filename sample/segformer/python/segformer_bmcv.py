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
import json
import cv2
import numpy as np
import argparse
import glob
import sophon.sail as sail
import logging
logging.basicConfig(level=logging.INFO)
import copy
import shutil
import sys
sys.path.append("tools")
import custom as dp

import os.path as osp


SELECT_NUMPY=False

class SegFormer(object):
    def __init__(self, args):

        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))
        # self.handle = self.net.get_handle()
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)

        # load bmodel
        self.graph_name = self.net.get_graph_names()[0]
        # get input
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype= self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_name, self.input_shapes)))
        
        # get output
        self.output_names = self.net.get_output_names(self.graph_name)
      
        self.output_name = self.output_names[0]
        self.output_scale = self.net.get_output_scale(self.graph_name, self.output_name)
        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_name)
        self.output_tensor = sail.Tensor(self.handle, self.output_shape, self.output_dtype,  True, True)
        self.output_tensors = {self.output_name: self.output_tensor}

        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_names, self.output_shape)))
        
   
        # 用于normalize
        #self.input_scale = float(1.0)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.use_resize_padding = True
        self.use_vpp = False
        self.resize_imgs=[]

        # 修改图片
        # 模型输入大小
        self.net_sacle=(1024,512)
        self.flip=False
        self.flip_direction="horizontal"
        self.keep_ratio=True
        self.to_rgb=True

        # 归一化处理
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
        self.size_divisor=32
        # y=ax+b
        self.a = [1/x for x in self.std]
        self.b = [-x/y for x,y in zip(self.mean, self.std)]

        self.ab = []
        for i in range(3):
            self.ab.append(self.a[i]*self.input_scale)
            self.ab.append(self.b[i]*self.input_scale)

        # 时间计算
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time=0.0
        self.decode_time=0.0
    # a=self.show_img(rgb_planar_img)

    
    def preprocess_bmcv(self, input_bmimg):
        rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                          sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)
        
        img_w = rgb_planar_img.width()
        img_h = rgb_planar_img.height()
        if self.use_resize_padding:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            if r_h > r_w:
                tw = self.net_w
                th = int(r_w * img_h)
                tx1 = tx2 = 0
                ty1 = int((self.net_h - th) / 2)
                ty2 = self.net_h - th - ty1
            else:
                tw = int(r_h * img_w)
                th = self.net_h
                tx1 = int((self.net_w - tw) / 2)
                tx2 = self.net_w - tw - tx1
                ty1 = ty2 = 0

            ratio = (min(r_w, r_h), min(r_w, r_h))
            txy = (tx1, ty1)
            attr = sail.PaddingAtrr()
            attr.set_stx(tx1)
            attr.set_sty(ty1)
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)
            
            preprocess_fn = self.bmcv.vpp_crop_and_resize_padding if self.use_vpp else self.bmcv.crop_and_resize_padding
            resized_img_rgb = preprocess_fn(rgb_planar_img, 0, 0, img_w, img_h, self.net_w, self.net_h, attr)
        else:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            ratio = (r_w, r_h)
            txy = (0, 0)
            preprocess_fn = self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
            resized_img_rgb = preprocess_fn(rgb_planar_img, self.net_w, self.net_h)

        # self.bmcv.imwrite("resize.png", resized_img_rgb);    
        # output_bmimg=self.ab[0]*resize_bmimg_rgb+self.ab[1]
        self.resize_imgs.append(resized_img_rgb)

        output_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, \
                        sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(resized_img_rgb, output_bmimg, \
                                       ((self.ab[0], self.ab[1]), \
                                       (self.ab[2], self.ab[3]), \
                                       (self.ab[4], self.ab[5])))
        return output_bmimg
        

    def predict(self, input_tensor):
        input_tensors = {self.input_name: input_tensor}
        self.net.process(self.graph_name, input_tensors, self.output_tensors)
        outputs = self.output_tensor.asnumpy() * self.output_scale
        return outputs[0]

    def postprocess(self,bmimg_list, result):
        palette=dp.get_palette(args.palette)
        output_bmimgs=[]
        for i, input_img in enumerate(bmimg_list):
            color_seg=dp.palette_bmcv(result[i],palette)
            if SELECT_NUMPY:
                #更换颜色通道
                resize_input_rgb=self.bmcv.bm_image_to_tensor(input_img).asnumpy()
                # 删除第一个维度
                resize_input_rgb = np.squeeze(resize_input_rgb)
                # 更换形状使与color_seg一致
                resize_input_rgb=resize_input_rgb.transpose(1,2,0)
            
                output_bmimg=color_seg*0.5+resize_input_rgb*0.5
            else :
                # # 第二种方式k
                color_seg_transpose=color_seg.transpose(2,0,1)
                # 形状为：[1, 3, 512, 1024]
                color_seg_expand = np.expand_dims(color_seg_transpose, axis=0)
                
                color_seg_tensor=sail.Tensor(self.handle,color_seg_expand,False)
                color_seg_bmimg=self.bmcv.tensor_to_bm_image(color_seg_tensor)
                color_seg_rgb= sail.BMImage(self.handle, color_seg_bmimg.height(), color_seg_bmimg.width(),
                                            sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
                self.bmcv.convert_format(color_seg_bmimg, color_seg_rgb)
               
                output_bmimg = sail.BMImage()
                # 图片叠加
                self.bmcv.image_add_weighted(color_seg_rgb, float(0.5), input_img, float(0.5), float(0.0),output_bmimg)
                
            output_bmimgs.append(output_bmimg)

        return output_bmimgs

    def __call__(self, bmimg_list):
        img_num = len(bmimg_list)
        if self.batch_size == 1:
            preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, \
                        sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
    
            start_time = time.time()
            preprocessed_bmimg= self.preprocess_bmcv(bmimg_list[0])
            self.preprocess_time += time.time() - start_time

            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimg, input_tensor)  
            
            start_time = time.time()
            outputs = [self.predict(input_tensor)]
            self.inference_time += time.time() - start_time

        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                start_time = time.time()
                preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, \
                        sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
                preprocessed_bmimg = self.preprocess_bmcv(bmimg_list[i])
                self.preprocess_time += time.time() - start_time
                bmimgs[i] = preprocessed_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)
            
            start_time = time.time()
            outputs = self.predict(input_tensor)[:img_num]
            self.inference_time += time.time() - start_time
        
        start_time = time.time()
        color_seg_img = self.postprocess(self.resize_imgs,outputs)
        self.postprocess_time += time.time() - start_time
        self.resize_imgs=[]
        
        return outputs,color_seg_img

       
    def get_time(self):
        return self.dt

def get_image_files_recursive(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))
    return image_files


data = dict(
        data_root='datasets/cityscapes_small',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        img_suffix='_leftImg8bit.png',
        seg_map_suffix='_gtFine_labelTrainIds.png')


def main(args):
    # creat save path
    output_dir = "python/results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 

    # initialize net
    segformer = SegFormer(args)
    batch_size=segformer.batch_size;
    
    decode_time = 0.0
    # test images
    if os.path.isdir(args.input): 
        # creat save path
        res_dir = "datasets/result_cl"
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
            
        data["data_root"]=args.input
        img_dir = osp.join(data["data_root"], data["img_dir"])
        ann_dir = osp.join(data["data_root"], data["ann_dir"])
        
        image_files = get_image_files_recursive(img_dir)

        result_jsons={}
        result_jsons["data_root"]=data["data_root"]
        result_jsons["img_dir"]=img_dir
        result_jsons["ann_dir"]=ann_dir

        img_info=[]
        bmimg_list = []
        filename_list = []
        filepath_list = []
        frame_num = 0
        img_num = len(image_files)

        for filepath in image_files:  
            filename=os.path.basename(filepath)
            # 解码
            logging.info("{}, img_file: {}".format(frame_num, filename))
            # decode
            start_time = time.time()
            decoder = sail.Decoder(filepath, True, args.dev_id)
            bmimg = sail.BMImage()
            ret = decoder.read(segformer.handle, bmimg)    
            # print(bmimg.format(), bmimg.dtype())
            if ret != 0:
                logging.error("{} decode failure.".format(filepath))
                continue
            decode_time += time.time() - start_time

            frame_num += 1
            bmimg_list.append(bmimg)
            filename_list.append(filename)
            filepath_list.append(filepath)

            if len(bmimg_list) != batch_size and frame_num != (img_num - 1):
                continue
            if (frame_num % batch_size == 0 or flag == False) and len(bmimg_list):
                # 解决多batch，数据集不能被batch整除情况
                if(frame_num % batch_size == 0):
                    record_size=batch_size
                else:
                    record_size=frame_num % batch_size

                while(frame_num % batch_size != 0):
                    frame_num += 1
                    bmimg_list.append(bmimg)
                    filename_list.append(filename)
                    filepath_list.append(filepath)

                output, seg= segformer(bmimg_list)
                for i in range(record_size):
                    filename=filename_list[i]
                    result_json={}
                    filepath=filepath_list[i]
                    filename_no_ext = os.path.splitext(filename)[0]
                    res_file =f'{filename_no_ext}.png' # 构造保存的np文件名
                    res_file = os.path.join(res_dir, res_file)
                    result_json["res"] = res_file  # 将文件路径存储在字典中
                        # 将 NumPy 数组转换为 cv::Mat 格式
                    v_mat = cv2.UMat(output[i][0,:,:])
                    cv2.imwrite(res_file,v_mat)

                    relative_path = os.path.relpath(filepath, img_dir)
                    result_json["filename"]=relative_path                   
                    seg_map = relative_path.replace(data["img_suffix"], data["seg_map_suffix"])
                    result_json["ann"]={"seg_map":seg_map}
                        # save image   
                    if SELECT_NUMPY:
                        cv2.imwrite(os.path.join(output_img_dir, filename), seg[i])
                    else:
                        segformer.bmcv.imwrite(os.path.join(output_img_dir, filename), seg[i])
                    img_info.append(result_json)                      
                bmimg_list.clear()
                filename_list.clear()
                filepath_list.clear()
        result_jsons["img_info"]=img_info
       # save result
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_bmcv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            # json.dump(results_list, jf)
            json.dump(result_jsons, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))
    # test video
    else:
        video_output_dir = 'python/results/video'
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
        else:
            shutil.rmtree(video_output_dir)
            os.makedirs(video_output_dir)
        cap = cv2.VideoCapture()
        if not cap.open(args.input):
            raise Exception("can not open the video")
        frame_list = []
        frame_num = 0
        flag = True
        while flag:
            frame = sail.BMImage()
            start_time = time.time()
            ret = decoder.read(segformer.handle, frame)
            decode_time += time.time() - start_time
            if ret:
                flag = False
            else:
                frame_list.append(frame)
                frame_num += 1
            if (frame_num % batch_size == 0 or flag == False) and len(frame_list):
                # 解决多batch，数据集不能被batch整除情况
                if(frame_num % batch_size == 0):
                    record_size=batch_size
                else:
                    record_size=frame_num % batch_size

                while(frame_num % batch_size != 0):
                    frame_num += 1
                    frame_list.append(bmimg)
   
                output, seg= segformer(frame_list)
                for i, in range(record_size):
                    save_name = os.path.join(video_output_dir, str(frame_num - len(frame_list) + i + 1))
                    # save image   
                    if SELECT_NUMPY:
                        cv2.imwrite(save_name+".jpg",seg[i])
                        print("frame:"+str(frame_num)+" success !!!")
                    else:
                        segformer.bmcv.imwrite(save_name+".jpg", seg[i])
                        print("frame:"+str(frame_num)+" success !!!")
                frame_list.clear()
        logging.info("result saved in {}".format(output_dir))
        decoder.release()
    
    logging.info("------------------ Inference Time Info ----------------------")
    decode_time = decode_time / frame_num
    preprocess_time = segformer.preprocess_time / frame_num
    inference_time = segformer.inference_time / frame_num
    postprocess_time = segformer.postprocess_time / frame_num
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
        
def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='datasets/cityscapes_small', help='path of input, must be image directory')
    parser.add_argument('--bmodel', type=str, default='models/BM1684/segformer.b0.512x1024.city.160k_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argsparser()
    main(args)
