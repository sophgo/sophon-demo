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
import shutil
import sys
sys.path.append("tools")
import custom as dp

import os.path as osp

RESEZE_ORIGIN=0


class SegFormer(object):
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.debug("load {} success!".format(args.bmodel))

        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.input_shapes = [self.net.get_input_shape(self.graph_name, name) for name in self.input_names]
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_names, self.input_shapes)))
        
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_shapes = [self.net.get_output_shape(self.graph_name, name) for name in self.output_names]
        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_names, self.output_shapes)))
        
        
        self.input_name = self.input_names[0]
        self.input_shape = self.input_shapes[0]

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        # 模型输入大小
        self.net_sacle=(self.net_w, self.net_h)
        self.flip=False
        self.flip_direction="horizontal"
        self.keep_ratio=True
        self.to_rgb=True

        self.resize_imgs=[]
        # 归一化处理
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
        self.size_divisor=32

        # 时间计算
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.decode_time =0.0 
    # align 
    def _align(self, img, size_divisor, interpolation=None):
        align_h = int(np.ceil(img.shape[0] / size_divisor)) * size_divisor
        align_w = int(np.ceil(img.shape[1] / size_divisor)) * size_divisor
        if interpolation == None:
            img = cv2.resize(img, (align_w, align_h))
        else:
            img = cv2.resize(img, (align_w, align_h), interpolation=interpolation)
        return img
    

    def _resize_img(self, results):
        img = results['img']
        # self.keep_ratio=False
        if self.keep_ratio:
            shape = img.shape[:2]
             # Scale ratio (new / old)
            r = min(self.net_sacle[0] / shape[1], self.net_sacle[1] / shape[0])   
            # Compute padding
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dh, dw = self.net_sacle[1] - new_unpad[1], self.net_sacle[0] - new_unpad[0]  # wh padding

            dw /= 2  # divide padding into 2 sides
            dh /= 2
            if shape[::-1] != new_unpad:  # resize
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
            
        else:
            # 使用cv2.resize函数进行图像缩放
            img = cv2.resize(img,self.net_sacle)
          
    
        results['img']=img

        return img 
    
    def _flip(self,results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
        # flip image
            results['img'] = cv2.flip(results['img'], 1 if results['flip_direction'] == 'horizontal' else 0)
            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = cv2.flip(results[key], 1 if results['flip_direction'] == 'horizontal' else 0).copy()
        return results
    
    def _normalize(self,results,to_rgb=True):
        self.mean = np.array(self.mean, dtype=np.float32)
        self.std = np.array(self.std, dtype=np.float32)
        self.to_rgb = to_rgb
    
        img=results['img']
        img = img.copy().astype(np.float32)

        assert img.dtype != np.uint8
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        results['img']=img
    
    def _totensor(self, results):
        img = results["img"]
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        results["img"] = img.transpose(2, 0, 1) 
        return results
    
    def preprocess(self, img):
        self.origin_scale=img.shape 
        results={}
        # 图像大小调整
        results['img'] = img
        results['flip']= self.flip
        results['flip_direction']=self.flip_direction
        results['keep_ratio']=self.keep_ratio
        results['to_rgb']=self.to_rgb

        """Resize images with ``results['scale']``."""

        self.resize_imgs.append(self._resize_img(results))
        """Call function to flip bounding boxes, masks, semantic segmentation maps"""
        self._flip(results)
        self._normalize(results)
        self._totensor(results)
        # 返回处理后的数据
        return results['img']


    def predict(self, img_list):
        input_data = {self.input_name: img_list}
        outputs = self.net.process(self.graph_name, input_data)
        return [list(outputs.values())[0][0][0]]

    def postprocess(self,bmimg_list, result):
        palette=dp.get_palette(args.palette)
        output_bmimgs=[]
        for i, input_img in enumerate(bmimg_list):
            img=dp.palette_img(input_img,result[i],palette)
            if RESEZE_ORIGIN:
            # 还原原来的图片大小
                if self.origin_scale!=img.shape:
                    h, w=self.origin_scale[0],self.origin_scale[1]
                    img=cv2.resize(img,(w,h))
            output_bmimgs.append(img)
        return output_bmimgs

    def sav_result(self,palette_img):
        dp.save_and_show_palette_img(palette_img)

    def __call__(self, img_list):

        img_num = len(img_list)
        preprocessed_img_list = []
        for ori_img in img_list:
            start_time = time.time()
            preprocessed_img= self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_img)

    
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
            start_time = time.time()
            outputs = [self.predict(input_img)]
            self.inference_time += time.time() - start_time
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
            start_time = time.time()
            outputs = self.predict(input_img)[:img_num]
            self.inference_time += time.time() - start_time
        
        start_time = time.time()
        color_seg_img = self.postprocess(self.resize_imgs,outputs)
        self.postprocess_time += time.time() - start_time
        self.resize_imgs=[]

        return outputs[0],color_seg_img

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
        img_list = []
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
            src_img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
            # print(bmimg.format(), bmimg.dtype())
            if src_img is None:
                    logging.error("{} imdecode is None.".format(filepath))
                    continue
            if len(src_img.shape) != 3:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
            decode_time += time.time() - start_time

            frame_num += 1
            img_list.append(src_img)
            filename_list.append(filename)
            filepath_list.append(filepath)

            if len(img_list) != batch_size and frame_num != (img_num - 1):
                continue
            if (frame_num % batch_size == 0 or flag == False) and len(img_list):
                # 解决多batch，数据集不能被batch整除情况
                if(frame_num % batch_size == 0):
                    record_size=batch_size
                else:
                    record_size=frame_num % batch_size

                while(frame_num % batch_size != 0):
                    frame_num += 1
                    img_list.append(src_img)
                    filename_list.append(filename)
                    filepath_list.append(filepath)
                output, seg= segformer(img_list)
                for i in range(record_size):
                    filename=filename_list[i]
                    result_json={}
                    filepath=filepath_list[i]
                    filename_no_ext = os.path.splitext(filename)[0]
                    res_file =f'{filename_no_ext}.png' # 构造保存的np文件名
                    res_file = os.path.join(res_dir, res_file)
                    result_json["res"] = res_file  # 将文件路径存储在字典中
                        # 将 NumPy 数组转换为 cv::Mat 格式
                    v_mat = cv2.UMat(output[i])
                    cv2.imwrite(res_file,v_mat)

                    relative_path = os.path.relpath(filepath, img_dir)
                    result_json["filename"]=relative_path                   
                    seg_map = relative_path.replace(data["img_suffix"], data["seg_map_suffix"])
                    result_json["ann"]={"seg_map":seg_map}

                    # save image   
                    cv2.imwrite(os.path.join(output_img_dir, filename), seg[i])
                    img_info.append(result_json)
                img_list.clear()
                filename_list.clear()
                filepath_list.clear()
        result_jsons["img_info"]=img_info
       # save result
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_opencv" + "_python_result.json"
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
            start_time = time.time()
            ret, frame = cap.read()
            decode_time += time.time() - start_time
            if not ret or frame is None:
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
                    frame_list.append(frame)

                output, seg= segformer(frame_list)
                for i, in range(record_size):
                    save_name = os.path.join(video_output_dir, str(frame_num - len(frame_list) + i + 1))
                    cv2.imwrite(save_name+".jpg",seg[i])
                    print("frame:"+str(frame_num)+" success !!!")
            frame_list.clear()
        cap.release()
        logging.info("result saved in {}".format(output_dir))
    
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
