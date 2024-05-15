#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import time
import argparse
import numpy as np
import sophon.sail as sail
import logging
import cv2 as cv2
logging.basicConfig(level=logging.INFO)

class Real_ESRGAN:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))
        # self.handle = self.net.get_handle()
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]
        
        # get input
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype= self.net.get_input_dtype(self.graph_name, self.input_name)
        self.input_img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False) #let tensor to attach BMImage's device memory
        self.input_tensors = {self.input_name: self.input_tensor} 
        self.batch_size = self.input_shape[0]
        self.net_w = self.input_shape[3]
        self.net_h = self.input_shape[2]
        
        # get output
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.output_dtype= self.net.get_output_dtype(self.graph_name, self.output_name)
        self.output_img_dtype = self.bmcv.get_bm_image_data_format(self.output_dtype)
        self.output_scale = self.net.get_output_scale(self.graph_name, self.output_name)
        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        self.output_tensor = sail.Tensor(self.handle, self.output_shape, self.output_dtype, True, True)
        self.output_tensors = {self.output_name: self.output_tensor} 
        self.upsample_scale = self.output_shape[3] / self.input_shape[3]

        # init preprocess
        self.use_resize_padding = True
        self.use_vpp = True
        self.converto_attr_input = ((self.input_scale / 255., 0), (self.input_scale / 255., 0), (self.input_scale / 255., 0))        
        # init postprocess
        self.converto_attr_output = ((self.output_scale * 255., 0), (self.output_scale * 255., 0), (self.output_scale * 255, 0))  
        
        # init time info
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess_bmcv(self, input_bmimg):
        rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                          sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)
        resized_img_rgb, ratio, txy = self.resize_bmcv(rgb_planar_img)
        preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.input_img_dtype)
        self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, self.converto_attr_input)
        return preprocessed_bmimg, ratio, txy

    def resize_bmcv(self, bmimg):
        """
        resize for single sail.BMImage
        :param bmimg:
        :return: a resize image of sail.BMImage
        """
        img_w = bmimg.width()
        img_h = bmimg.height()
        if self.use_resize_padding:
            r_w = self.net_w / img_w  
            r_h = self.net_h / img_h
            if r_h > r_w:
                tw = self.net_w
                th = int(r_w * img_h)
                tx1 = tx2 = 0
                ty1 = (self.net_h - th) / 2
                ty2 = self.net_h - th - ty1
            else:
                tw = int(r_h * img_w)
                th = self.net_h
                tx1 = (self.net_w - tw) / 2
                tx2 = self.net_w - tw - tx1
                ty1 = ty2 = 0

            ratio = (min(r_w, r_h), min(r_w, r_h))
            txy = (tx1, ty1)
            attr = sail.PaddingAtrr()
            attr.set_stx(int(tx1))
            attr.set_sty(int(ty1))
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)
            
            preprocess_fn = self.bmcv.vpp_crop_and_resize_padding if self.use_vpp else self.bmcv.crop_and_resize_padding
            resized_img_rgb = preprocess_fn(bmimg, 0, 0, img_w, img_h, self.net_w, self.net_h, attr, sail.BMCV_INTER_LINEAR)
        else:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            ratio = (r_w, r_h)
            txy = (0, 0)
            preprocess_fn = self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
            resized_img_rgb = preprocess_fn(bmimg, self.net_w, self.net_h)
        return resized_img_rgb, ratio, txy

    def __call__(self, bmimg_list):
        img_num = len(bmimg_list)
        ori_size_list = []
        ori_w_list = []
        ori_h_list = []
        ratio_list = []
        txy_list = []
        if self.batch_size == 1:
            ori_h, ori_w =  bmimg_list[0].height(), bmimg_list[0].width()
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()      
            preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(bmimg_list[0])
            self.preprocess_time += time.time() - start_time
            ratio_list.append(ratio)
            txy_list.append(txy)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimg, self.input_tensor)
        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                ori_h, ori_w =  bmimg_list[i].height(), bmimg_list[i].width()
                ori_size_list.append((ori_w, ori_h))
                start_time = time.time()
                preprocessed_bmimg, ratio, txy  = self.preprocess_bmcv(bmimg_list[i])
                self.preprocess_time += time.time() - start_time
                ratio_list.append(ratio)
                txy_list.append(txy)
                bmimgs[i] = preprocessed_bmimg.data()
            self.bmcv.bm_image_to_tensor(bmimgs, self.input_tensor)
        
        start_time = time.time()
        self.net.process(self.graph_name, self.input_tensors, self.output_tensors)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        #cpu postprocess
        outputs_arr = self.output_tensor.asnumpy().astype(np.float32)
        outputs_arr *= self.output_scale * 255
        outputs_arr = outputs_arr.transpose(0, 2, 3, 1)
        clipped_arr = []
        for i in range(len(outputs_arr)):
            if txy_list[i][0] != 0: #tx
                tx = int(txy_list[i][0] * self.upsample_scale)
                clipped_arr.append(outputs_arr[i][:, tx : self.output_shape[3] - tx, :])
            elif txy_list[i][1] != 0: #ty
                ty = int(txy_list[i][1] * self.upsample_scale)
                clipped_arr.append(outputs_arr[i][ty : self.output_shape[2] - ty, :, :])
            else:
                clipped_arr.append(outputs_arr[i])  
        self.postprocess_time += time.time() - start_time
        return clipped_arr

        
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
    output_img_dir = os.path.join(output_dir, 'images_bmcv')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)

    # initialize net
    real_esrgan = Real_ESRGAN(args)
    batch_size = real_esrgan.batch_size
    
    handle = sail.Handle(args.dev_id)
    bmcv = sail.Bmcv(handle)

    decode_time = 0.0
    # test images
    if os.path.isdir(args.input):     
        bmimg_list = []
        filename_list = []
        results_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                img_file = os.path.join(root, filename)
                cn += 1
                # decode
                start_time = time.time()
                decoder = sail.Decoder(img_file, True, args.dev_id)
                bmimg = sail.BMImage()
                ret = decoder.read(handle, bmimg)    
                # print(bmimg.format(), bmimg.dtype())
                if ret != 0:
                    logging.error("{} decode failure.".format(img_file))
                    continue
                decode_time += time.time() - start_time
                logging.info("{}, img_file: {}, shape: [{},{}]".format(cn, img_file, bmimg.height(), bmimg.width()))
                bmimg_list.append(bmimg)
                filename_list.append(filename)
                if (len(bmimg_list) == batch_size or cn == len(filenames)) and len(bmimg_list):
                    # predict
                    outputs_arr = real_esrgan(bmimg_list)
                    for i in range(len(outputs_arr)):
                        save_path = os.path.join(output_img_dir, filename_list[i])
                        arr = cv2.cvtColor(outputs_arr[i], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path, arr)
                    bmimg_list.clear()
                    filename_list.clear()
    else:
        print("unsupport input format.")
    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = real_esrgan.preprocess_time / cn
    inference_time = real_esrgan.inference_time / cn
    postprocess_time = real_esrgan.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/coco128', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/real_esrgan_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')








