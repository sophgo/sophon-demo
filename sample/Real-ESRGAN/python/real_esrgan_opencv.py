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
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
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
        self.batch_size = self.input_shape[0]
        self.net_w = self.input_shape[3]
        self.net_h = self.input_shape[2]
        
        # get output
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.output_dtype= self.net.get_output_dtype(self.graph_name, self.output_name)
        self.output_img_dtype = self.bmcv.get_bm_image_data_format(self.output_dtype)
        self.output_scale = self.net.get_output_scale(self.graph_name, self.output_name)
        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        self.upsample_scale = self.output_shape[3] / self.input_shape[3]

        # init time info
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, ori_img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        letterbox_img, ratio, (tx1, ty1) = self.letterbox(
            ori_img,
            new_shape=(self.net_h, self.net_w),
            color=(114, 114, 114),
            auto=False,
            scaleFill=False,
            scaleup=True,
            stride=32
        )

        img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.astype(np.float32)
        # input_data = np.expand_dims(input_data, 0)
        img = np.ascontiguousarray(img / 255.0)
        return img, ratio, (tx1, ty1) 
    
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(shape[1] * r), int(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def __call__(self, cvimg_list):
        img_num = len(cvimg_list)
        ori_size_list = []
        preprocessed_img_list=[]
        ori_w_list = []
        ori_h_list = []
        ratio_list = []
        txy_list = []
        if self.batch_size == 1:
            ori_h, ori_w = cvimg_list[0].shape[0], cvimg_list[0].shape[1]
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()      
            preprocessed_img, ratio, txy = self.preprocess(cvimg_list[0])
            preprocessed_img_list.append(preprocessed_img)
            self.preprocess_time += time.time() - start_time
            ratio_list.append(ratio)
            txy_list.append(txy)
        else:
            # print("only support onnx batch_size=1")
            start_time = time.time()
            for i in range(img_num):
                ori_h, ori_w = cvimg_list[i].shape[0], cvimg_list[i].shape[1]
                ori_size_list.append((ori_w, ori_h))
                preprocessed_img, ratio, txy = self.preprocess(cvimg_list[i])
                preprocessed_img_list.append(preprocessed_img)
                ratio_list.append(ratio)
                txy_list.append(txy)
            self.preprocess_time += time.time() - start_time

        start_time = time.time()
        # input_data = np.expand_dims(preprocessed_bmimg, axis=0)
        input_data = np.stack(preprocessed_img_list, axis=0)
    
        input_tensors = {self.input_name: input_data}
        outputs_arr = self.net.process(self.graph_name, input_tensors)[self.output_name]
        self.inference_time += time.time() - start_time

        start_time = time.time()
        #cpu postprocess
        outputs_arr = outputs_arr.astype(np.float32)
        outputs_arr *= 255
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
    output_img_dir = os.path.join(output_dir, 'images_opencv')
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
        cvimg_list = []
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
                cvimg = cv2.imread(img_file)
                if cvimg is None:
                    logging.error("{} decode failure.".format(img_file))
                    continue
                decode_time += time.time() - start_time
                logging.info("{}, img_file: {}, shape: [{},{}]".format(cn, img_file, cvimg.shape[0], cvimg.shape[1]))
                cvimg_list.append(cvimg)
                filename_list.append(filename)
                if (len(cvimg_list) == batch_size or cn == len(filenames)) and len(cvimg_list):
                    # predict
                    outputs_arr = real_esrgan(cvimg_list)
                    for i in range(len(outputs_arr)):
                        save_path = os.path.join(output_img_dir, filename_list[i])
                        arr = cv2.cvtColor(outputs_arr[i], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path, arr)
                    cvimg_list.clear()
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
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/real_esrgan_int8_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')








