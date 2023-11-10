#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
import cv2
import logging
import sophon.sail as sail
import time
class Extractor(object):
    def __init__(self, model_path, dev_id):
        self.net = sail.Engine(model_path, dev_id, sail.IOMode.SYSO)
        logging.info("load {} success!".format(model_path))
        # self.handle = self.net.get_handle()
        self.handle = sail.Handle(dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_name)
        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.size = (self.net_w, self.net_h)
        self.output_tensor = sail.Tensor(self.handle, self.output_shape, self.output_dtype,  True, True)
        self.output_tensors = {self.output_name: self.output_tensor}
        

        self.mean = [0.406, 0.456, 0.485] #BGR
        self.std = [0.225, 0.224, 0.229]

        self.a = [1/(255.*x) for x in self.std]
        self.b = [-x/y for x,y in zip(self.mean, self.std)]
        self.ab = []
        for i in range(3):
            self.ab.append(self.a[i]*self.input_scale)
            self.ab.append(self.b[i]*self.input_scale)

        self.use_vpp = False
        self.preprocess_time = 0.0
        self.inference_time = 0.0


    def preprocess(self, im_batch):            
        if len(im_batch) > self.batch_size:
            raise KeyError("Batchsize incorrect! Must less than bmodel batchsize")
        
        if self.batch_size == 1:
            input_bmimg = im_batch[0]
            rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                          sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
            self.bmcv.convert_format(input_bmimg, rgb_planar_img)
            resized_img_rgb = self.bmcv.resize(rgb_planar_img, self.net_w, self.net_h, sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)

            preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
            self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, ((self.ab[0], self.ab[1]), \
                                                                    (self.ab[2], self.ab[3]), \
                                                                    (self.ab[4], self.ab[5])))
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimg, input_tensor)
        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(len(im_batch)):
                input_bmimg = im_batch[i]
                rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                            sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
                self.bmcv.convert_format(input_bmimg, rgb_planar_img)
                resized_img_rgb = self.bmcv.resize(rgb_planar_img, self.net_w, self.net_h, sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)

                preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
                self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, ((self.ab[0], self.ab[1]), \
                                                                        (self.ab[2], self.ab[3]), \
                                                                        (self.ab[4], self.ab[5])))
                bmimgs[i] = preprocessed_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)

        
        return input_tensor


    def __call__(self, im_crops):
        i = 0
        features = []
        while i < len(im_crops):
            im_batch = im_crops[i:min(i + self.batch_size, len(im_crops))]

            start_preprocess = time.time()           
            input_batch = self.preprocess(im_batch)
            self.preprocess_time += time.time() - start_preprocess
            input_data = {self.input_name: input_batch}
            start_inference = time.time()
            self.net.process(self.graph_name, input_data, self.output_tensors)
            self.inference_time += time.time() - start_inference
            features_batch = self.output_tensors[self.output_name].asnumpy()
            features.append(features_batch)
            i += self.batch_size

        return np.concatenate(features,axis=0)