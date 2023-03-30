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
        self.net = sail.Engine(model_path, dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(model_path))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.size = (self.net_w, self.net_h)
        self.mean = [0.406, 0.456, 0.485] #BGR
        self.std = [0.225, 0.224, 0.229]
        self.preprocess_time = 0.0
        self.inference_time = 0.0
    def preprocess(self, im_batch):            
        if len(im_batch) > self.batch_size:
            raise KeyError("Batchsize incorrect! Must less than bmodel batchsize")
        preprocessed_numpy_array = [] #shape: [self.batch_size, 3, 128, 64]
        for im in im_batch:
            resized = cv2.resize(im.astype(np.float32) / 255., self.size)
            meaned = resized - [[self.mean]]
            stded = meaned / [[self.std]] #shape: [128,64,3]
            transposed = np.transpose(stded, (2, 0, 1))[::-1] #shape: [3, 128, 64], bgr2rgb
            preprocessed_numpy_array.append(transposed)
        if len(preprocessed_numpy_array) == self.batch_size:
            input_numpy_array = preprocessed_numpy_array
        else:
            input_numpy_array = np.zeros(self.input_shape, dtype='float32')
            input_numpy_array[:len(preprocessed_numpy_array)] = preprocessed_numpy_array
        return input_numpy_array

    def __call__(self, im_crops):
        i = 0
        features = []
        while i < len(im_crops):
            im_batch = im_crops[i:min(i + self.batch_size, len(im_crops))]
            start_preprocess = time.time()
            
            # temp = cv2.imread("../cpp/deepsort_bmcv/temp/0.jpg")
            # temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            # temp = np.transpose(temp, (2, 0, 1))
            # for i in temp:
            #     for j in i:
            #         print(j)
            #         exit(1)
            # temp_batch=[]
            # temp_batch.append(temp)
            # input_batch = self.preprocess(temp_batch)
            
            input_batch = self.preprocess(im_batch)
            self.preprocess_time += time.time() - start_preprocess
            input_data = {self.input_name: input_batch}
            start_inference = time.time()
            features_batch = self.net.process(self.graph_name, input_data)[self.output_name]
            self.inference_time += time.time() - start_inference
            features.extend(features_batch)
            i += self.batch_size

        return features