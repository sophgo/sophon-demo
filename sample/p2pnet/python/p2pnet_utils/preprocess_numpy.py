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

class PreProcess:
    def __init__(self, width, height):
        self.net_w = width
        self.net_h = height
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

    def preprocess(self, img):
        h, w, _ = img.shape
        if h != self.net_h or w != self.net_w:
            img = cv2.resize(img, (self.net_w, self.net_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img = (img/255-self.mean)/self.std
        img = np.transpose(img, (2, 0, 1))
        return img

    def __call__(self, img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (1,3,h,w) numpy.ndarray after pre-processing

        """
        img = self.preprocess(img)
        input_data = np.expand_dims(img, 0)
        inp = np.ascontiguousarray(input_data)
        return inp.astype(np.float32)


    def infer_batch(self, img_list):
        """
        batch pre-processing
        Args:
            img_list: a list of (h,w,3) numpy.ndarray or numpy.ndarray with (n,h,w,3)

        Returns: (n,3,h,w) numpy.ndarray after pre-processing

        """
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for img in img_list:
            preprocessed_img = self(img)
            preprocessed_img_list.append(preprocessed_img)
        return np.concatenate(preprocessed_img_list)