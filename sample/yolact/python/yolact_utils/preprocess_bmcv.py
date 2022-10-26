#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
import sophon.sail as sail

class PreProcess:
    def __init__(self, cfg, batch_size, img_dtype, input_scale=None):
        self.cfg = cfg
        self.img_dtype = img_dtype
        mean_bgr = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        std_bgr = np.array([57.38, 57.12, 58.40], dtype=np.float32)
        self.mean = mean_bgr[::-1]  # bmcv use mean_rgb after bgr2rgb
        self.std = std_bgr[::-1]    # bmcv use std_rgb after bgr2rgb
        self.input_scale = float(1.0) if input_scale is None else input_scale

        self.normalize = cfg['normalize']
        self.subtract_means = cfg['subtract_means']
        self.to_float = cfg['to_float']

        self.batch_size = batch_size
        self.width = cfg['width']
        self.height = cfg['height']
        self.use_vpp = False

    def __call__(self, img, handle, bmcv):
        """
        pre-processing
        Args:
            img: sail.BMImage
            handle:
            bmcv:

        Returns: sail.BMImage after pre-processing

        """
        resized_img_rgb = self.resize(img, handle, bmcv)

        preprocessed_img = sail.BMImage(handle, self.height, self.width,
                                        sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)

        if self.normalize:
            a = 1 / self.std
            b = - self.mean / self.std
        elif self.subtract_means:
            a = (1, 1, 1)
            b = - self.std
        elif self.to_float:
            a = 1 / self.std
            b = (0, 0, 0)
        else:
            raise NotImplementedError
        alpha_beta = tuple([(ia * self.input_scale, ib * self.input_scale) for ia, ib in zip(a, b)])
        bmcv.convert_to(resized_img_rgb, preprocessed_img, alpha_beta)
        return preprocessed_img

    def infer_batch(self, img_list, handle, bmcv):
        """
        batch pre-processing
        Args:
            img_list: a list of sail.BMImage
            handle:
            bmcv:

        Returns: a list of sail.BMImage after pre-processing

        """
        preprocessed_img_list = []
        for img in img_list:
            preprocessed_img = self(img, handle, bmcv)
            preprocessed_img_list.append(preprocessed_img)
        return preprocessed_img_list

    def resize(self, img, handle, bmcv):
        """
        resize for single sail.BMImage
        :param img:
        :param handle:
        :param bmcv:
        :return: a resize image of sail.BMImage
        """
        tmp_planar_img = sail.BMImage(handle, img.height(), img.width(),
                                      sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        bmcv.convert_format(img, tmp_planar_img)
        # resize and bgr2rgb
        preprocess_fn = bmcv.vpp_resize if self.use_vpp else bmcv.resize
        resized_img_rgb = preprocess_fn(tmp_planar_img, self.width, self.height)
        return resized_img_rgb

    def resize_batch(self, img_list, handle, bmcv):
        """
        resize in a batch using single resize for loop
        :param img_list: a list of sail.BMImage
        :param handle:
        :param bmcv:
        :return: a list of resized image of sail.BMImage
        """
        resized_img_list = []
        for img in img_list:
            resized_img = self.resize(img, handle, bmcv)
            resized_img_list.append(resized_img)
        return resized_img_list

    def norm_batch(self, resized_images, handle, bmcv):
        """
        (resized_images - mean) / std, use bmcv.convert_to only once for batch
        :param resized_images: resized image of BMImageArray
        :param handle:
        :param bmcv:
        :return: normalized_images of BMImageArray
        """
        bm_array = eval('sail.BMImageArray{}D'.format(self.batch_size))

        preprocessed_imgs = bm_array(handle,
                                     self.height,
                                     self.width,
                                     sail.FORMAT_RGB_PLANAR,
                                     self.img_dtype)
        if self.normalize:
            a = 1 / self.std
            b = - self.mean / self.std
        elif self.subtract_means:
            a = (1, 1, 1)
            b = - self.std
        elif self.to_float:
            a = 1 / self.std
            b = (0, 0, 0)
        else:
            raise NotImplementedError
        alpha_beta = tuple([(ia * self.input_scale, ib * self.input_scale) for ia, ib in zip(a, b)])
        bmcv.convert_to(resized_images, preprocessed_imgs, alpha_beta)
        return preprocessed_imgs





