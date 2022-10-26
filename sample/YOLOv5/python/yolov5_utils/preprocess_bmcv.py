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
    def __init__(self, width, height, batch_size, img_dtype, input_scale=None):
        self.std = np.array([255., 255., 255.], dtype=np.float32)
        self.batch_size = batch_size
        self.input_scale = float(1.0) if input_scale is None else input_scale
        self.img_dtype = img_dtype

        self.width = width
        self.height = height
        self.use_resize_padding = True
        self.use_vpp = False
        print("self.use_vpp:{}".format(self.use_vpp))


    def __call__(self, img, handle, bmcv):
        """
        pre-processing in single image of BMImage
        Args:
            img: sail.BMImage
            handle:
            bmcv:

        Returns: sail.BMImage after pre-processing

        """
        resized_img_rgb, ratio, txy = self.resize(img, handle, bmcv)
        preprocessed_img = sail.BMImage(handle, self.height, self.width,
                                        sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)

        a = 1 / self.std
        b = (0, 0, 0)

        alpha_beta = tuple([(ia * self.input_scale, ib * self.input_scale) for ia, ib in zip(a, b)])
        bmcv.convert_to(resized_img_rgb, preprocessed_img, alpha_beta)

        return preprocessed_img, ratio, txy

    def infer_batch(self, img_list, handle, bmcv):
        """
        batch pre-processing using single image pre-processing for loop
        :param img_list: a list of sail.BMImage
        :param handle:
        :param bmcv:
        :return: a list of sail.BMImage after pre-processing
        """
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for img in img_list:
            preprocessed_img, ratio, txy = self(img, handle, bmcv)
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append(txy)
        return preprocessed_img_list, ratio_list, txy_list

    def resize(self, img, handle, bmcv):
        """
        resize for single sail.BMImage
        :param img:
        :param handle:
        :param bmcv:
        :return: a resize image of sail.BMImage
        """
        if self.use_resize_padding:
            img_w = img.width()
            img_h = img.height()
            r_w = self.width / img_w
            r_h = self.height / img_h

            if r_h > r_w:
                tw = self.width
                th = int(r_w * img_h)
                tx1 = tx2 = 0
                ty1 = int((self.height - th) / 2)
                ty2 = self.height - th - ty1

            else:
                tw = int(r_h * img_w)
                th = self.height
                tx1 = int((self.width - tw) / 2)
                tx2 = self.width - tw - tx1
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

            tmp_planar_img = sail.BMImage(handle, img.height(), img.width(),
                                          sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
            bmcv.convert_format(img, tmp_planar_img)
            preprocess_fn = bmcv.vpp_crop_and_resize_padding if self.use_vpp else bmcv.crop_and_resize_padding
            resized_img_rgb = preprocess_fn(tmp_planar_img,
                                            0, 0, img.width(), img.height(),
                                            self.width, self.height, attr)
        else:
            r_w = self.width / img.width()
            r_h = self.height / img.height()
            ratio = (r_w, r_h)
            txy = (0, 0)
            tmp_planar_img = sail.BMImage(handle, img.height(), img.width(),
                                          sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
            bmcv.convert_format(img, tmp_planar_img)
            preprocess_fn = bmcv.vpp_resize if self.use_vpp else bmcv.resize
            resized_img_rgb = preprocess_fn(tmp_planar_img, self.width, self.height)
        return resized_img_rgb, ratio, txy

    def resize_batch(self, img_list, handle, bmcv):
        """
        resize in a batch using single resize for loop
        :param img_list: a list of sail.BMImage
        :param handle:
        :param bmcv:
        :return: a list of resized image of sail.BMImage
        """
        resized_img_list = []
        ratio_list = []
        txy_list = []
        for img in img_list:
            resized_img, ratio, txy = self.resize(img, handle, bmcv)
            resized_img_list.append(resized_img)
            ratio_list.append(ratio)
            txy_list.append(txy)
        return resized_img_list, ratio_list, txy_list

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

        a = 1 / self.std
        b = (0, 0, 0)
        alpha_beta = tuple([(ia * self.input_scale, ib * self.input_scale) for ia, ib in zip(a, b)])

        bmcv.convert_to(resized_images, preprocessed_imgs, alpha_beta)
        return preprocessed_imgs





