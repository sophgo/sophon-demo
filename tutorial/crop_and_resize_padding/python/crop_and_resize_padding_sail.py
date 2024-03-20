#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import argparse
import sophon.sail as sail

class SailCropandResizePadding(object):
    def __init__(self, image_path):
        self.path = os.path.abspath(os.path.expanduser(image_path))
        if not os.path.exists(self.path):
            raise FileNotFoundError('{} is not found'.format(self.path))
        
        # params
        self.use_vpp = True
        self.dev_id = 0
        self.output_width = 640
        self.output_height = 640
        self.interpolation = sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR

        self.handle = sail.Handle(self.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.input_dtype = sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE
        self.output_dtype = sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE
        

    def read(self):
        decoder = sail.Decoder(self.path, True, self.dev_id)
        bmimg = sail.BMImage()
        ret = decoder.read(self.handle, bmimg)    
        if ret != 0:
            raise ValueError("{} decode failure.".format(self.path))
        return bmimg

    def letterbox(self, bmimage):
        img_w = bmimage.width()
        img_h = bmimage.height()

        r_w = self.output_width / img_w
        r_h = self.output_height / img_h
        if r_h > r_w:
            tw = self.output_width
            th = int(r_w * img_h)
            tx1 = tx2 = 0
            ty1 = int((self.output_height - th) / 2)
            ty2 = self.output_height - th - ty1
        else:
            tw = int(r_h * img_w)
            th = self.output_height
            tx1 = int((self.output_width - tw) / 2)
            tx2 = self.output_width - tw - tx1
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
        resized_img_rgb = preprocess_fn(bmimage, 0, 0, img_w, img_h, self.output_width, self.output_height, attr)

        return resized_img_rgb, ratio, txy


    def run(self):
        input_image = self.read()
        output_image, ratio, txy = self.letterbox(input_image)
        self.bmcv.imwrite('crop_and_resize_padding.jpg', output_image)

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--image_path', type=str, help='input image path')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    sail_resize = SailCropandResizePadding(opt.image_path)
    sail_resize.run()
    print('all done.')
