import os
import argparse
import sophon.sail as sail

class SailCrop(object):
    def __init__(self, image_path):
        self.path = os.path.abspath(os.path.expanduser(image_path))
        if not os.path.exists(self.path):
            raise FileNotFoundError('{} is not found'.format(self.path))
        
        # params
        self.use_vpp = False
        self.dev_id = 0
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


    def run(self):
        input_image = self.read()
        resize_fn = self.bmcv.vpp_crop if self.use_vpp else self.bmcv.crop
        crop_x0 = input_image.width() // 4
        crop_y0 = input_image.height() // 4
        crop_w = input_image.width() // 2
        crop_h = input_image.height() // 2
        output_image = resize_fn(input_image, crop_x0, crop_y0, crop_w, crop_h)
        self.bmcv.imwrite('crop.jpg', output_image)

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--image_path', type=str, help='input image path')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    sail_crop = SailCrop(opt.image_path)
    sail_crop.run()
    print('all done.')
