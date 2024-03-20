import os
import argparse
import sophon.sail as sail

class SailResize(object):
    def __init__(self, image_path):
        self.path = os.path.abspath(os.path.expanduser(image_path))
        if not os.path.exists(self.path):
            raise FileNotFoundError('{} is not found'.format(self.path))
        
        # params
        self.use_vpp = True
        self.dev_id = 0
        self.output_width = 233
        self.output_height = 233
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
        resize_fn = self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
        output_image = resize_fn(input_image, self.output_width, self.output_height)
        self.bmcv.imwrite('resize.jpg', output_image)

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--image_path', type=str, help='input image path')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    sail_resize = SailResize(opt.image_path)
    sail_resize.run()
    print('all done.')
