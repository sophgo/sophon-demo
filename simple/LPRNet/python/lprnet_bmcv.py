# -*- coding: utf-8 -*- 
import time
import os
import numpy as np
import argparse
import json
import sophon.sail as sail
import logging
logging.basicConfig(level=logging.DEBUG)


CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {i:char for i, char in enumerate(CHARS)}

# input: x.1, [1, 3, 24, 96], float32, scale: 1
class LPRNet(object):
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.tpu_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))
        # define input and ouput
        self.handle = self.net.get_handle()
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.input_dtype= self.net.get_input_dtype(self.graph_name, self.input_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        self.input_shapes = {self.input_name: self.input_shape}
        self.output_tensor = sail.Tensor(self.handle, self.output_shape, self.output_dtype, True,  True)
        self.output_tensors = {self.output_name: self.output_tensor}
        # init bmcv for preprocess
        self.bmcv = sail.Bmcv(self.handle)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.ab = [x * self.input_scale * 0.0078125  for x in [1, -127.5, 1, -127.5, 1, -127.5]]
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        # batch size check
        suppoort_batch_size = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
        if self.batch_size not in suppoort_batch_size:
            raise ValueError('batch_size must be {} for bmcv, but got {}'.format(suppoort_batch_size, self.batch_size))
        
        self.dt = 0.0
        

    def preprocess_bmcv(self, input_bmimg, output_bmimg):
        input_bmimg_bgr = self.bmcv.yuv2bgr(input_bmimg)
        if input_bmimg_bgr.width() != self.net_w or input_bmimg_bgr.height() != self.net_h:
            input_bmimg_bgr = self.bmcv.resize(input_bmimg_bgr, self.net_w, self.net_h)    
        self.bmcv.convert_to(input_bmimg_bgr, output_bmimg, ((self.ab[0], self.ab[1]), \
                                       (self.ab[2], self.ab[3]), \
                                       (self.ab[4], self.ab[5])))
        return output_bmimg

    def predict(self, input_tensor):
        input_tensors = {self.input_name: input_tensor} 
        t0 = time.time()
        self.net.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)
        self.dt += time.time() - t0
        outputs = self.output_tensor.asnumpy()
        return outputs

    def postprocess(self, outputs):
        res = list()
        outputs = np.argmax(outputs, axis = 1)
        for output in outputs:
            no_repeat_blank_label = list()
            pre_c = output[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(CHARS_DICT[pre_c])
            for c in output:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(CHARS_DICT[c])
                pre_c = c
            res.append(''.join(no_repeat_blank_label)) 

        return res

    def __call__(self, bmimg_list):
        img_num = len(bmimg_list)
        if self.batch_size == 1:
            for bmimg in bmimg_list:            
                output_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, \
                        sail.Format.FORMAT_BGR_PLANAR, self.img_dtype)
                output_bmimg = self.preprocess_bmcv(bmimg, output_bmimg)
                input_tensor = sail.Tensor(self.handle, self.input_shape,  self.input_dtype,  False, False)
                self.bmcv.bm_image_to_tensor(output_bmimg, input_tensor)
                outputs = self.predict(input_tensor)
                
        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                output_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, \
                    sail.Format.FORMAT_BGR_PLANAR, self.img_dtype)
                output_bmimg = self.preprocess_bmcv(bmimg_list[i], output_bmimg)
                bmimgs[i] = output_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape,  self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)
            outputs = self.predict(input_tensor)[:img_num]

        res = self.postprocess(outputs)

        return res

    def get_time(self):
        return self.dt

def main(args):
    lprnet = LPRNet(args)
    batch_size = lprnet.batch_size

    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    if not os.path.isdir(args.input_path):
        raise Exception('{} is not a directory.'.format(args.input_path))

    bmimg_list = []
    filename_list = []
    res_dict = {}
    t1 = time.time()
    for root, dirs, filenames in os.walk(args.input_path):
        filenames.sort(key=lambda x:x[1:-4])
        for filename in filenames:
            if os.path.splitext(filename)[-1] not in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
                continue
            img_file = os.path.join(root, filename)
            decoder = sail.Decoder(img_file, True, args.tpu_id)
            bmimg = sail.BMImage()
            ret = decoder.read(lprnet.handle, bmimg)    
            if ret != 0:
                logging.error("{} decode failure.".format(img_file))
                continue
            bmimg_list.append(bmimg)
            filename_list.append(filename)
            if len(bmimg_list) == batch_size:
                res_list = lprnet(bmimg_list)
                for i, filename in enumerate(filename_list):
                    logging.info("filename: {}, res: {}".format(filename, res_list[i]))
                    res_dict[filename] = res_list[i]
                bmimg_list = []
                filename_list = []

    if len(bmimg_list):
        res_list = lprnet(bmimg_list)
        for i, filename in enumerate(filename_list):
            logging.info("filename: {}, res: {}".format(filename, res_list[i]))
            res_dict[filename] = res_list[i]

    t2 = time.time()

    # save result
    json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input_path)[-1] + "_bmcv" + "_python_result.json"
    with open(os.path.join(output_dir, json_name), 'w') as jf:
        json.dump(res_dict, jf, indent=4, ensure_ascii=False)
    logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))
        
    # calculate speed   
    cn = len(res_dict)    
    logging.info("------------------ Inference Time Info ----------------------")
    inference_time = lprnet.get_time() / cn
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    total_time = t2 - t1
    logging.info("total_time(ms): {:.2f}, img_num: {}".format(total_time * 1000, cn))
    average_latency = total_time / cn
    qps = 1 / average_latency
    logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input_path', type=str, default='../data/images/test', help='path of input, must be image directory')
    parser.add_argument('--bmodel', type=str, default='../data/models/BM1684/lprnet_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--tpu_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)
