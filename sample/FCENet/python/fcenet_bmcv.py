# -*- coding: utf-8 -*- 
import os
import sys
import time
import cv2
import numpy as np
import argparse
import glob
import sophon.sail as sail
import json
import tqdm
import logging

logging.basicConfig(level=logging.DEBUG)

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)
from utils.fce_postprocess import FCEPostProcess


def resize_image_type0(img):
    limit_side_len = 736
    h, w, c = img.shape

    if min(h, w) < limit_side_len:
        if h < w:
            ratio = float(limit_side_len) / h
        else:
            ratio = float(limit_side_len) / w
    else:
        ratio = 1.
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    if int(resize_w) <= 0 or int(resize_h) <= 0:
        return None, (None, None)
    img = cv2.resize(img, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return img, [ratio_h, ratio_w]


def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points

def filter_tag_det_res_only_clip(dt_boxes, image_shape):
       img_height, img_width = image_shape[0:2]
       dt_boxes_new = []
       for box in dt_boxes:
           if type(box) is list:
               box = np.array(box)
           box = clip_det_res(box, img_height, img_width)
           dt_boxes_new.append(box)
       dt_boxes = np.array(dt_boxes_new)
       return dt_boxes

def draw_text_det_res(dt_boxes, img):
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
    return img

class FCENet(object):
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.tpu_id, sail.IOMode.SYSO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.input_shapes = [self.net.get_input_shape(self.graph_name, name) for name in self.input_names]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_shapes = [self.net.get_output_shape(self.graph_name, name) for name in self.output_names]
        logging.debug("load {} success!".format(args.bmodel))
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_names, self.input_shapes)))
        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_names, self.output_shapes)))
        self.input_name = self.input_names[0]
        self.input_shape = self.input_shapes[0]

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.output_scales = dict([[name, self.net.get_output_scale(self.graph_name, name)] for name in self.output_names])

        self.a = [1/(255.*x) for x in self.std]
        self.b = [-x/y for x,y in zip(self.mean, self.std)]

        self.ab = []
        for i in range(3):
            self.ab.append(self.a[i]*self.input_scale)
            self.ab.append(self.b[i]*self.input_scale)

        self.dt = 0.0

        self.handle = self.net.get_handle()
        self.bmcv = sail.Bmcv(self.handle)
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.output_dtypes = [self.net.get_output_dtype(self.graph_name, name) for name in self.output_names]
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        logging.info('input_dtype: {}, input_bm_image_data_format: {}, output_dtype: {}'.format(str(self.input_dtype), str(self.img_dtype), str(self.output_dtypes)))
        
        self.output_tensors = {}
        for index in range(len(self.output_names)): 
            output_name = self.output_names[index]
            output_shape = self.output_shapes[index]
            output_dtype = self.output_dtypes[index]
            output_tensor = sail.Tensor(self.handle, output_shape, output_dtype,  True, True)
            self.output_tensors[output_name] = output_tensor

        postprocess_params = {}
        postprocess_params["scales"] = [8, 16, 32]
        postprocess_params["alpha"] = 1.0
        postprocess_params["beta"] = 1.0
        postprocess_params["fourier_degree"] = 5
        postprocess_params["box_type"] = "poly"
        self.postprocess_op = FCEPostProcess(**postprocess_params)

    def preprocess_bmcv(self, input_bmimg, output_bmimg):
        src_h, src_w = input_bmimg.height(), input_bmimg.width()
        if input_bmimg.format()==sail.Format.FORMAT_YUV420P:
            print('debug_yuv420p')
            input_bmimg_bgr = self.bmcv.yuv2bgr(input_bmimg)
        else:
            input_bmimg_bgr = input_bmimg

        resize_bmimg = self.bmcv.resize(input_bmimg_bgr, self.net_w, self.net_h)

        self.bmcv.convert_to(resize_bmimg, output_bmimg, ((self.ab[0], self.ab[1]), \
                                       (self.ab[2], self.ab[3]), \
                                       (self.ab[4], self.ab[5])))
        ratio_h, ratio_w = self.net_h/float(src_h), self.net_w/float(src_w) 
        shape = np.array([src_h, src_w, ratio_h, ratio_w])
        return output_bmimg, shape

    def predict(self, input_tensor):
        input_tensors = {self.input_name: input_tensor}
        t0 = time.time()
        self.net.process(self.graph_name, input_tensors, self.output_tensors)
        self.dt += time.time() - t0
        outputs = {}
        for k, v in self.output_tensors.items():
            outputs[k] = v.asnumpy() * self.output_scales[k]
        return outputs

    def postprocess(self, outputs, shape_list):
        res = []
        for index in range(len(shape_list)):
            outputs_e = {}
            for k, v in outputs.items():
                outputs_e[k] = np.expand_dims(v[index, :, :, :], axis=0)
            res_e = self.postprocess_op(outputs_e, np.array([shape_list[index]]))
            res.extend(res_e)
        res_new = []
        for res_e, image_shape in zip(res, shape_list):
            dt_boxes = res_e['points']
            dt_boxes = filter_tag_det_res_only_clip(dt_boxes, image_shape)
            res_e['points'] = dt_boxes
            res_new.append(res_e)
        return res_new

    def __call__(self, img_list):
        input_tensor = sail.Tensor(self.handle, self.input_shape,  self.input_dtype,  False, False)
        shape_list = []
        img_num = len(img_list)
        if self.batch_size == 1:
            output_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_BGR_PLANAR, self.img_dtype)
            output_bmimg, shape = self.preprocess_bmcv(img_list[0], output_bmimg)
            self.bmcv.bm_image_to_tensor(output_bmimg, input_tensor)
            shape_list.append(shape)
        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                output_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_BGR_PLANAR, self.img_dtype)
                output_bmimg, shape = self.preprocess_bmcv(img_list[i], output_bmimg)
                bmimgs[i] = output_bmimg.data()
                shape_list.append(shape)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)

        outputs = self.predict(input_tensor)

        for k, v in outputs.items():
            outputs[k] = v[:img_num]
        shape_list = np.stack(shape_list)
        res = self.postprocess(outputs, shape_list)

        return res

    def get_time(self):
        return self.dt

def main(args):
    fcenet = FCENet(args)
    batch_size = fcenet.batch_size

    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    if not os.path.isdir(args.input_path):
        raise Exception('{} is not a directory.'.format(args.input_path))
        
    img_list = []
    filename_list = []
    res_dict = {}
    t1 = time.time()
    for filename in tqdm.tqdm(glob.glob(args.input_path+'/*')):
        if os.path.splitext(filename)[-1] not in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            continue
        decoder = sail.Decoder(filename, True, args.tpu_id)
        img = sail.BMImage()
        print(filename)
        ret = decoder.read(fcenet.handle, img)    
        if ret != 0:
            logging.error("{} decode failure.".format(filename))
            continue
        img_list.append(img)
        filename_list.append(filename)
        if len(img_list) == batch_size:
            res_list = fcenet(img_list)
            for i, filename in enumerate(filename_list):
                res_dict[filename] = res_list[i]
            img_list = []
            filename_list = []
    if len(img_list):
        res_list = fcenet(img_list)
        for i, filename in enumerate(filename_list):
            res_dict[filename] = res_list[i]

    t2 = time.time()

    # save result
    result_file = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input_path)[-1] + "_bmcv" + "_python_result.json"
    fout = open(os.path.join(output_dir, result_file), 'w')
    vis = False
    vis_img_dir = output_dir + '/vis_img'
    if vis:
        if os.path.exists(vis_img_dir):
            os.system('rm -rf {}'.format(vis_img_dir))
        os.mkdir(vis_img_dir)
    for filename, res in res_dict.items(): 
        points = res['points']
        scores = res['scores']
        res_str = json.dumps({"points": points.tolist(), "scores": scores}, ensure_ascii=False)
        fout.write(filename.split('/')[-1]+'\t'+res_str+'\n')
        if vis:
            res_img = draw_text_det_res(points, cv2.imread(filename))
            vis_img_path = vis_img_dir + '/' + filename.split('/')[-1]
            cv2.imwrite(vis_img_path, res_img)
    fout.close()

    logging.info("result saved in {}".format(os.path.join(output_dir, result_file)))
	    
    # calculate speed  
    cn = len(res_dict)    
    logging.info("------------------ Inference Time Info ----------------------")
    inference_time = fcenet.get_time() / cn
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    total_time = t2 - t1
    logging.info("total_time(ms): {:.2f}, img_num: {}".format(total_time * 1000, cn))
    average_latency = total_time / cn
    qps = 1 / average_latency
    logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))
        
def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input_path', type=str, default='../datasets/ctw1500/imgs/test_opencv_read_write', help='path of input, must be image directory')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684/fcenet_fp32_b1.bmodel', help='path of bmodel')
    parser.add_argument('--tpu_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)