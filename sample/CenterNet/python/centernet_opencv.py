#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import json
import time
import cv2
import argparse
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
from utils import COLORS, COCO_CLASSES
import logging
logging.basicConfig(level=logging.INFO)
# sail.set_print_flag(1)

class CenterNet:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.mean = np.array([0.408, 0.447, 0.470], dtype=np.float32)
        self.std = np.array([0.289, 0.274, 0.278], dtype=np.float32)
        
        self.conf_thresh = args.conf_thresh
        self.agnostic = False
        self.multi_label = True
        self.max_det = 1000
        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            
        )
        
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
    
    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
            
    # def preprocess(self, ori_img):
    #     """
    #     pre-processing
    #     Args:
    #         img: numpy.ndarray -- (h,w,3)

    #     Returns: (3,h,w) numpy.ndarray after pre-processing

    #     """
    #     letterbox_img, ratio, (tx1, ty1) = self.letterbox(
    #         ori_img,
    #         new_shape=(self.net_h, self.net_w),
    #         color=(0, 0, 0),
    #         auto=False,
    #         scaleFill=False,
    #         scaleup=True,
    #         stride=32
    #     )

    #     # img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    #     # img = cv2.cvtColor(letterbox_img, cv2.COLOR_BGR2RGB)
    #     # img = img.astype(np.float32)
    #     # letterbox_img = cv2.imread('origin.bmp')
    #     img = letterbox_img.astype(np.float32)
    #     # input_data = np.expand_dims(input_data, 0)
    #     img = np.ascontiguousarray((img / 255.0 - self.mean)/self.std)
    #     img = np.transpose(img, (2, 0, 1))
    #     return img, ratio, (tx1, ty1) 
    
    # def letterbox(self, im, new_shape=(512, 512), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=32):
    #     # Resize and pad image while meeting stride-multiple constraints
    #     shape = im.shape[:2]  # current shape [height, width]
    #     if isinstance(new_shape, int):
    #         new_shape = (new_shape, new_shape)

    #     # Scale ratio (new / old)
    #     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    #     if not scaleup:  # only scale down, do not scale up (for better val mAP)
    #         r = min(r, 1.0)

    #     # Compute padding
    #     ratio = r, r  # width, height ratios
    #     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    #     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    #     if auto:  # minimum rectangle
    #         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    #     elif scaleFill:  # stretch
    #         dw, dh = 0.0, 0.0
    #         new_unpad = (new_shape[1], new_shape[0])
    #         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    #     dw /= 2  # divide padding into 2 sides
    #     dh /= 2

    #     if shape[::-1] != new_unpad:  # resize
    #         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    #     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    #     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    #     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    #     return im, ratio, (dw, dh)

#---------------------------add-v-----------------------------#
    def preprocess(self, image, scale=1, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width  = int(width * scale)

        inp_height, inp_width = 512, 512
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = self.get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)

        # inp_image = cv2.warpAffine(
        # resized_image, trans_input, (new_width, new_height),
        # flags=cv2.INTER_LINEAR)
        # cv2.imwrite('opencv_trans.jpg', inp_image)
        

        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1)
        
        return images, 0, (0,0)
    
    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)
    
    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result
    
    def get_affine_transform(self,center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans
    
    

#---------------------------------add-^----------------------------- 
    def predict(self, input_img, img_num):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)
        
        # resort
        out_keys = list(outputs.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n == k:
                    ord.append(i)
                    break
        out = [outputs[out_keys[i]][:img_num] for i in ord]
        return out
    
    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
            
        start_time = time.time()
        outputs = self.predict(input_img, img_num)[0]
        self.inference_time += time.time() - start_time

        # np.save('inference.npy', outputs)
        
        start_time = time.time()
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results

def draw_numpy(image, boxes, masks=None, classes_ids=None, conf_scores=None, conf_thresh=0.35):
    for idx in range(len(boxes)):
        if conf_scores[idx]>conf_thresh:
            x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
            if classes_ids is not None:
                color = COLORS[int(classes_ids[idx]) + 1]
            else:
                color = (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
            if classes_ids is not None and conf_scores is not None:
                classes_ids = classes_ids.astype(np.int8)
                cv2.putText(image, COCO_CLASSES[classes_ids[idx] + 1] + ':' + str(round(conf_scores[idx], 2)),
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
            if masks is not None:
                mask = masks[:, :, idx]
                image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
            logging.debug("class id={}, score={}, (x1={},y1={},x2={},y2={})".format(classes_ids[idx],conf_scores[idx], x1, y1, x2, y2))
    return image
   
def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 
    
    # initialize net
    centernet = CenterNet(args)
    batch_size = centernet.batch_size
    
    # warm up 
    # for i in range(10):
    #     results = centernet([np.zeros((640, 640, 3))])
    # centernet.init()
    
    decode_time = 0.0
    # test images

    img_list = []
    filename_list = []
    results_list = []
    cn = 0
    for root, dirs, filenames in os.walk(args.input):
        for filename in filenames:
            if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                continue
            img_file = os.path.join(root, filename)
            cn += 1
            logging.info("{}, img_file: {}".format(cn, img_file))
            # decode
            start_time = time.time()
            src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
            if src_img is None:
                logging.error("{} imdecode is None.".format(img_file))
                continue
            if len(src_img.shape) != 3:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
            decode_time += time.time() - start_time
            
            img_list.append(src_img)
            filename_list.append(filename)
            if len(img_list) == batch_size:
                # predict
                results = centernet(img_list)
                
                for i, filename in enumerate(filename_list):
                    det = results[i]
                    # save image
                    res_img = draw_numpy(img_list[i], det[:,:4], masks=None, classes_ids=det[:, -1], conf_scores=det[:, -2], conf_thresh=centernet.conf_thresh)
                    cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
                    
                    # save result
                    res_dict = dict()
                    res_dict['image_name'] = filename
                    res_dict['bboxes'] = []
                    for idx in range(det.shape[0]):
                        bbox_dict = dict()
                        x1, y1, x2, y2, score, category_id = det[idx]
                        bbox_dict['bbox'] = [int(x1), int(y1), int(x2 - x1), int(y2 -y1)]
                        bbox_dict['category_id'] = int(category_id)
                        bbox_dict['score'] = float(score)
                        res_dict['bboxes'].append(bbox_dict)
                    results_list.append(res_dict)
                    
                img_list.clear()
                filename_list.clear()
                
    if len(img_list):
        results = centernet(img_list)
        for i, filename in enumerate(filename_list):
            det = results[i]
            res_img = draw_numpy(img_list[i], det[:,:4], masks=None, classes_ids=det[:, -1], conf_scores=det[:, -2], conf_thresh=centernet.conf_thresh)
            cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
            res_dict = dict()
            res_dict['image_name'] = filename
            res_dict['bboxes'] = []
            for idx in range(det.shape[0]):
                bbox_dict = dict()
                x1, y1, x2, y2, score, category_id = det[idx]
                bbox_dict['bbox'] = [int(x1), int(y1), int(x2 - x1), int(y2 -y1)]
                bbox_dict['category_id'] = int(category_id)
                bbox_dict['score'] = float(score)
                res_dict['bboxes'].append(bbox_dict)
            results_list.append(res_dict)
        img_list.clear()
        filename_list.clear()   

    # save results
    if args.input[-1] == '/':
        args.input = args.input[:-1]
    json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_opencv" + "_python_result.json"
    with open(os.path.join(output_dir, json_name), 'w') as jf:
        # json.dump(results_list, jf)
        json.dump(results_list, jf, indent=4, ensure_ascii=False)
    logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))
        
   
        
    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = centernet.preprocess_time / cn
    inference_time = centernet.inference_time / cn
    postprocess_time = centernet.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    # average_latency = decode_time + preprocess_time + inference_time + postprocess_time
    # qps = 1 / average_latency
    # logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684/centernet_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.35, help='confidence threshold')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')