#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import time
import json
import argparse
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
from utils import COLORS
import logging
logging.basicConfig(level=logging.INFO)
# sail.set_print_flag(1)

class YOLOx:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))
        # self.handle = self.net.get_handle()
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]
        
        # get input
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype= self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}
        
        
        # get output
        self.output_names = self.net.get_output_names(self.graph_name)
        if len(self.output_names) not in [1, 3]:
            raise ValueError('only suport 1 or 3 outputs, but got {} outputs bmodel'.format(len(self.output_names)))
        
        self.output_tensors = {}
        self.output_scales = {}
        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            output_dtype = self.net.get_output_dtype(self.graph_name, output_name)
            output_scale = self.net.get_output_scale(self.graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.output_tensors[output_name] = output
            self.output_scales[output_name] = output_scale
        
        # check batch size 
        self.batch_size = self.input_shape[0]
        suppoort_batch_size = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
        if self.batch_size not in suppoort_batch_size:
            raise ValueError('batch_size must be {} for bmcv, but got {}'.format(suppoort_batch_size, self.batch_size))
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.input_size = [self.net_w,self.net_h]
        
        # init preprocess
        self.use_resize_padding = True
        self.use_vpp = False
        # self.ab = [x * self.input_scale / 255.  for x in [1, 0, 1, 0, 1, 0]]
        self.ab = [x * self.input_scale / 1.  for x in [1, 0, 1, 0, 1, 0]]
        
        # init postprocess
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.agnostic = False
        self.multi_label = True
        self.max_det = 1000
        self.postprocess = PostProcess(
            input_h=self.net_h,
            input_w=self.net_w, 
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
            input_h=self.net_h,
            input_w=self.net_w
        )
        
        # init time
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        
    def preprocess_bmcv(self, input_bmimg):
        rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                          sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)
        resized_img_rgb, ratio, txy = self.resize_bmcv(rgb_planar_img)
        preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, ((self.ab[0], self.ab[1]), \
                                                                 (self.ab[2], self.ab[3]), \
                                                                 (self.ab[4], self.ab[5])))
        return preprocessed_bmimg, ratio, txy

    def resize_bmcv(self, bmimg):
        """
        resize for single sail.BMImage
        :param bmimg:
        :return: a resize image of sail.BMImage
        """
        img_w = bmimg.width()
        img_h = bmimg.height()
        if self.use_resize_padding:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            if r_h > r_w:
                tw = self.net_w
                th = int(r_w * img_h)
                tx1 = tx2 = 0
                ty1 = 0
                ty2 = int(self.net_h - th)
            else:
                tw = int(r_h * img_w)
                th = self.net_h
                tx1 = 0
                tx2 = int((self.net_w - tw))
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
            resized_img_rgb = preprocess_fn(bmimg, 0, 0, img_w, img_h, self.net_w, self.net_h, attr)
        else:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            ratio = (r_w, r_h)
            txy = (0, 0)
            preprocess_fn = self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
            resized_img_rgb = preprocess_fn(bmimg, self.net_w, self.net_h)
        return resized_img_rgb, ratio, txy
    
    def predict(self, input_tensor, img_num):
        """
        ensure output order: loc_data, conf_preds, mask_data, proto_data
        Args:
            input_tensor:
        Returns:
        """
        input_tensors = {self.input_name: input_tensor} 
        self.net.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)
        outputs_dict = {}
        for name in self.output_names:
            # outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num] * self.output_scales[name]
            outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num]
        # resort
        out_keys = list(outputs_dict.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n in k:
                    ord.append(i)
                    break
        out = [outputs_dict[out_keys[i]] for i in ord]
        return out

    def __call__(self, bmimg_list):
        img_num = len(bmimg_list)
        ori_size_list = []
        ratio_list = []
        txy_list = []
        if self.batch_size == 1:
            ori_h, ori_w =  bmimg_list[0].height(), bmimg_list[0].width()
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()      
            preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(bmimg_list[0])
            self.preprocess_time += time.time() - start_time
            ratio_list.append(ratio)
            txy_list.append(txy)
            
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimg, input_tensor)
                
        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                ori_h, ori_w =  bmimg_list[i].height(), bmimg_list[i].width()
                ori_size_list.append((ori_w, ori_h))
                start_time = time.time()
                preprocessed_bmimg, ratio, txy  = self.preprocess_bmcv(bmimg_list[i])
                self.preprocess_time += time.time() - start_time
                ratio_list.append(ratio)
                txy_list.append(txy)
                bmimgs[i] = preprocessed_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)
            
        start_time = time.time()
        outputs = self.predict(input_tensor, img_num)
        self.inference_time += time.time() - start_time
        
        start_time = time.time()
        results = self.postprocess(outputs, self.input_size, ori_size_list, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results

def draw_bmcv(bmcv, bmimg, boxes, classes_ids=None, conf_scores=None, save_path=""):
    img_bgr_planar = bmcv.convert_format(bmimg)
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        logging.debug("class id={}, score={}, (x1={},y1={},w={},h={})".format(int(classes_ids[idx]), conf_scores[idx], x1, y1, x2-x1, y2-y1))
        if conf_scores[idx] < 0.25:
            continue
        if classes_ids is not None:
            color = np.array(COLORS[int(classes_ids[idx]) + 1]).astype(np.uint8).tolist()
        else:
            color = (0, 0, 255)
        bmcv.rectangle(img_bgr_planar, x1, y1, (x2 - x1), (y2 - y1), color, 2)
    bmcv.imwrite(save_path, img_bgr_planar)
        

def main(args):
    # check params
    if not args.input.startswith("rtsp") and not args.input.startswith("rtmp") and not os.path.exists(args.input):
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
    yolox = YOLOx(args)
    batch_size = yolox.batch_size
    
    handle = sail.Handle(args.dev_id)
    bmcv = sail.Bmcv(handle)
    
    # warm up 
    # bmimg = sail.BMImage(handle, 1080, 1920, sail.Format.FORMAT_YUV420P, sail.DATA_TYPE_EXT_1N_BYTE)
    # for i in range(10):
    #     results = yolox([bmimg])
    yolox.init()

    decode_time = 0.0
    # test images
    if os.path.isdir(args.input):     
        bmimg_list = []
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
                decoder = sail.Decoder(img_file, True, args.dev_id)
                bmimg = sail.BMImage()
                ret = decoder.read(handle, bmimg)    
                # print(bmimg.format(), bmimg.dtype())
                if ret != 0:
                    logging.error("{} decode failure.".format(img_file))
                    continue
                decode_time += time.time() - start_time
                
                bmimg_list.append(bmimg)
                filename_list.append(filename)
                if len(bmimg_list) == batch_size:
                    # predict
                    results = yolox(bmimg_list)
                    
                    for i, filename in enumerate(filename_list):
                        det = results[i]
                        # save image
                        save_path = os.path.join(output_img_dir, filename)
                        draw_bmcv(bmcv, bmimg_list[i], det[:,:4], classes_ids=det[:, -1], conf_scores=det[:, -2], save_path=save_path)
                        
                        # save result
                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['bboxes'] = []
                        for idx in range(det.shape[0]):
                            bbox_dict = dict()
                            x1, y1, x2, y2, score, category_id = det[idx]
                            bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
                            bbox_dict['category_id'] = int(category_id)
                            bbox_dict['score'] = float(round(score,5))
                            res_dict['bboxes'].append(bbox_dict)
                        results_list.append(res_dict)
                        
                    bmimg_list.clear()
                    filename_list.clear()
        if len(bmimg_list):
            results = yolox(bmimg_list)
            for i, filename in enumerate(filename_list):
                det = results[i]
                save_path = os.path.join(output_img_dir, filename)
                draw_bmcv(bmcv, bmimg_list[i], det[:,:4], classes_ids=det[:, -1], conf_scores=det[:, -2], save_path=save_path)
                res_dict = dict()
                res_dict['image_name'] = filename
                res_dict['bboxes'] = []
                for idx in range(det.shape[0]):
                    bbox_dict = dict()
                    x1, y1, x2, y2, score, category_id = det[idx]
                    bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
                    bbox_dict['category_id'] = int(category_id)
                    bbox_dict['score'] = float(round(score,5))
                    res_dict['bboxes'].append(bbox_dict)
                results_list.append(res_dict)
            bmimg_list.clear()
            filename_list.clear()
            
        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_bmcv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            # json.dump(results_list, jf)
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))
        
    # test video
    else:
        decoder = sail.Decoder(args.input, True, args.dev_id)
        if not decoder.is_opened():
            raise Exception("can not open the video")
        video_name = os.path.splitext(os.path.split(args.input)[1])[0]
        cn = 0
        frame_list = []
        while True:
            if args.max_frames and cn > args.max_frames:
                break
            frame = sail.BMImage()
            start_time = time.time()
            ret = decoder.read(handle, frame)
            if ret:
                break
            decode_time += time.time() - start_time
            frame_list.append(frame)
            if len(frame_list) == batch_size:
                results = yolox(frame_list)
                for i, frame in enumerate(frame_list):
                    det = results[i]
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                    save_path = os.path.join(output_img_dir, video_name + '_' + str(cn) + '.jpg')
                    draw_bmcv(bmcv, frame_list[i], det[:,:4], classes_ids=det[:, -1], conf_scores=det[:, -2], save_path=save_path)
                frame_list.clear()
        if len(frame_list):
            results = yolox(frame_list)
            for i, frame in enumerate(frame_list):
                det = results[i]
                cn += 1
                logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                save_path = os.path.join(output_img_dir, video_name + '_' + str(cn) + '.jpg')
                draw_bmcv(bmcv, frame_list[i], det[:,:4], classes_ids=det[:, -1], conf_scores=det[:, -2], save_path=save_path)
        decoder.release()
        logging.info("result saved in {}".format(output_img_dir))

    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = yolox.preprocess_time / cn
    inference_time = yolox.inference_time / cn
    postprocess_time = yolox.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    # average_latency = decode_time + preprocess_time + inference_time + postprocess_time
    # qps = 1 / average_latency
    # logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))              

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='./datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='./models/BM1684X/yolox_s_fp16_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.40, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    parser.add_argument('--max_frames', type=int, default=None, help='max number of frames to process')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')








