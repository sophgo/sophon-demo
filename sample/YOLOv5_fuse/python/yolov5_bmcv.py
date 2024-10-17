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
from utils import COLORS
import logging
logging.basicConfig(level=logging.INFO)
import random
# sail.set_print_flag(1)

class YOLOv5:
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
        if len(self.output_names) not in [1]:
            raise ValueError('only suport 1 outputs, but got {} outputs bmodel'.format(len(self.output_names)))
        
        self.output_tensors = {}
        self.output_scales = {}
        self.output_shapes = []
        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            output_dtype = self.net.get_output_dtype(self.graph_name, output_name)
            output_scale = self.net.get_output_scale(self.graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.output_tensors[output_name] = output
            self.output_scales[output_name] = output_scale
            self.output_shapes.append(output_shape)
        
        # check batch size 
        self.batch_size = self.input_shape[0]
        suppoort_batch_size = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
        if self.batch_size not in suppoort_batch_size:
            raise ValueError('batch_size must be {} for bmcv, but got {}'.format(suppoort_batch_size, self.batch_size))
        
        self.net_h = self.input_shape[1]
        self.net_w = self.input_shape[2]

        # init preprocess
        self.use_resize_padding = True
        self.use_vpp = False
        

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        
    def prepare_data_bmcv(self, input_bmimg):
        temp, ratio, txy = self.resize_bmcv(input_bmimg)
        resized_img = sail.BMImage(self.handle, temp.height(), temp.width(),
                                          sail.Format.FORMAT_BGR_PACKED, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(temp, resized_img)
        # self.bmcv.imwrite("{}_{}.jpg".format(resized_img.format(),random.random()),resized_img)
        # print(input_bmimg.format(), resized_img.format())
        return resized_img, ratio, txy

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
                ty1 = int((self.net_h - th) / 2)
                ty2 = self.net_h - th - ty1
            else:
                tw = int(r_h * img_w)
                th = self.net_h
                tx1 = int((self.net_w - tw) / 2)
                tx2 = self.net_w - tw - tx1
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
        input_tensors = {self.input_name: input_tensor} 
        self.net.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)
        output_shape = self.net.get_output_shape(self.graph_name, self.output_names[0])
        
        outputs_dict = {}
        for name in self.output_names:
            tensor = self.output_tensors[name].asnumpy()[:img_num]
            tensor = tensor[:, :, :output_shape[2], :]
            outputs_dict[name] = tensor
            
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

    def get_results(self, output, img_num: int, ratio_list, txy_list):
        res = np.array(output[0][0][0])
        results = [[] for _ in range(img_num)]

        for row in res:
            image_index = int(row[0])
            if image_index >= img_num:
                continue
            results[image_index].append(row.tolist())  

        for i in range(img_num):
            results[i] = np.array(results[i])
            for item in results[i]:
                x1, y1, x2, y2 = item[3:] 
                item[-4] = int((x1 - x2/2 - txy_list[i][0]) / ratio_list[i][0])  
                item[-3] = int((y1 - y2/2 - txy_list[i][1]) / ratio_list[i][1])  
                item[-2] = int((x1 + x2/2 - txy_list[i][0]) / ratio_list[i][0])  
                item[-1] = int((y1 + y2/2 - txy_list[i][1]) / ratio_list[i][1])

        results = np.array([np.array(x) for x in results], dtype=object) 
        return results


    def __call__(self, bmimg_list):
        img_num = len(bmimg_list)
        ori_size_list = []
        ori_w_list = []
        ori_h_list = []
        ratio_list = []
        txy_list = []
        if self.batch_size == 1:
            ori_h, ori_w =  bmimg_list[0].height(), bmimg_list[0].width()
            ori_size_list.append((ori_w, ori_h))
            ori_w_list.append(ori_w)
            ori_h_list.append(ori_h)
            start_time = time.time()      
            preprocessed_bmimg, ratio, txy = self.prepare_data_bmcv(bmimg_list[0])
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
                ori_w_list.append(ori_w)
                ori_h_list.append(ori_h)
                start_time = time.time()
                preprocessed_bmimg, ratio, txy  = self.prepare_data_bmcv(bmimg_list[i])

                self.preprocess_time += time.time() - start_time
                ratio_list.append(ratio)
                txy_list.append(txy)
                bmimgs[i] = preprocessed_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)
        # print(ratio_list, txy_list)
        # print(preprocessed_bmimg.height(), preprocessed_bmimg.weight(), preprocessed_bmimg.format())
        
        
        # for output_name in self.output_names:
        #     self.output_tensors[output_name].memory_set(0)

        start_time = time.time()
        outputs = self.predict(input_tensor, img_num)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        results = self.get_results(outputs, img_num, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results

def draw_bmcv(bmcv, bmimg, boxes, classes_ids=None, conf_scores=None, save_path="", draw_thresh=None):
    img_bgr_planar = bmcv.convert_format(bmimg)
    thickness = 2
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        logging.debug("class id={}, score={}, (x1={},y1={},w={},h={})".format(int(classes_ids[idx]), conf_scores[idx], x1, y1, x2-x1, y2-y1))
        if conf_scores[idx] < draw_thresh:
            continue
        if classes_ids is not None:
            color = np.array(COLORS[int(classes_ids[idx]) + 1]).astype(np.uint8).tolist()
        else:
            color = (0, 0, 255)
        if (x2 - x1) <= thickness * 2 or (y2 - y1) <= thickness * 2:
            logging.info("width or height too small, this rect will not be drawed: (x1={},y1={},w={},h={})".format(x1, y1, x2-x1, y2-y1))
        else:
            bmcv.rectangle(img_bgr_planar, x1, y1, (x2 - x1), (y2 - y1), color, thickness)
    bmcv.imwrite(save_path, img_bgr_planar)
        
def main(args):
    np.set_printoptions(precision=2, suppress=True)
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
    yolov5 = YOLOv5(args)
    batch_size = yolov5.batch_size
    
    handle = sail.Handle(args.dev_id)
    bmcv = sail.Bmcv(handle)
    # warm up 
    # bmimg = sail.BMImage(handle, 1080, 1920, sail.Format.FORMAT_YUV420P, sail.DATA_TYPE_EXT_1N_BYTE)
    # for i in range(10):
    #     results = yolov5([bmimg])
    yolov5.init()

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
                if (len(bmimg_list) == batch_size or cn == len(filenames)) and len(bmimg_list):
                    # predict
                    results = yolov5(bmimg_list)
                    for i, filename in enumerate(filename_list):
                        
                        det = results[i]
                        # save image
                        save_path = os.path.join(output_img_dir, filename)

                        if len(det.shape) > 1:
                            draw_bmcv(bmcv, bmimg_list[i], det[:,3:7], classes_ids=det[:, 1], conf_scores=det[:, 2], save_path=save_path, draw_thresh=args.draw_thresh)

                        
                        # save result
                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['bboxes'] = []
                        for idx in range(det.shape[0]):
                            bbox_dict = dict()
                            category_id, score, x1, y1, x2, y2 = det[idx][1:7]
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
        end_flag = False
        while not end_flag:
            frame = sail.BMImage()
            start_time = time.time()
            ret = decoder.read(handle, frame)
            decode_time += time.time() - start_time
            if ret: #differ from cv.
                end_flag = True
            else:
                frame_list.append(frame)
            if (len(frame_list) == batch_size or end_flag) and len(frame_list):
                results = yolov5(frame_list)
                
                for i, frame in enumerate(frame_list):
                    det = results[i]
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                    save_path = os.path.join(output_img_dir, video_name + '_' + str(cn) + '.jpg')
                    if det.shape[0] >= 1:
                        draw_bmcv(bmcv, frame_list[i],  det[:,3:7], classes_ids=det[:, 1], conf_scores=det[:, 2], save_path=save_path, draw_thresh=args.draw_thresh)
                frame_list.clear()
        decoder.release()
        logging.info("result saved in {}".format(output_img_dir))

    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = yolov5.preprocess_time / cn
    inference_time = yolov5.inference_time / cn
    postprocess_time = yolov5.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

    # average_latency = preprocess_time + inference_time + postprocess_time
    # qps = 1 / average_latency
    # logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))              

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/yolov5s_v6.1_3output_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--draw_thresh', type=float, default=0.5, help='draw threshold')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')








