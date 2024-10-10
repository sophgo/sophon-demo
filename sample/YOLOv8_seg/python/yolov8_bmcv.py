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
from pycocotools.mask import encode

from utils import *
import logging
logging.basicConfig(level=logging.INFO)

class YOLOv8:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))
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
        
        # init preprocess
        self.use_resize_padding = True
        self.use_vpp = False
        self.ab = [x * self.input_scale / 255.  for x in [1, 0, 1, 0, 1, 0]]
        
        # init postprocess
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.agnostic = False
        self.multi_label = False
        self.max_det = 300

        self.postprocess = PostProcess(
            conf_thres=self.conf_thresh,
            iou_thres=self.nms_thresh
        )
        
        # Related to TPU post-processing
        if 'use_tpu_opt' in getattr(args, '__dict__', {}):
            self.use_tpu_opt = args.use_tpu_opt
        else:
            self.use_tpu_opt = False
        
        self.tpu_opt_process = None
        
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
            r = min(r_w, r_h)
            tw = int(round(r * img_w))
            th = int(round(r * img_h))
            tx1, ty1 = self.net_w - tw, self.net_h - th  # wh padding

            tx1 /= 2  # divide padding into 2 sides
            ty1 /= 2

            ratio = (r, r)
            txy = (tx1, ty1)
            attr = sail.PaddingAtrr()
            attr.set_stx(int(round(tx1 - 0.1)))
            attr.set_sty(int(round(ty1 - 0.1)))
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)

            preprocess_fn = self.bmcv.vpp_crop_and_resize_padding if self.use_vpp else self.bmcv.crop_and_resize_padding
            resized_img_rgb = preprocess_fn(bmimg, 0, 0, img_w, img_h, self.net_w, self.net_h, attr, sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)

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
            ori_size_list.append((ori_h, ori_w))
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
                ori_size_list.append((ori_h, ori_w))
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
    
        # TPU post processing
        if self.use_tpu_opt:
            getmask_bmodel_path = args.getmask_bmodel
            
            if self.tpu_opt_process is None:
                detection_shape = list(outputs[0].shape)  
                segmentation_shape = list(outputs[1].shape)  
                self.tpu_opt_process = sail.algo_yolov8_seg_post_tpu_opt(getmask_bmodel_path, args.dev_id, detection_shape, segmentation_shape, self.net_h, self.net_w)
                
            results = []
            for i in range(img_num):
                
                detection_input = dict(detection_input = sail.Tensor(self.handle, outputs[0][i:i+1, :, :], True))
                segmentation_input = dict(segmentation_input = sail.Tensor(self.handle, outputs[1][i:i+1, :, :, :], True))
                
                start_time = time.time()
                results_sail = self.tpu_opt_process.process(detection_input, segmentation_input, ori_size_list[i][0], ori_size_list[i][1], self.conf_thresh, self.nms_thresh, True, False)
                self.postprocess_time += time.time() - start_time
                
                boxes = []
                contours = []
                masks = []
                for item in results_sail:
                    boxes.append(list(item[:6]))
                    contours.append([item[6]])
                    masks.append(np.array(item[7]))
                
                result_tuple = (boxes, contours, masks)
                results.append(result_tuple)
            
        else:
            start_time = time.time()
            results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
            self.postprocess_time += time.time() - start_time

        return results
        


def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    if not os.path.exists(args.getmask_bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.getmask_bmodel))
    
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 

    yolov8 = YOLOv8(args)
    batch_size = yolov8.batch_size
    
    # warm up 
    # bmimg = sail.BMImage(yolov8.handle, 1080, 1920, sail.Format.FORMAT_YUV420P, sail.DATA_TYPE_EXT_1N_BYTE)
    # for i in range(10):
    #     results = yolov8([bmimg])
    yolov8.init()
    bmcv = sail.Bmcv(sail.Handle(0))
    decode_time = 0.0
    # test images
    if os.path.isdir(args.input): 
        bmimg_list = []
        filename_list = []
        results_list = []
        cn = 0
        yolov8_handle = sail.Handle(args.dev_id)
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
                ret = decoder.read(yolov8_handle, bmimg)

                if ret != 0:
                    logging.error("{} decode failure.".format(img_file))
                    continue
                decode_time += time.time() - start_time
                bmimg_list.append(bmimg)
                filename_list.append(filename)

                if (len(bmimg_list) == batch_size or cn == len(filenames)) and len(bmimg_list):
                    # predict
                    results = yolov8(bmimg_list)
                    for i, filename in enumerate(filename_list):
                        boxes, segments, masks =  results[i]
                        def single_encode(x):
                            """Encode predicted masks as RLE and append results to jdict.
                            https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/models/yolo/segment/val.py#L195
                            """
                            rle = encode(np.asarray(x[:, :, None], order='F', dtype='uint8'))[0]
                            rle['counts'] = rle['counts'].decode('utf-8')
                            return rle
                        save_basename = 'res_bmcv_{}'.format(os.path.basename(filename_list[i]))
                        save_name = os.path.join(output_img_dir, save_basename.replace('.jpg', ''))
                        yolov8.postprocess.draw_and_visualize(save_name,bmimg_list[i].asmat(), boxes, segments, vis=False, save=True)
                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['bboxes'] = []
                        res_dict['segs'] = []
                        for idx in range(len(boxes)):
                            rles = single_encode(masks[idx])

                            bbox_dict = dict()
                            x1, y1, x2, y2, score, category_id = boxes[idx]
                            bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
                            bbox_dict['category_id'] = int(category_id)
                            bbox_dict['score'] = float(round(score,5))
                            res_dict['bboxes'].append(bbox_dict)
                            res_dict['segs'].append(rles)

                        results_list.append(res_dict)
                        
                        
                    bmimg_list.clear()
                    filename_list.clear()

        
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_bmcv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))
    
       
    
    # test videos
    else:
        decoder = sail.Decoder(args.input, True, args.dev_id)
        if not decoder.is_opened():
            raise Exception("can not open the video")
        video_name = os.path.splitext(os.path.split(args.input)[1])[0]
        cn = 0
        frame_list = []
        yolov8_handle = sail.Handle(args.dev_id)
        end_flag = False
        while not end_flag:
            frame = sail.BMImage()
            start_time = time.time()
            ret = decoder.read(yolov8_handle, frame)
            decode_time += time.time() - start_time
            if ret: #differ from cv.
                end_flag = True
            else:
                frame_list.append(frame)
            if (len(frame_list) == batch_size or end_flag) and len(frame_list):
                results = yolov8(frame_list)
                for i, frame in enumerate(frame_list):
                    boxes, segments, _ =  results[i] 
                    cn += 1
                    logging.info("{}".format(cn))
                    
                    save_basename = 'res_bmcv_{}'.format(os.path.basename(video_name))+str(cn)
                    save_name = os.path.join(output_img_dir, save_basename.replace('.jpg', ''))
                    yolov8.postprocess.draw_and_visualize(save_name,frame_list[i].asmat(), boxes, segments, vis=False, save=True)
                frame_list.clear()
        decoder.release()
        logging.info("result saved in {}".format(output_img_dir))


    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = yolov8.preprocess_time / cn
    inference_time = yolov8.inference_time / cn
    postprocess_time = yolov8.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/yolov8s_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='nms threshold')
    parser.add_argument('--use_tpu_opt', action="store_true", default=False, help='use TPU to accelerate postprocessing')
    parser.add_argument('--getmask_bmodel', type=str, default='../models/yolov8s_getmask_32_fp32.bmodel', help='path of getmask bmodel')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')



