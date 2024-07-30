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
import argparse
import numpy as np
import sophon.sail as sail
from utils import *
import logging
logging.basicConfig(level=logging.INFO)

class YOLOv8:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.info("load {} success!".format(args.bmodel))
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        
        # get input
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype= self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        # get output
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_tensors = {}
        self.output_scales = {}
        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            if(output_shape[1]>output_shape[2]):
                raise ValueError('Python programs do not support the OPT model')
            output_dtype = self.net.get_output_dtype(self.graph_name, output_name)
            output_scale = self.net.get_output_scale(self.graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.output_tensors[output_name] = output
            self.output_scales[output_name] = output_scale

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
                if n == k:
                    ord.append(i)
                    break
        out = [outputs_dict[out_keys[i]] for i in ord]
        return out
    
    def postprocess(self, preds, ori_size_list, ratio_list, txy_list):
        """
        post-processing
        Args:
            preds: numpy.ndarray -- (n,8400,56) [cx,cy,w,h,conf,17*3]

        Returns: 
            results: list of numpy.ndarray -- (n, 56) [x1, y1, x2, y2, conf, 17*3]

        """
        results = []

        preds = preds[0]
        for i, pred in enumerate(preds):
            # Transpose and squeeze the output to match the expected shape
            pred = np.transpose(pred, (1, 0))   # [8400,56]

            pred = pred[pred[:, 4] > self.conf_thresh]

            if len(pred) == 0:
                print("none detected")
                results.append(np.zeros((0, 56)))
            else:
                # Box (center x, center y, width, height) to (x1, y1, x2, y2)
                pred = self.xywh2xyxy(pred)
                results.append(self.nms_boxes(pred, self.nms_thresh))

        # Rescale boxes and keypoints from img_size to im0 size
        for det, (org_w, org_h), ratio, (tx1, ty1) in zip(results, ori_size_list, ratio_list, txy_list):
            if len(det):
                # Rescale boxes from img_size to im0 size
                coords = det[:, :4]
                coords[:, [0, 2]] -= tx1  # x padding
                coords[:, [1, 3]] -= ty1  # y padding
                coords[:, [0, 2]] /= ratio[0]
                coords[:, [1, 3]] /= ratio[1]

                coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, org_w - 1)  # x1, x2
                coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, org_h - 1)  # y1, y2

                det[:, :4] = coords

                # Rescale keypoints from img_size to im0 size
                num_kpts = (det.shape[1] - 5) // 3
                for kid in range(num_kpts):
                    det[:, 5 + kid * 3] -= tx1
                    det[:, 5 + kid * 3 + 1] -= ty1
                    det[:, 5 + kid * 3] /= ratio[0]
                    det[:, 5 + kid * 3 + 1] /= ratio[1]
                    det[:, 5 + kid * 3] = det[:, 5 + kid * 3].clip(0, org_w - 1)
                    det[:, 5 + kid * 3 + 1] = det[:, 5 + kid * 3 + 1].clip(0, org_h - 1)

        return results

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.copy() if isinstance(x, np.ndarray) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def nms_boxes(self, pred, iou_thres):
        x = pred[:, 0]
        y = pred[:, 1]
        w = pred[:, 2] - pred[:, 0]
        h = pred[:, 3] - pred[:, 1]

        scores = pred[:, 4]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        output = []
        for i in keep:
            output.append(pred[i].tolist())
        return np.array(output)

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
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results


def draw_bmcv(dev_id, image, det_draw, output_img_dir, file_name, cn, masks=None, classes_ids=None, conf_scores=None, isvideo=False):
    bmcv = sail.Bmcv(sail.Handle(dev_id))
    # img_bgr_planar = bmcv.convert_format(image)
    img_bgr_planar = bmcv.vpp_convert_format(image,sail.FORMAT_YUV444P)
    boxes = det_draw[:, :4]
    kpts = det_draw[:, 5:]
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) + 1]
        else:
            color = (0, 0, 255)
        # draw boxs
        bmcv.rectangle(img_bgr_planar, x1, y1, (x2 - x1), (y2 - y1), color, thickness=2)

        # draw keypoints
        for i in range(0, len(kpts[idx]), 3):
            x, y, conf = kpts[idx, i:i + 3]
            # if conf > 0.5:
                # bmcv.drawPoint(img_bgr_planar, (int(x), int(y)), color, 5)
        
        # draw skeleton
        for sk_id, sk in enumerate(skeleton):
            r, g, b = pose_limb_color[sk_id]
            pos1 = (int(kpts[idx, (sk[0]-1)*3]), int(kpts[idx, (sk[0]-1)*3+1]))
            pos2 = (int(kpts[idx, (sk[1]-1)*3]), int(kpts[idx, (sk[1]-1)*3+1]))
            conf1 = kpts[idx, (sk[0]-1)*3+2]
            conf2 = kpts[idx, (sk[1]-1)*3+2]
            if conf1 >0.5 and conf2 >0.5:  # 对于肢体，相连的两个关键点置信度 必须同时大于 0.5
                bmcv.polylines(img_bgr_planar, [[pos1, pos2]], True, [int(r), int(g), int(b)], thickness=2)

        logging.debug("score={}, (x1={},y1={},x2={},y2={})".format(conf_scores[idx], x1, y1, x2, y2))
    if isvideo:
        bmcv.imwrite(os.path.join(output_img_dir, file_name + '_' + str(cn) + '.jpg'), img_bgr_planar)
    else:
        bmcv.imwrite(os.path.join(output_img_dir, file_name), img_bgr_planar)

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

    yolov8 = YOLOv8(args)
    batch_size = yolov8.batch_size
    
    # warm up 
    # for i in range(10):
    #     results = yolov8([np.zeros((640, 640, 3))])
    yolov8.init()

    decode_time = 0.0
    # test images
    if os.path.isdir(args.input): 
        bmimg_list = []
        filename_list = []
        results_list = []
        cn = 0
        yolov8_handle = sail.Handle(args.dev_id)
        for root, dirs, filenames in os.walk(args.input):
            filenames.sort()
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
                if len(bmimg_list) == batch_size:
                    # predict
                    results = yolov8(bmimg_list)
                    for i, filename in enumerate(filename_list):
                        
                        det = results[i] #1, 56
                        # save image
                        det_draw = det[det[:, 4] > 0.25]
                        draw_bmcv(  args.dev_id,
                                    bmimg_list[i], 
                                    det_draw, 
                                    output_img_dir,
                                    filename,
                                    cn,
                                    masks=None, 
                                    classes_ids=None, 
                                    conf_scores=det_draw[:, 4]
                                    )
                        
                        kpts = det_draw[:, 5:]

                        # save result
                        for n in range(kpts.shape[0]):
                            res_dict = dict()
                            res_dict['image_name'] = filename
                            res_dict['score'] = det[n, 4]
                            res_dict['keypoints'] = []
                            for m in range(0, len(kpts[n]), 3):
                                x, y, score = kpts[n, m:m + 3]
                                res_dict['keypoints'].append(x)
                                res_dict['keypoints'].append(y)
                                res_dict['keypoints'].append(score)
                            results_list.append(res_dict)
                        
                    bmimg_list.clear()
                    filename_list.clear()

        if len(bmimg_list):
            # predict
            results = yolov8(bmimg_list)
            for i, filename in enumerate(filename_list):
                det = results[i]
                # save image
                det_draw = det[det[:, 4] > 0.25]
                draw_bmcv(  args.dev_id,
                            bmimg_list[i], 
                            det_draw, 
                            output_img_dir,
                            filename,
                            cn,
                            masks=None, 
                            classes_ids=None, 
                            conf_scores=det_draw[:, 4]
                            )
                
                # save result
                res_dict = dict()
                res_dict['image_name'] = filename
                res_dict['keypoints'] = []
                for n in range(kpts.shape[0]):
                    for m in range(0, len(kpts[n]), 3):
                        x, y, score = kpts[n, i:i + 3]
                        res_dict['keypoints'].append(x)
                        res_dict['keypoints'].append(y)
                        res_dict['keypoints'].append(score)
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
        yolov8_handle = sail.Handle(args.dev_id)
        while True:
            frame = sail.BMImage()
            start_time = time.time()
            ret = decoder.read(yolov8_handle, frame)
            if ret:
                break
            decode_time += time.time() - start_time
            frame_list.append(frame)
            if len(frame_list) == batch_size:
                results = yolov8(frame_list)
                for i, frame in enumerate(frame_list):
                    det = results[i]
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                    det_draw = det[det[:, 4] > 0.25]
                    draw_bmcv(  args.dev_id,
                            frame_list[i], 
                            det_draw, 
                            output_img_dir,
                            video_name,
                            cn,
                            masks=None, 
                            classes_ids=None, 
                            conf_scores=det_draw[:, 4],
                            isvideo=True
                            )
                frame_list.clear()
        if len(frame_list):
            results = yolov8(frame_list)
            for i, frame in enumerate(frame_list):
                det = results[i]
                cn += 1
                logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                det_draw = det[det[:, 4] > 0.25]
                draw_bmcv(  args.dev_id,
                            frame_list[i], 
                            det_draw, 
                            output_img_dir,
                            video_name,
                            cn,
                            masks=None, 
                            classes_ids=None, 
                            conf_scores=det_draw[:, 4],
                            isvideo=True
                            )
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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
