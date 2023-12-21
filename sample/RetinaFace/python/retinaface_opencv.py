#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""
A RetinaFace demo using Sophon SAIL api to make inferences.
"""

# -*- coding: utf-8 -*
import argparse
from ctypes import resize
from fileinput import filename
import os
import sys
import time

import cv2
import numpy as np

import sophon.sail as sail

from data.config import cfg_mnet, cfg_re50
from utils.box_utils import PriorBox, decode, decode_landm, py_cpu_nms, draw_one
from utils.print_utils import print_info, print_infos
from utils.time_utils import timeit

from loguru import logger

opt = None

# 设置numpy运算精度
# np.set_printoptions(threshold=np.inf)

class Retinaface_sophon(object):
    """
    description: A Retineface class that warps Sophon ops, preprocess and postprocess ops.
    """

    def __init__(self, cfg, bmodel_file_path, tpu_id, score_threshold = 0.5, nms_threshold = 0.3):
        """
        :param cfg: retinaface backbone config file
        :param bmodel_file_path: bmodel file
        :param tpu_id: tpu id
        :param score_threshold: confidence
        :param nms_threshold: nms
        """
        
        # Create a Context on sophon device
        tpu_count = sail.get_available_tpu_num()
        #logger.debug('{} TPUs Detected, using TPU {} \n'.format(tpu_count, tpu_id))
        self.engine = sail.Engine(bmodel_file_path, tpu_id, sail.IOMode.SYSIO)
        self.handle = self.engine.get_handle()
        self.graph_name = self.engine.get_graph_names()[0]
        graph_count = len(self.engine.get_graph_names())
        #logger.warning("{} graphs in {}, using {}".format(graph_count, bmodel_file_path, self.graph_name))

        # create input tensors
        input_names     = self.engine.get_input_names(self.graph_name)
        input_tensors   = {}
        input_shapes    = {}
        input_scales    = {}
        input_dtypes    = {}
        inputs          = []
        input_w         = 0
        input_h         = 0

        for input_name in input_names:
            input_shape = self.engine.get_input_shape(self.graph_name, input_name)
            input_dtype = self.engine.get_input_dtype(self.graph_name, input_name)
            input_scale = self.engine.get_input_scale(self.graph_name, input_name)
         
            input_w = int(input_shape[-1])
            input_h = int(input_shape[-2])

            # logger.debug("[{}] create sail.Tensor for input: {} ".format(input_name, input_shape))
            input = sail.Tensor(self.handle, input_shape, input_dtype, False, False)

            inputs.append(input)
            input_tensors[input_name] = input
            input_shapes[input_name] = input_shape
            input_scales[input_name] = input_scale
            input_dtypes[input_name] = input_dtype

        # create output tensors
        output_names    = self.engine.get_output_names(self.graph_name)
        output_tensors  = {}
        output_shapes   = {}
        output_scales   = {}
        output_dtypes   = {}
        outputs         = []

        for output_name in output_names:
            output_shape = self.engine.get_output_shape(self.graph_name, output_name)
            output_dtype = self.engine.get_output_dtype(self.graph_name, output_name)
            output_scale = self.engine.get_output_scale(self.graph_name, output_name)

            # create sail.Tensor for output
            # logger.debug("[{}] create sail.Tensor for output: {} ".format(output_name, output_shape))
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)

            outputs.append(output)
            output_tensors[output_name] = output
            output_shapes[output_name] = output_shape
            output_scales[output_name] = output_scale
            output_dtypes[output_name] = output_dtype

        # Store
        self.inputs = inputs
        self.input_name = input_names[0]
        self.input_tensors = input_tensors
        self.input_scale = input_scales[input_names[0]]
        self.input_dtype = input_dtypes[input_names[0]]
        self.input_shape = self.engine.get_input_shape(self.graph_name,self.input_name)
        self.batch_size = self.engine.get_input_shape(self.graph_name,self.input_name)[0]
        
        self.outputs = outputs
        self.output_names = output_names
        self.output_tensors = output_tensors
        self.output_shapes = output_shapes

         # since RetinaFace Net has only one input, set input width and height for preprocessing to use
        self.input_w = input_w
        self.input_h = input_h

        # create bmcv handle
        self.bmcv = sail.Bmcv(self.handle)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)

        logger.info("===========================================")
        logger.info("BModel: {}".format(bmodel_file_path))
        logger.info("Input : {}, {}".format(input_shapes, input_dtypes))
        logger.info("Output: {}, {}".format(output_shapes, output_dtypes))
        logger.info("===========================================")

        self.keep_top_k = 50
        # self.keep_top_k = 750
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        # self.ab_bgr = [x * self.input_scale for x in [1, -104, 1, -117, 1, -123]]
        self.mean_bgr = (104, 117, 123)

        self.cfg = cfg
        priorbox = PriorBox(cfg, image_size=(self.input_h, self.input_w))
        self.priors = priorbox.forward()
        
        self.dt = 0.0

    @timeit
    def preprocess_with_opencv(self, img_raw):

        """
        description: preprocess the image with opencv
        steps:
            1. resize: First resize the long side to 640 and then pad the short side to 640.
            2. normalization
            3. HWC -> NCHW
        """

        ###预处理第一步：resize
        #1.先将长边resize到640
        target_size = 640
        im_shape = img_raw.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size)/float(im_size_max)
        img = cv2.resize(img_raw, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        #2.再将短边pad到640
        W,H = self.input_w, self.input_h
        top = 0 
        bottom = H - img.shape[0]
        left = 0 
        right = W - img.shape[1]
        pad_img = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(0,0,0))
        pad_img = np.float32(pad_img)

        ### 预处理第二步： 减均值
        pad_img -= self.mean_bgr

        ### 预处理第三步：HWC -> NCHW
        # 1.HWC to CHW format:
        pad_img = pad_img.transpose(2, 0, 1)
        # 2.CHW to NCHW format
        pad_img = np.expand_dims(pad_img, axis=0)
        
        # # 预处理第四步：Convert the image to row-major order, also known as "C order":
        pad_img = np.ascontiguousarray(pad_img)

        return pad_img,resize

    @timeit
    def infer_numpy(self, input_data, USE_NP_FILE_AS_INPUT=False):

        if USE_NP_FILE_AS_INPUT:
            logger.debug("use numpy data as input")
            input_data = np.load("./np_input.npy")
        else:
            logger.debug("use decoder data as input")

        inputs = {self.input_name: np.array(input_data, dtype=np.float32)}
        t0 = time.time()
        output = self.engine.process(self.graph_name, inputs)
        self.dt += time.time() - t0
        return list(output.values())

    @timeit
    def postprocess(self, outputs, resize):
        """
        description: postprocess the prediction
        param:
            output:  A tensor likes [num_boxes,x1,y1,x2,y2,conf,landmark_x1,landmark_y1,
            landmark_x2,landmark_y2,...]
            resize:  The ratio of resize
        """
        logger.debug("outputs size: {}".format(len(outputs)))
        logger.debug("outputs size: {}".format(len(outputs)))
        # [1, 25500, 4], bbox, xywh
        # [1, 25500, 10] landmarks, x1,y1,x2,y2,x3,y3,x4,y4,x5,y5
        # [1, 25500, 2]  conf

        logger.debug("output tensor 0 = {} , output tensor 1 = {}, output tensor 2 = {} ".format(
            outputs[0].shape, outputs[1].shape, outputs[2].shape))

        logger.debug("output tensor 0 = {} , output tensor 1 = {}, output tensor 2 = {} ".format(
            outputs[0].shape, outputs[1].shape, outputs[2].shape))

        # get tensor by shape
        for i in range(3):
            if outputs[i].shape[-1] == 2:
                conf = outputs[i]
            elif outputs[i].shape[-1] == 4:
                loc = outputs[i]
            else:
                landms = outputs[i] 

        logger.debug("loc = {} , landms = {}, conf = {} ".format(loc.shape, landms.shape, conf.shape))
        logger.debug("loc = {} , landms = {}, conf = {} ".format(loc.shape, landms.shape, conf.shape))

        
        scale = np.array([self.input_w,self.input_h,self.input_w,self.input_h])
        boxes = decode(loc.squeeze(0), self.priors, self.cfg['variance'])
        boxes = boxes * scale / resize
       
        scores = conf.squeeze(0)[:, 1]

        landms = decode_landm(landms.squeeze(0), self.priors, self.cfg['variance'])

        scale1 = np.array([self.input_w, self.input_h, self.input_w, self.input_h,
                           self.input_w, self.input_h, self.input_w, self.input_h,
                           self.input_w, self.input_h])

        landms = landms * scale1 / resize
        logger.debug("after output decode: boxes = {} , landmarks = {} , scores = {} ".format(boxes.shape, landms.shape, scores.shape))

        # filter
        inds = np.where(scores >= self.score_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        landms = landms[inds]
        logger.debug("after threshold filter: boxes = {} , landmarks = {} , scores = {} ".format(boxes.shape, landms.shape, scores.shape))

        # keep top-K before NMS
        # order = scores.argsort()[::-1][:self.keep_top_k]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        logger.debug("after keep-topk: boxes = {} , landmarks = {} , scores = {} ".format(boxes.shape, landms.shape, scores.shape))

        # do NMS
        boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(boxes, self.nms_threshold)
        result_boxes = boxes[keep, :]
        result_landmarks = landms[keep, :]
        logger.debug("after nms: result_boxes = {} , result_landmarks = {} ".format( \
            result_boxes.shape, result_landmarks.shape))
        # dets = np.concatenate((result_boxes, result_landmarks), axis=1)
        # logger.debug("after nms: result_dets = {}".format( \
        #     dets.shape))
        return result_boxes, result_landmarks

    @timeit
    def postprocess_batch(self, outputs, resize_list):
        
        result_boxes_list = []
        result_landmarks_list = []
        confs=[]
        locs=[]
        landmss=[]

        ##这一步是当图片数小于batch数时，需要补充
        if len(resize_list)<self.batch_size:
            for i in range(self.batch_size-len((resize_list))):
                resize_list.append(1)              
     
        for output in outputs:
            for i in range(self.batch_size):
                if output.shape[-1] == 2:
                    confs.append(output[i])
                    confs[i] = np.expand_dims(confs[i], axis=0)
                    
                if output.shape[-1] == 4:
                    locs.append(output[i])
                    locs[i] = np.expand_dims(locs[i], axis=0)
                    
                if output.shape[-1] == 10:
                    landmss.append(output[i])
                    landmss[i] = np.expand_dims(landmss[i], axis=0)
                    

        for i in range(self.batch_size):
            conf = confs[i]
            loc  = locs[i]
            landms = landmss[i]
            resize = resize_list[i]

            scale = np.array([self.input_w,self.input_h,self.input_w,self.input_h])
            boxes = decode(loc.squeeze(0), self.priors, self.cfg['variance'])
            boxes = boxes * scale / resize
        
            scores = conf.squeeze(0)[:, 1]

            landms = decode_landm(landms.squeeze(0), self.priors, self.cfg['variance'])

            scale1 = np.array([self.input_w, self.input_h, self.input_w, self.input_h,
                           self.input_w, self.input_h, self.input_w, self.input_h,
                           self.input_w, self.input_h])

            landms = landms * scale1 / resize

            logger.debug("after output decode: boxes = {} , landmarks = {} , scores = {} ".format(boxes.shape, landms.shape, scores.shape))

            # filter
            inds = np.where(scores >= self.score_threshold)[0]
            boxes = boxes[inds]
            scores = scores[inds]
            landms = landms[inds]

            logger.debug("after threshold filter: boxes = {} , landmarks = {} , scores = {} ".format(boxes.shape, landms.shape, scores.shape))

            # keep top-K before NMS
            order = scores.argsort()[::-1]
            # order = scores.argsort()[::-1][:self.keep_top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            logger.debug("after keep-topk: boxes = {} , landmarks = {} , scores = {} ".format(boxes.shape, landms.shape, scores.shape))

            # do NMS
            boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(boxes, self.nms_threshold)
            result_boxes = boxes[keep, :]
            result_landmarks = landms[keep, :]
            logger.debug("after nms: result_boxes = {} , result_landmarks = {} ".format( \
                 result_boxes.shape, result_landmarks.shape))
            # dets = np.concatenate((result_boxes, result_landmarks), axis=1)

            result_boxes_list.append(result_boxes)
            result_landmarks_list.append(result_landmarks)
            # result_dets_list.append(dets)

        return result_boxes_list,result_landmarks_list


    def predict_numpy(self, frame):

        if not isinstance(frame, type(None)):

            # Do image preprocess
            img, resize = self.preprocess_with_opencv(frame)
            # Do inference
            outputs = self.infer_numpy(img)
            # Do postprocess
            result_boxes, result_landmarks = self.postprocess(outputs, resize)

            # Draw rectangles and labels
            result_image = frame.copy()

            # Save image
            for i in range(len(result_boxes)):
                box = result_boxes[i]
                if box[4] < 0.5:
                    continue

                landmark = result_landmarks[i]
                logger.info("face {}: x1, y1, x2, y2, conf = {}".format(i + 1, box))
            
                draw_one(
                        box,
                        landmark,
                        result_image,
                        label="{}:{:.2f}".format( 'Face', box[4]))
            
            return result_image

    def predict_batch(self, img_list):
        img_num = len(img_list)
        img_input_list = []
        resize_list = []

        for img in img_list:
            img, resize = self.preprocess_with_opencv(img)
            img_input_list.append(img)
            resize_list.append(resize)

        if img_num == self.batch_size:
            input_img = np.vstack(img_input_list)
            outputs = self.infer_numpy(input_img)
        else:
            input_img = np.zeros(self.input_shape,dtype='float32')
            input_img[:img_num] = np.vstack(img_input_list)
            outputs = self.infer_numpy(input_img)

        result_boxes_list,result_landmarks_list= self.postprocess_batch(outputs, resize_list)

        return result_boxes_list,result_landmarks_list
        
    def get_time(self):
        return self.dt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Demo of RetinaFace with preprocess by OpenCV")

    parser.add_argument('--bmodel',
                        type=str,
                        default="../data/models/BM1684X/retinaface_mobilenet0.25_fp32_1b.bmodel",               
                        required=False,
                        help='bmodel file path.')

    parser.add_argument('--network',
                        type=str,
                        default="mobile0.25",
                        required=False,
                        help='backbone network type: mobile0.25 , resnet50.')
    
        
    parser.add_argument('--input_path',
                        type=str,
                        default="../data/images/WIDERVAL",
                        required=False,
                        help='input pic/video file path.')

    parser.add_argument('--tpu_id',
                        default=0,
                        type=int,
                        required=False,
                        help='tpu dev id(0,1,2,...).')

    parser.add_argument("--conf",
                        default=0.02,
                        type=float,
                        help="test conf threshold.")

    parser.add_argument("--nms",
                        default=0.3,
                        type=float,
                        help="test nms threshold.")

    parser.add_argument('--use_np_file_as_input',
                        default=False,
                        type=bool,
                        required=False,
                        help="whether use dumped numpy file as input.")

    opt = parser.parse_args()

    logger.remove() # remove default handler to avoid from repeated output log
    handler_id = logger.add(sys.stderr, level="INFO") # add a new handler


    cfg = None
    if opt.network == "mobile0.25":
        cfg = cfg_mnet
    elif opt.network == "resnet50":
        cfg = cfg_re50

    retinaface = Retinaface_sophon(
        cfg = cfg,
        bmodel_file_path=opt.bmodel,
        tpu_id=opt.tpu_id,
        score_threshold=opt.conf,
        nms_threshold=opt.nms)
    
    batch_size = retinaface.batch_size


    img_list = []
    filename_list = []

    # if it is a single file
    if os.path.isfile(opt.input_path):
        if not os.path.exists("./results/"):
            os.makedirs("./results/")
        frame = cv2.imread(opt.input_path)
        if frame is not None:   # if it is a pic
            result_image = retinaface.predict_numpy(frame)
            cv2.imwrite(os.path.join("results", os.path.split(opt.input_path)[-1]), result_image)
        else: # if it is a video
            cap = cv2.VideoCapture(opt.input_path)
            id = 0
            end_flag = False
            while not end_flag:
                ret, frame = cap.read()
                if not ret:
                    if len(img_list) == 0:
                        break
                    end_flag = True
                else:
                    img_list.append(frame)
                if(len(img_list)==batch_size or end_flag):
                    res1, res2= retinaface.predict_batch(img_list)
                    for i in range(len(img_list)):
                        for j in range(res1[i].shape[0]):
                            x = int(res1[i][j][0])
                            y = int(res1[i][j][1])
                            w = int(res1[i][j][2])-int(res1[i][j][0])
                            h = int(res1[i][j][3])-int(res1[i][j][1])
                            confidence = str(res1[i][j][4])
                            logger.info("face {}: x,y,w,h,conf = {}".format(j+1, str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence))
                    for i in range(len(img_list)):
                        result_image = img_list[i]
                        for j in range(res1[i].shape[0]):
                            box = res1[i][j]
                            landmark = res2[i][j]
                            if box[4]<0.5:
                                continue
                            draw_one(
                                box,
                                landmark,
                                result_image,
                                label="{}:{:.2f}".format('Face',box[4])
                            )
                        cv2.imwrite(os.path.join("results", str(id)+".jpg"), result_image)
                        id += 1
                    img_list = []
            cap.release()
    
    else:   # if it is a folder
        save_path = os.path.join(os.path.dirname(
        __file__), "results", os.path.split(opt.bmodel)[-1] + "_opencv_" + os.path.split(opt.input_path)[-1] + "_python_result")

        os.makedirs(save_path, exist_ok=True)
        save_name = save_path + ".txt"
        dt1 = 0.0
        num = 0
        for root, dirs, filenames in os.walk(opt.input_path):
            for filename in filenames:
                img_file = os.path.join(root, filename)
                src_img = cv2.imread(img_file,cv2.IMREAD_COLOR)     
                if src_img is not None:
                    img_list.append(src_img)
                    filename_list.append(filename)
                    if(len(img_list)==batch_size):
                        t1 = time.time()
                        res1, res2= retinaface.predict_batch(img_list)
                        dt1 += time.time()-t1
                        with open(save_name,"a") as fd:
                            for i in range(batch_size):
                                # logger.info("Detect {} faces:".format(res1[i].shape[0]))
                                fd.write(filename_list[i]+"\n")
                                fd.write(str(res1[i].shape[0])+"\n")
                                for j in range(res1[i].shape[0]):
                                    x = int(res1[i][j][0])
                                    y = int(res1[i][j][1])
                                    w = int(res1[i][j][2])-int(res1[i][j][0])
                                    h = int(res1[i][j][3])-int(res1[i][j][1])
                                    confidence = str(res1[i][j][4])
                                    fd.write(str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n")
                                    logger.info("face {}: x,y,w,h,conf = {}".format(j+1, str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence))
                        for i in range(batch_size):
                            result_image = img_list[i]
                            for j in range(res1[i].shape[0]):
                                box = res1[i][j]
                                landmark = res2[i][j]
                                if box[4]<0.5:
                                    continue
                                draw_one(
                                    box,
                                    landmark,
                                    result_image,
                                    label="{}:{:.2f}".format('Face',box[4])
                                )
                                cv2.imwrite(os.path.join(save_path,filename_list[i].split(".")[0]+"_opencv_out.jpg"), result_image)
                        img_list = []
                        filename_list = []
                num = num + 1
        if len(img_list):
            t2 = time.time()
            res1, res2= retinaface.predict_batch(img_list)
            dt1 += time.time()-t2
            with open(save_name,"a") as fd:
                for i in range(len(img_list)):
                    # logger.info("Detect {} faces:".format(res1[i].shape[0]))
                    fd.write(filename_list[i]+"\n")
                    fd.write(str(res1[i].shape[0])+"\n")
                    for j in range(res1[i].shape[0]):
                        x = int(res1[i][j][0])
                        y = int(res1[i][j][1])
                        w = int(res1[i][j][2])-int(res1[i][j][0])
                        h = int(res1[i][j][3])-int(res1[i][j][1])
                        confidence = str(res1[i][j][4])
                        fd.write(str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n")
                        logger.info("face {}: x,y,w,h,conf = {}".format(j+1, str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence))
            for i in range(len(img_list)):
                result_image = img_list[i]
                for j in range(res1[i].shape[0]):
                    box = res1[i][j]
                    landmark = res2[i][j]
                    if box[4] < 0.5:
                        continue
                    draw_one(
                        box,
                        landmark,
                        result_image,
                        label="{}:{:.2f}".format('Face',box[4])
                    )
                    cv2.imwrite(os.path.join(save_path,filename_list[i].split(".")[0]+"_opencv_out.jpg"), result_image)

        #########calculate speed#####################
        cn = num
        logger.info("------------------ Inference Time Info ----------------------")
        inference_time = retinaface.get_time() / cn
        logger.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
        total_time = dt1
        logger.info("total_time(ms): {:.2f}, img_num: {}".format(total_time * 1000, cn))
        # average_latency = total_time / cn
        # qps = 1 / average_latency
        # logger.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))
        average_latency_ = retinaface.get_time() / cn
        qps_ = 1 / average_latency_
        logger.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency_ * 1000, qps_))


    print("===================================================")

    from utils.time_utils import TimeStamp

    TimeStamp().print()
