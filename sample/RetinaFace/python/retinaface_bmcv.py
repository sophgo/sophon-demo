#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""
A Retinaface demo using Sophon SAIL api to make inferences.
"""

# -*- coding: utf-8 -*
import argparse
from ctypes import resize
import os
import sys
import time

import cv2
import numpy as np

import sophon.sail as sail

from data.config import cfg_mnet, cfg_re50
from utils.box_utils import PriorBox, decode, decode_landm, draw_one_on_bmimage, py_cpu_nms, draw_one
from utils.print_utils import print_info
from utils.time_utils import timeit

from loguru import logger

opt = None

# 设置numpy运算精度
# np.set_printoptions(threshold=np.inf)

class Retinaface_sophon(object):
    """
    description: A RetineFace class that warps Sophon ops, preprocess and postprocess ops.
    """

    def __init__(self, cfg, bmodel_file_path, tpu_id, score_threshold = 0.5, nms_threshold = 0.3):
        """
        :param cfg: retinaface backbone cfg file
        :param bmodel_file_path: bmodel file
        :param tpu_id: tpu id
        :param score_threshold: confidence
        :param nms_threshold: nms
        """
        # Create a Context on sophon device
        tpu_count = sail.get_available_tpu_num()
        logger.debug('{} TPUs Detected, using TPU {} \n'.format(tpu_count, tpu_id))
        self.engine = sail.Engine(bmodel_file_path, tpu_id, sail.IOMode.SYSO)
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

         # since Retinaface Net has only one input, set input width and height for preprocessing to use
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
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        self.ab_bgr = [x * self.input_scale for x in [1, -104, 1, -117, 1, -123]]
        # self.mean_bgr = (104, 117, 123)

        self.cfg = cfg
        priorbox = PriorBox(cfg, image_size=(self.input_h, self.input_w))
        self.priors = priorbox.forward()
        
        self.dt = 0.0

    @timeit
    def preprocess_with_bmcv(self, bm_image):
        """
        description: preprocess the input bm_image, resize and pad it to target size,
                     normalize to [0,1],transform to NCHW format.
        param:
            img: bmimage
        return:
            image:  the processed bm_image
            resize: the ratio of resize
        """
        if bm_image.format()==sail.Format.FORMAT_YUV420P:
            img = self.bmcv.yuv2bgr(bm_image)
        else:
            img = bm_image

        img_w = img.width()
        img_h = img.height()
        r_w = self.input_w / img_w
        r_h = self.input_h / img_h

        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * img_h)
            tx1 = tx2 = 0
            ty1 = 0
            ty2 = self.input_h - th

        else:
            tw = int(r_h * img_w)
            th = self.input_h
            tx1 = 0
            tx2 = self.input_w - tw
            ty1 = ty2 = 0

        attr = sail.PaddingAtrr()
        attr.set_stx(tx1)
        attr.set_sty(ty1)
        attr.set_w(tw)
        attr.set_h(th)
        attr.set_r(0)
        attr.set_g(0)
        attr.set_b(0)

        resized_img_bgr = bmcv.vpp_crop_and_resize_padding(img,
                                    0, 0, img.width(), img.height(),
                                    self.input_w, self.input_h, attr)

        input_img = sail.BMImage(self.handle, self.input_h, self.input_w,
                                           sail.Format.FORMAT_BGR_PLANAR, self.img_dtype)
        self.bmcv.convert_to(
            resized_img_bgr, input_img, \
            ((self.ab_bgr[0], self.ab_bgr[1]), (self.ab_bgr[2], self.ab_bgr[3]), (self.ab_bgr[4], self.ab_bgr[5])))

        resize = r_w if r_h > r_w else r_h 
    
        return input_img, resize
        

    @timeit
    def infer_bmimage(self, data, USE_NP_FILE_AS_INPUT=False):

        output_nps = []  # inference output

        if USE_NP_FILE_AS_INPUT:
            
            ref_data = np.load("./np_input.npy")
            logger.debug("using numpy data as input: {}".foramt(ref_data.shape))

            input = sail.Tensor(self.handle, ref_data)
            input_tensors = {self.input_name: input}
            input_shapes = {self.input_name: self.input_shape}

            logger.debug("engine process start")
            self.engine.process(self.graph_name, input_tensors, input_shapes, self.output_tensors)
            logger.debug("engine process end")
        else:
            logger.debug("using decoder data as input")
            self.bmcv.bm_image_to_tensor(data, self.inputs[0])
            logger.debug("engine process start")
            t0 = time.time()
            self.engine.process(self.graph_name, self.input_tensors, self.output_tensors)
            self.dt += time.time() - t0
            logger.debug("engine process end")
        
        # convert output tensor to numpy, sort by output_names      
        output_nps = [output_tensor.asnumpy(self.output_shapes[output_name]) \
            for output_name, output_tensor in self.output_tensors.items()]

        return output_nps

    @timeit
    def postprocess(self, outputs, resize):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,x1,y1,x2,y2,conf,landmark_x1,landmark_y1,
            landmark_x2,landmark_y2,...]
            resize: the ratio of resize
        """

        logger.debug("outputs size: {}".format(len(outputs)))
        # [1, 25500, 4], bbox, xywh
        # [1, 25500, 10] landmarks, x1,y1,x2,y2,x3,y3,x4,y4,x5,y5
        # [1, 25500, 2]  conf

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
        
        # decode output
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

        return result_boxes, result_landmarks

        
    @timeit
    def postprocess_batch(self, outputs, resize_list):
        
        result_boxes_list = []
        result_landmarks_list = []
        confs=[]
        locs=[]
        landmss=[]

        if len(resize_list)<self.batch_size:
            for i in range(self.batch_size-len(resize_list)):
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

            result_boxes_list.append(result_boxes)
            result_landmarks_list.append(result_landmarks)

        return result_boxes_list,result_landmarks_list


    def predict_bmimage(self, frame):

        if not isinstance(frame, type(None)):

            # Do image preprocess
            img, resize = self.preprocess_with_bmcv(frame)

            # Do inference
            outputs = self.infer_bmimage(img)

            # Do postprocess
            result_boxes, result_landmarks = self.postprocess(outputs, resize)

            # Draw rectangles and labels on the original image
            result_image = frame

            # Save image
            for i in range(len(result_boxes)):
                box = result_boxes[i]
                landmark = result_landmarks[i]
                if box[4] < 0.5:
                    continue

                logger.info("face {}: x1, y1, x2, y2, conf = {}".format(i, box))
                
                draw_one_on_bmimage(
                    self.bmcv,
                    box,
                    landmark,
                    result_image,
                    label="{}:{:.2f}".format( 'Face', box[4]))
            
            return result_image


    def predict_batch(self, bmimg_list):

        img_num = len(bmimg_list)
        resize_list = []

        if self.batch_size == 1:
            for bmimg in bmimg_list:
                img, resize = self.preprocess_with_bmcv(bmimg)
                resize_list.append(resize)
                outputs = self.infer_bmimage(img)
                res1, res2 = self.postprocess_batch(
                outputs, resize_list
                )
        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()

            for i in range(img_num):
                img,resize = self.preprocess_with_bmcv(bmimg_list[i])
                bmimgs[i] = img.data()
                resize_list.append(resize)
            outputs = self.infer_bmimage(bmimgs)
            res1, res2 = self.postprocess_batch(outputs, resize_list)

        return res1,res2
        
    def get_time(self):

        return self.dt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Demo of RetinaFace with preprocess by BMCV")

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
    bmcv = retinaface.bmcv

    bmimg_list = []
    filename_list = []


    # if it is a file
    if os.path.isfile(opt.input_path):
        if not os.path.exists("./results/"):
            os.makedirs("./results/")
        print("This is a file")
        frame = cv2.imread(opt.input_path,0)
        if frame is not None:    # it is a pic
            decoder = sail.Decoder(opt.input_path,False,0)
            input_bmimg = sail.BMImage()
            ret = decoder.read(retinaface.handle,input_bmimg)
            if ret:
                logger.error("decode error\n")
                exit(-1)
            result_image = retinaface.predict_bmimage(input_bmimg)
            retinaface.bmcv.imwrite(os.path.join("results",os.path.split(opt.input_path)[-1]),result_image)
        else:       # it is a video
            decoder = sail.Decoder(opt.input_path,True,0)
            if decoder.is_opened():
                logger.info("create decoder success")
                input_bmimg = sail.BMImage()
                id = 0
                while True:
                    print("This is a video")
                    ret = decoder.read(retinaface.handle,input_bmimg)
                    if ret:
                        break
                    result_image = retinaface.predict_bmimage(input_bmimg)
                    retinaface.bmcv.imwrite(os.path.join("results", str(id)+".jpg"),result_image)
                    id += 1
                print("stream end")
            else:
                logger.error("failed to create decoder")
    # it is a folder
    else:
        save_path = os.path.join(os.path.dirname(
        __file__), "results", os.path.split(opt.bmodel)[-1] + "_bmcv_" + os.path.split(opt.input_path)[-1] + "_python_result")

        os.makedirs(save_path, exist_ok=True)
        save_name = save_path + ".txt"
        t1 = time.time()
        num = 0
        dt1 = 0.0
        for root, dirs, filenames in os.walk(opt.input_path):
            for filename in filenames:
                img_file = os.path.join(root,filename)
                src_img = cv2.imread(img_file,cv2.IMREAD_COLOR)
                if src_img is not None:   # pic
                    decoder = sail.Decoder(img_file, False, 0)        #compressed = false
                    input_bmimg = sail.BMImage()
                    ret = decoder.read(retinaface.handle, input_bmimg)
                    if ret:
                        logger.error("decode error\n")
                        continue
                    bmimg_list.append(input_bmimg)
                    filename_list.append(filename)
                    if (len(bmimg_list)==batch_size):
                        t1 = time.time()
                        res1, res2= retinaface.predict_batch(bmimg_list)
                        dt1 += time.time()-t1
                    
                        with open(save_name,"a") as fd:
                            for i in range(batch_size):
                                # logger.info("Detect {} faces :".format(res1[i].shape[0]))
                                fd.write(filename_list[i]+"\n")
                                fd.write(str(res1[i].shape[0])+"\n")
                                for j in range(res1[i].shape[0]):
                                    x = int(res1[i][j][0])
                                    y = int(res1[i][j][1])
                                    w = int(res1[i][j][2])-int(res1[i][j][0])
                                    h = int(res1[i][j][3])-int(res1[i][j][1])
                                    confidence = str(res1[i][j][4])
                                    fd.write(str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n")
                                    logger.info("face {}: x, y, w, h, conf = {}".format(j+1,str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence))
                        for i in range(batch_size):
                            result_image = bmimg_list[i]
                            for j in range(res1[i].shape[0]):
                                box = res1[i][j]
                                landmark = res2[i][j]
                                if box[4] < 0.5:
                                    continue
                                draw_one_on_bmimage(
                                    bmcv,
                                    box,
                                    landmark,
                                    result_image,
                                    label="{}:{:.2f}".format( 'Face', box[4]))
                                retinaface.bmcv.imwrite(os.path.join(save_path, filename_list[i].split(".")[0]+"_bmcv_out.jpg"), result_image)
                        bmimg_list = []
                        filename_list = []         
                num = num +1
        if len(bmimg_list):
            t2 = time.time()
            res1, res2= retinaface.predict_batch(bmimg_list)
            dt1 += time.time() - t2
            with open(save_name,"a") as fd:
                for i in range(len(bmimg_list)):
                    # logger.info("Detect {} faces :".format(res1[i].shape[0]))
                    fd.write(filename_list[i]+"\n")
                    fd.write(str(res1[i].shape[0])+"\n")
                    for j in range(res1[i].shape[0]):
                        x = int(res1[i][j][0])
                        y = int(res1[i][j][1])
                        w = int(res1[i][j][2])-int(res1[i][j][0])
                        h = int(res1[i][j][3])-int(res1[i][j][1])
                        confidence = str(res1[i][j][4])
                        fd.write(str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n")
                        logger.info("face {}: x, y, w, h, conf = {}".format(j+1,str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence))
            for i in range(len(bmimg_list)):
                result_image = bmimg_list[i]
                for j in range(res1[i].shape[0]):
                    box = res1[i][j]
                    landmark = res2[i][j]
                    if box[4]<0.5:
                        continue
                    draw_one_on_bmimage(
                        bmcv,
                        box,
                        landmark,
                        result_image,
                        label="{}:{:.2f}".format( 'Face', box[4]))
                    retinaface.bmcv.imwrite(os.path.join(save_path, filename_list[i].split(".")[0]+"_bmcv_out.jpg"), result_image)

        ########calculate speed############################################################################
        cn = num
        logger.info("------------------ Inference Time Info ----------------------")
        inference_time = retinaface.get_time() / cn
        logger.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
        # total_time = t2 - t1
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
