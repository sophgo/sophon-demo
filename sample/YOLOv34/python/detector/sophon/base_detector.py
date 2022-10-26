#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import cv2
import six
import time
import requests
import numpy as np
# from ipdb import set_trace as pause

import sys; sys.path.append("..")
from utils.utils import draw_zh_cn
from utils.logger import logger

import sophon.sail as sail

class BaseDetector(object):
    def __init__(self, config):

        # Create a Context on sophon device
        logger.info("config : {}".format(config))
        tpu_count = sail.get_available_tpu_num()
        logger.debug('{} TPUs Detected, using TPU {} \n'.format(tpu_count, config.DEV_ID))
        self.engine = sail.Engine(config.ENGINE_FILE, config.DEV_ID, sail.IOMode.SYSIO)
        self.handle = self.engine.get_handle()
        self.graph_name = self.engine.get_graph_names()[0]
        graph_count = len(self.engine.get_graph_names())
        logger.warning("{} graphs in {}, using {}".format(graph_count, config.ENGINE_FILE, self.graph_name))

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
        
        self.outputs = outputs
        self.output_names = output_names
        self.output_tensors = output_tensors
        self.output_shapes = output_shapes

         # since YOLOv3/4 has only one input, set input width and height for preprocessing to use
        self.input_w = input_w
        self.input_h = input_h

        # create bmcv handle
        self.bmcv = sail.Bmcv(self.handle)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)

        logger.info("===========================================")
        logger.info("BModel: {}".format(config.ENGINE_FILE))
        logger.info("Input : {}, {}".format(input_shapes, input_dtypes))
        logger.info("Output: {}, {}".format(output_shapes, output_dtypes))
        logger.info("===========================================")

        self.min_confidence = config.MIN_CONFIDENCE
        self.nms_max_overlap = config.NMS_MAX_OVERLAP
        self.min_width = config.MIN_WIDTH
        self.min_height = config.MIN_HEIGHT

    def infer_numpy(self, input_data, USE_NP_FILE_AS_INPUT=False):

        output_nps = []  # inference output

        if USE_NP_FILE_AS_INPUT:
            logger.warning("use numpy data as input")
            input_data = np.load("./np_input.npy")
        else:
            logger.warning("use decoder data as input")
        
        logger.debug("input_data shape: {}".format(input_data.shape))

        inputs = {self.input_name: np.array(input_data, dtype=np.float32)}
        output = self.engine.process(self.graph_name, inputs)

        logger.debug(output.keys())

        return list(output.values())

    def infer_bmimage(self, data, USE_NP_FILE_AS_INPUT=False):

        output_nps = []  # inference output

        if USE_NP_FILE_AS_INPUT:
            
            ref_data = np.load("./np_input.npy")
            logger.info("using numpy data as input: {}".foramt(ref_data.shape))

            input = sail.Tensor(self.handle, ref_data)
            input_tensors = {self.input_name: input}
            input_shapes = {self.input_name: self.input_shape}

            logger.debug("engine process start")
            self.engine.process(self.graph_name, input_tensors, input_shapes, self.output_tensors)
            logger.debug("engine process end")
        else:
            logger.info("using decoder data as input")
            self.bmcv.bm_image_to_tensor(data, self.inputs[0])
            logger.debug("engine process start")
            self.engine.process(self.graph_name, self.input_tensors, self.output_tensors)
            logger.debug("engine process end")
        
        # convert output tensor to numpy, 遍历output_names过程中会将输出按照名称排序      
        output_nps = [output_tensor.asnumpy(self.output_shapes[output_name]) \
            for output_name, output_tensor in self.output_tensors.items()]

        return output_nps
    
    def __call__(self, frame):
        pass

    def _filter_confidence(self, bboxes):
        pass

    def _filter_small_boxes(self, bboxes_tlbr, scores, labels):
        if bboxes_tlbr is not None:
            w = bboxes_tlbr[:, 2] - bboxes_tlbr[:, 0]
            h = bboxes_tlbr[:, 3] - bboxes_tlbr[:, 1]
            keep = np.where((w >= self.min_width) & (h >= self.min_height))[0]
            bboxes_tlbr = bboxes_tlbr[keep]
            scores = scores[keep]
            labels = labels[keep]
        return bboxes_tlbr, scores, labels

    def _bbox_draw(self, frame, bboxes_tlbr, scores, labels):
        if bboxes_tlbr is not None:
            for i in range(bboxes_tlbr.shape[0]):
                x1, y1, x2, y2 = bboxes_tlbr[i]
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                text = str(round(scores[i], 2)) +" "+ labels[i]
                # frame = draw_zh_cn(frame, str(score+" "+labels[i]), (255,0,0), (x1, y1-10))
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    def _nms(self, bboxes_tlbr, scores=None):
        if len(bboxes_tlbr) == 0: return []

        bboxes_tlbr = bboxes_tlbr.astype(np.float)
        pick_indices = []

        x1 = bboxes_tlbr[:, 0]
        y1 = bboxes_tlbr[:, 1]
        x2 = bboxes_tlbr[:, 2]
        y2 = bboxes_tlbr[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if scores is not None:
            idxs = np.argsort(scores)
        else:
            idxs = np.argsort(y2)

        while len(idxs) > 0:
            # The index of largest confidence score and pick it 
            last = len(idxs) - 1
            i = idxs[last]
            pick_indices.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(
                    ([last], np.where(overlap > self.nms_max_overlap)[0])
                    )
                    )

        return bboxes_tlbr[pick_indices].astype("int")

    @classmethod
    def tlbr_to_xywh(tlbr):
        """center point: xywh"""
        x1, y1, x2, y2 = tlbr
        w, h = int(x2-x1), int(y2-y1)
        x, y = int((2*x1 +w)/2), int((2*y1 +h)/2)
        return x,y,w,h

    def _xywh_to_tlbr(self, xywh):
        cx, cy, w, h = xywh
        xmin = int(cx-w/2)
        ymin = int(cy-h/2)
        xmax = int(cx+w/2)
        ymax = int(cy+h/2)
        return xmin, ymin, xmax, ymax

    def _tlwh_to_tlbr(self, tlwh):
        xmin, ymin, w, h = tlwh
        xmax = int(xmin+w)
        ymax = int(ymin+h)
        xmin = int(xmin)
        ymin = int(ymin)
        return xmin, ymin, xmax, ymax

    def _cut_to_chest(self, bboxes_tlbr):
        new_bboxes_tlbr = []
        if bboxes_tlbr.shape[0]>0:
            for idx in range(bboxes_tlbr.shape[0]):
                det = bboxes_tlbr[idx, :]
                x1, y1, x2, y2 = det
                cx,cy,w,h = self._tlbr_to_xywh(det)
                new_bboxes_tlbr.append([x1, y1, x2, y1+h/3.])
        return np.array(new_bboxes_tlbr).astype(int)
