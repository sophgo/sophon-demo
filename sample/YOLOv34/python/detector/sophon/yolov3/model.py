#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""
A YOLOv3/v4 demo using Sophon SAIL api to make inferences.
"""
#-*-coding:utf-8-*-
import sophon.sail as sail

from detector.sophon.yolov3.data_processing import PreprocessYOLO,PostprocessYOLO

from detector.sophon.base_detector import BaseDetector

from utils.colors import _COLORS

import numpy as np
import cv2
import os
import time

import logging
logger = logging.getLogger()

class YOLOV3(BaseDetector):
    def __init__(self, config):

        super(YOLOV3, self).__init__(config)

        self.output_tensor_channels= config.OUTPUT_TENSOR_CHANNELS
        self.engine_file_path = config.ENGINE_FILE
        self.input_resolution_HW = (self.input_h, self.input_w)
        self.detector_classes = self._load_label_categories(config.LABEL_FILE)
        self.preprocessor = PreprocessYOLO(self.input_resolution_HW)

        if len(config.YOLO_MASKS) == 9:
            yolo_masks_shape = (3, 3)
        elif len(config.YOLO_MASKS) == 6:
            yolo_masks_shape = (2, 3)
        else:
            raise NotImplementedError
        yolo_masks = np.array(config.YOLO_MASKS).reshape(*yolo_masks_shape)

        if len(config.YOLO_ANCHORS) == 18:
            yolo_anchors_shape = (9, 2)
        elif len(config.YOLO_ANCHORS) == 12:
            yolo_anchors_shape = (6, 2)
        else:
            raise NotImplementedError
        yolo_anchors = np.array(config.YOLO_ANCHORS).reshape(*yolo_anchors_shape)

        postprocessor_args = { 
            "yolo_masks": yolo_masks,
            # "yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
            # A list of 3 three-dimensional tuples for the YOLO masks
            "yolo_anchors": yolo_anchors,
            # "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
            #                 (59, 119), (116, 90), (156, 198), (373, 326)],

            "obj_threshold": self.min_confidence,  # Threshold for object coverage, float value between 0 and 1
            "nms_threshold": self.nms_max_overlap,
            "yolo_input_resolution": self.input_resolution_HW[::-1]}

        self.postprocessor = PostprocessYOLO(**postprocessor_args)

    def _load_label_categories(self, label_file_path):
        categories = [line.rstrip('\n') for line in open(label_file_path)]
        return categories

    def bboxes_info(self, image_raw, x_coord, y_coord, width, height):
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))
        bbox = [left,top,right,bottom]
        return bbox


    def detect(self, image):

        result_image = image.copy()
        
        image_raw, image = self.preprocessor.process(image)

        org_h, org_w = image_raw.size

        # Do inference
        s=time.time()
        outputs = self.infer_numpy(image)
        logger.info("yolov3 cost: %f seconds" % (time.time() - s))

        # 根据shape取出相应的tensor，根据shape调整tensor顺序
        out_index = list()
        for i in range(len(outputs)):
            if outputs[i].shape[-1] == self.output_tensor_channels[0]:
                out_index += [0]
            elif outputs[i].shape[-1] == self.output_tensor_channels[1]:
                out_index += [1]
            else:
                out_index += [2]

        outputs = [outputs[ind] for ind in out_index]

        [logger.debug(output.shape) for output in outputs]

        # Run the post-processing algorithms on the outputs and get the bounding box details of detected objects
        boxes, classes, scores =  self.postprocessor.process(outputs, (org_h, org_w))

        object_bboxes = []
        
        #if classes is not None:
        for i in range(len(classes)): # classes的数量就是检测出的目标的数量
            assert classes[i] < len(self.detector_classes)
            classID = classes[i]
            classLabel = self.detector_classes[classes[i]]
            conf = round(scores[i], 3)
            bbox = self.bboxes_info(image_raw, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])

            object_bboxes.append((classLabel,
                                  conf,
                                  bbox))
            
            self.drawPred(result_image, classID, classLabel, conf, bbox[0], bbox[1], bbox[2], bbox[3])

        return object_bboxes, result_image

    def drawPred(self, frame, classId, classLabel, conf, left, top, right, bottom):

        color = (_COLORS[classId] * 255).astype(np.uint8).tolist()

        print("classID=%d, classLabel=%s, conf=%f, bbox=(%d,%d,%d,%d)" %
                (classId, classLabel, conf, left, top, right, bottom))

        cv2.rectangle(frame, (left, top), (right, bottom),
                        color, thickness=4)

        label = '%s:%s' % (classLabel, '%.2f' % conf)
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
        return frame