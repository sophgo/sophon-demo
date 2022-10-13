#-*-coding:utf-8-*-
from __future__ import division
from cProfile import label
# from other.function_other import *
import argparse
import os
import time
import cv2

from detector import DetectFactory

from configs.config import update_config

from utils.logger import logger

opt = None
save_path = os.path.join(os.path.dirname(
    __file__), "result_imgs", os.path.basename(__file__).split('.')[0])

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# args = update_config(BASE_DIR + "/configs/yolov3_416.yml")
# args = update_config(BASE_DIR + "/configs/yolov3_608.yml")
# args = update_config(BASE_DIR + "/configs/yolov4_416.yml")
# args = update_config(BASE_DIR + "/configs/yolov4_608.yml")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Demo of YOLOV3/4 with preprocess by PIL")

    parser.add_argument('--cfgfile',
                        type=str,
                        default="./configs/yolov4_416.yml",
                        required=False,
                        help='config file path.')

    parser.add_argument('--input',
                        type=str,
                        default="../data/images/person.jpg",
                        required=False,
                        help='input pic file path.')    
    
    opt = parser.parse_args()

    save_path = os.path.join(
        save_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    )

    os.makedirs(save_path, exist_ok=True)

    args = update_config(opt.cfgfile)

    # 1.load yolov3/4-coco model
    detector = DetectFactory(name=args.DETECTOR.NAME, config=args.DETECTOR)
    logger.info("Yolov3/4 model load successful!")

    # 2. read image
    frame = cv2.imread(opt.input)

    # assert frame is None

    logger.info(frame.shape)

    # 3. inference
    detected_boxes, result_image = detector.detect(frame)

    # 4. print result bbox info
    logger.info(detected_boxes)

    # 5. save result image
    cv2.imwrite(os.path.join(save_path, os.path.splitext(os.path.split(opt.input)[-1])[0] + "_result.jpg"), result_image)
