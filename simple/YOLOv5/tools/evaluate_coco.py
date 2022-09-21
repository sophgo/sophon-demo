import cv2
import os
import sys
import shutil
import numpy as np
import argparse

os.chdir(os.path.abspath(os.path.dirname(sys.argv[0])))
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(__dir__ + "/../")
print(sys.path)

from python.yolov5_opencv import YOLOv5
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
import json


class YoloTest(object):
    def __init__(self, json_path, image_dir, save_json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError('{} is not existed.'.format(json_path))
        if not os.path.exists(image_dir):
            raise FileNotFoundError('{} is not existed.'.format(image_dir))

        self.json_path = json_path
        self.image_dir = image_dir
        self.save_json_path = save_json_path
        json_dir = os.path.dirname(self.save_json_path)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        self.cocoGt = self.load_coco(json_path)
        print('cocoGt is loaded.')

        self.yolov5 = YOLOv5(model_path="../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel",
                             device_id = 0,
                             conf_thresh=0.001,
                             nms_thresh=0.65)
        print('yolov5 is loaded.')

        self.speed_result = [0] * 4
        self.coco91class = self.coco80_to_coco91_class()

    def load_coco(self, json_path):
        annFile = json_path
        cocoGt = COCO(annFile)
        return cocoGt


    def evaluate(self):
        jdict = []
        for i, (k, obj) in enumerate(tqdm(self.cocoGt.imgs.items())):
            file_name = obj['file_name']
            id = obj['id']
            width = obj['width']
            height = obj['height']

            if k != id:
                raise ValueError('key is {}, but id is {}'.format(k, id))

            file_path = os.path.join(self.image_dir, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError('{} is not existed.'.format(file_path))

            dets = self.yolov5.do_once_proc(file_path)

            # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            for det in dets:
                x = int(det[0])
                y = int(det[1])
                w = int(det[2] - det[0])
                h = int(det[3] - det[1])
                box = [x,y,w,h]
                score = float(det[4])
                cls = int(det[5])
                jdict.append(
                    {
                        'image_id': id,
                        'category_id': self.coco91class[cls],
                        'bbox': [round(v, 3) for v in box],
                        'score': round(score, 5),
                    }
                )

        with open(self.save_json_path, 'w') as f:
            json.dump(jdict, f)
        print('{} is saved.',format(self.save_json_path))



    def coco80_to_coco91_class(self):  # converts 80-index (val2014) to 91-index (paper)
        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
             59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return x


    def coco_2017_test(self):

        anno = self.cocoGt
        pred = anno.loadRes(self.save_json_path)
        cocoEval = COCOeval(anno, pred, 'bbox')

        imgIds = sorted(anno.getImgIds())
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize() # cocoEval.stats

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_json_path", "-gt", type=str, default="../data/images/coco/annotations/instances_val2017.json")
    parser.add_argument("--image_dir", "-i", type=str, default="../data/images/coco/images/val2017")
    parser.add_argument("--save_json_path", "-s", type=str, default="./results_json/yolov7_coco_pb_0001_065.json")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    yolotest = YoloTest(
        args.anno_json_path,
        args.image_dir,
        args.save_json_path,

    )

    yolotest.evaluate()

    yolotest.coco_2017_test()

    print('done.')