#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import argparse
import json
import logging
logging.basicConfig(level=logging.DEBUG)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--label_json', type=str, default='project/OpenPose/data/images/person_keypoints_val2017.json', help='path of label json')
    parser.add_argument('--result_json', type=str, default='project/OpenPose/python/results/pose_coco_fp32_1b.bmodel_val2017_opencv_python_result.json', help='path of result json')
    parser.add_argument('--ann_type', type=str, default='keypoints', help='type of evaluation')
    args = parser.parse_args()
    return args

def convert_to_coco(json_file, cocoGt):
    keypoints_openpose_map = ["nose", "Neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist",  \
                        "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]
    temp_json = []
    images_list = cocoGt.dataset["images"]
    with open(json_file, 'r') as f:
        res_json = json.load(f)
    for res in res_json:
        image_name = res["image_name"]
        keypoints = res["keypoints"]
        if len(keypoints) == 0:
            continue
        for image in images_list:
            if image_name == image["file_name"]:
                image_id = image["id"]
                break
        person_num = int(len(keypoints) / (len(keypoints_openpose_map) * 3))
        # print(len(keypoints), len(keypoints_openpose_map), person_num)
        for i in range(person_num):
            data = dict()
            data['image_id'] = int(image_id)
            data['category_id'] = 1
            data['keypoints'] = []
            score_list = []
            for point_name in cocoGt.dataset["categories"][0]["keypoints"]:
                point_id = keypoints_openpose_map.index(point_name)
                x = keypoints[(i * len(keypoints_openpose_map) + point_id) * 3]
                y = keypoints[(i * len(keypoints_openpose_map) + point_id) * 3 + 1]
                score = keypoints[(i * len(keypoints_openpose_map) + point_id) * 3 + 2]
                data['keypoints'].append(x)
                data['keypoints'].append(y)
                data['keypoints'].append(score)
                score_list.append(score)
            data['score'] = float(sum(score_list)/len(score_list) + 1.25 * max(score_list))
            temp_json.append(data)
    with open('temp.json', 'w') as fid:
        json.dump(temp_json, fid)
    
def main(args):
    cocoGt = COCO(args.label_json)
    convert_to_coco(args.result_json, cocoGt)
    cocoDt = cocoGt.loadRes('temp.json')

    cocoEval = COCOeval(cocoGt, cocoDt, args.ann_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    logging.info("mAP = {}".format(cocoEval.stats[0]))
    
if __name__ == '__main__':
    args = argsparser()
    main(args)