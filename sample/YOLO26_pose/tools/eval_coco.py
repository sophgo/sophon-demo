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
logging.basicConfig(level=logging.INFO)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 结果 json 中每个人体实例的框置信度低于该值时丢弃（与 YOLOv8_pose 例程保持一致）
SCORE_THRESH = 0.5


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--gt_path', type=str, default='./datasets/coco/person_keypoints_val2017_1000.json', help='path of label json')
    parser.add_argument('--result_json', type=str, default='./results/yolo26s-pose_fp32_1b.bmodel_val2017_1000_bmcv_python_result.json', help='path of result json')
    parser.add_argument('--ann_type', type=str, default='keypoints', help='type of evaluation, keypoints|bbox')
    args = parser.parse_args()
    return args


def convert_to_coco_keypoints(json_file, cocoGt):
    """convert the result json to coco dt json

    The result json contains one entry per person:
        {"image_name": xx.jpg, "score": box score, "keypoints": [x, y, v] * 17}
    Keypoints are in COCO order (nose, left_eye, ..., right_ankle), which is
    exactly what COCOeval expects for the person category.
    """
    temp_json = []
    images_list = cocoGt.dataset["images"]
    with open(json_file, 'r') as f:
        res_json = json.load(f)
    for res in res_json:
        image_name = res["image_name"]
        keypoints = res["keypoints"]
        if res["score"] < SCORE_THRESH:
            continue

        image_id = None
        for image in images_list:
            if image_name == image["file_name"]:
                image_id = image["id"]
                break
        if image_id is None:
            logging.warning("{} is not in the gt json, skip.".format(image_name))
            continue

        score_list = [keypoints[i * 3 + 2] for i in range(int(len(keypoints) / 3))]
        data = dict()
        data['image_id'] = int(image_id)
        data['category_id'] = 1
        data['keypoints'] = keypoints
        data['score'] = sum(score_list) / len(score_list) + res["score"]

        temp_json.append(data)
    with open('temp.json', 'w') as fid:
        json.dump(temp_json, fid)


def convert_to_coco_bbox(json_file, cocoGt):
    """convert the result json to coco dt json for bbox evaluation

    The result json contains one entry per person:
        {"image_name": xx.jpg, "score": box score, "bbox": [x, y, w, h]}
    """
    temp_json = []
    images_list = cocoGt.dataset["images"]
    with open(json_file, 'r') as f:
        res_json = json.load(f)
    for res in res_json:
        image_name = res["image_name"]
        if res["score"] < SCORE_THRESH:
            continue

        image_id = None
        for image in images_list:
            if image_name == image["file_name"]:
                image_id = image["id"]
                break
        if image_id is None:
            logging.warning("{} is not in the gt json, skip.".format(image_name))
            continue

        data = dict()
        data['image_id'] = int(image_id)
        data['category_id'] = 1
        data['bbox'] = res["bbox"]
        data['score'] = res["score"]

        temp_json.append(data)
    with open('temp.json', 'w') as fid:
        json.dump(temp_json, fid)


def main(args):
    cocoGt = COCO(args.gt_path)
    if args.ann_type == 'keypoints':
        convert_to_coco_keypoints(args.result_json, cocoGt)
    elif args.ann_type == 'bbox':
        convert_to_coco_bbox(args.result_json, cocoGt)
    else:
        raise ValueError('ann_type {} is not supported.'.format(args.ann_type))

    cocoDt = cocoGt.loadRes('temp.json')
    cocoEval = COCOeval(cocoGt, cocoDt, args.ann_type)

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    logging.info("mAP = {}".format(cocoEval.stats[0]))


if __name__ == '__main__':
    args = argsparser()
    main(args)
