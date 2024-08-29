#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import json
import argparse
import numpy as np
import cv2

from tqdm import tqdm
from pycocotools.coco import COCO

def create_mask_from_polygons(polygons, image_shape):
    # 创建一个空白的mask
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    for polygon in polygons:
        # polygon 需要转换为 int32 类型的 numpy 数组
        np_polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [np_polygon], 1)
    return mask

def main(args):

    coco = COCO(args.gt_path)

    with open(args.res_path) as f:
        pred_annotations = json.load(f)

    IoUs = []
    for pred_info in tqdm(pred_annotations):
        image_id = pred_info['image_id']
        image_info = coco.loadImgs(image_id)[0]
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        segmentations = []
        for annotation_id in annotation_ids:
            annotation = coco.loadAnns(annotation_id)[0]
            try:
                segmentation = annotation['segmentation'][0]
            except KeyError:
                segmentation = annotation['segmentation']
            segmentations.append(segmentation)

        try:
            mask_ref = create_mask_from_polygons(segmentations, (image_info['height'], image_info['width']))
        except ValueError:
            continue
            
        mask_pred = np.zeros(mask_ref.shape, dtype=np.uint8)
        pred_segs = pred_info['segmentation'] * 2
        for pred_seg in pred_segs:
            mask_pred += create_mask_from_polygons(pred_seg, (image_info['height'], image_info['width']))
        
        mask_ref = (mask_ref > 0.0).astype(np.uint8)
        mask_pred = (mask_pred > 0.5).astype(np.uint8)

        # 计算交集
        intersection = np.logical_and(mask_ref, mask_pred)
        # 计算并集
        union = np.logical_or(mask_ref, mask_pred)
        IoU = 0
        if union.sum() == 0:
            # print("output images from the target detector has no union area with the reference detector !!!")
            continue
        IoU = intersection.sum() / union.sum()
        IoUs.append(IoU)
    
    mIoU = sum(IoUs) / len(IoUs)

    print("mIoU={}".format(round(mIoU, 4)))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--gt_path', type=str, default='../datasets/instances_val2017.json', help='Path of result json')
    parser.add_argument('--res_path', type=str, default='../results/sam2_encoder_f16_1b_2core_COCODataset_opencv_python_result.json', help='Path of label json')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)