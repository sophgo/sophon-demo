import argparse
import json
import logging
import numpy as np
import os
import copy
logging.basicConfig(level=logging.DEBUG)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--gt_path', type=str, default='../datasets/coco/coco_style_sampled_val.json', help='path of label json')
    parser.add_argument('--result_json', type=str, default='../python/results/directmhp_fp32_1b.bmodel_test_opencv_python_result.json', help='path of result json')
    parser.add_argument('--ann_type', type=str, default='bbox', help='type of evaluation')
    args = parser.parse_args()
    return args

def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict['image_id'])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict

def sort_images_by_image_id(images_list):
    images_images_dict = {}
    for i, images_dict in enumerate(images_list):
        image_id = str(images_dict['id'])
        images_images_dict[image_id] = images_dict
    return images_images_dict

def calculate_bbox_iou(bboxA, bboxB, format='xyxy'):
    if format == 'xywh':  # xy is in top-left, wh is size
        [Ax, Ay, Aw, Ah] = bboxA[0:4]
        [Ax0, Ay0, Ax1, Ay1] = [Ax, Ay, Ax+Aw, Ay+Ah]
        [Bx, By, Bw, Bh] = bboxB[0:4]
        [Bx0, By0, Bx1, By1] = [Bx, By, Bx+Bw, By+Bh]
    if format == 'xyxy':
        [Ax0, Ay0, Ax1, Ay1] = bboxA[0:4]
        [Bx0, By0, Bx1, By1] = bboxB[0:4]
        
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        return crossArea/(areaA + areaB - crossArea)

def mean_absolute_error_calculate_v2(gt_json_path, pd_json_path, frontal_face):
    matched_iou_threshold = 0.5  # our nms_iou_thre is 0.65
    score_threshold = 0.7
    gt_data, pd_data = [], []  # shapes of both should be N*3
    gt_data_frontal, pd_data_frontal = [], []  # shapes of both should be N*3
    gt_data_backward, pd_data_backward = [], []  # shapes of both should be N*3

    gt_json = json.load(open(gt_json_path, "r"))
    pd_json = json.load(open(pd_json_path, "r"))
    
    gt_labels_list = gt_json['annotations']
    # gt_images_labels_dict = sort_labels_by_image_id(gt_labels_list)
    pd_images_labels_dict = sort_labels_by_image_id(pd_json)
    
    if frontal_face:
        gt_json_frontal_face = copy.deepcopy(gt_json)
        gt_json_frontal_face['images'] = []
        gt_json_frontal_face['annotations'] = []
        pd_json_frontal_face = []
        pd_json_full_face = []
        gt_images_images_dict = sort_images_by_image_id(gt_json['images'])
        appeared_image_id_list = []
        
    
    for gt_label_dict in gt_labels_list:  # matching for each GT label
        image_id = str(gt_label_dict['image_id'])
        gt_bbox = gt_label_dict['bbox']
        [gt_pitch, gt_yaw, gt_roll] = gt_label_dict['euler_angles']
        
        if image_id not in pd_images_labels_dict:  # this image has no bboxes been detected
            continue
            
        pd_results = pd_images_labels_dict[image_id]
        max_iou, matched_index = 0, -1
        for i, pd_result in enumerate(pd_results):  # match predicted bboxes in target image
            score = pd_result['score']
            if score < score_threshold:  # remove head bbox with low confidence
                continue
                
            pd_bbox = pd_result['bbox']
            temp_iou = calculate_bbox_iou(pd_bbox, gt_bbox, format='xywh')
            if temp_iou > max_iou:
                max_iou = temp_iou
                matched_index = i
                
        if max_iou > matched_iou_threshold:
            pd_pitch = pd_results[matched_index]['pitch']
            pd_yaw = pd_results[matched_index]['yaw']
            pd_roll = pd_results[matched_index]['roll']
            gt_data.append([gt_pitch, gt_yaw, gt_roll])
            pd_data.append([pd_pitch, pd_yaw, pd_roll])
 
            pd_results[matched_index]['gt_bbox'] = gt_bbox

            if abs(gt_yaw) > 90:
                gt_data_backward.append([gt_pitch, gt_yaw, gt_roll])
                pd_data_backward.append([pd_pitch, pd_yaw, pd_roll])
            else:
                gt_data_frontal.append([gt_pitch, gt_yaw, gt_roll])
                pd_data_frontal.append([pd_pitch, pd_yaw, pd_roll])
                
            if frontal_face:
                pd_results[matched_index]['gt_pitch'] = gt_pitch
                pd_results[matched_index]['gt_yaw'] = gt_yaw
                pd_results[matched_index]['gt_roll'] = gt_roll
                if abs(gt_yaw) < 90:
                    if str(image_id) not in appeared_image_id_list:
                        appeared_image_id_list.append(str(image_id))
                        gt_json_frontal_face['images'].append(gt_images_images_dict[str(image_id)])
                    gt_json_frontal_face['annotations'].append(gt_label_dict)
                    '''
                    This json file will be used for comparing with FAN, 3DDFA, FSA-Net and WHE-Net in narrow-range.
                    It will also be used for comparing with gt_json_frontal_face to calculate frontal bbox mAP.
                    '''
                    pd_json_frontal_face.append(pd_results[matched_index])  
                '''
                This json file will only be used for comparing with WHE-Net in full-range.
                '''
                pd_json_full_face.append(pd_results[matched_index])
                
         
    total_num = len(gt_labels_list)
    left_num = len(gt_data)
    
    if left_num == 0:
        return total_num, [30,60,30], 2, [30,60,30], 1, [30,60,30], 1
    
    if frontal_face:
        with open(pd_json_path[:-5]+"_pd_frontal.json", 'w') as f:
            json.dump(pd_json_frontal_face, f)
        with open(pd_json_path[:-5]+"_gt_frontal.json", 'w') as f:
            json.dump(gt_json_frontal_face, f)
        with open(pd_json_path[:-5]+"_pd_full.json", 'w') as f:
            json.dump(pd_json_full_face, f)
            
            
    error_list = np.abs(np.array(gt_data) - np.array(pd_data))
    error_list[:, 1] = np.min((error_list[:, 1], 360 - error_list[:, 1]), axis=0)  # yaw range is [-180,180]
    pose_matrix = np.mean(error_list, axis=0)
    
    left_num_b = len(gt_data_backward)
    if left_num_b != 0:
        error_list_backward = np.abs(np.array(gt_data_backward) - np.array(pd_data_backward))
        error_list_backward[:, 1] = np.min((error_list_backward[:, 1], 360 - error_list_backward[:, 1]), axis=0)  # yaw range is [-180,180]
        pose_matrix_b = np.mean(error_list_backward, axis=0)
    else:
        pose_matrix_b, left_num_b = [30,60,30], 1

    left_num_f = len(gt_data_frontal)
    if left_num_f != 0:
        error_list_frontal = np.abs(np.array(gt_data_frontal) - np.array(pd_data_frontal))
        error_list_frontal[:, 1] = np.min((error_list_frontal[:, 1], 360 - error_list_frontal[:, 1]), axis=0)  # yaw range is [-180,180]
        pose_matrix_f = np.mean(error_list_frontal, axis=0)
    else:
        pose_matrix_f, left_num_f = [30,60,30], 1
        
    return total_num, pose_matrix, left_num, pose_matrix_b, left_num_b, pose_matrix_f, left_num_f

def convert_to_coco_bbox(json_file, cocoGt):
    temp_json = []
    images_list = cocoGt.dataset["images"]
    with open(json_file, 'r') as f:
        res_json = json.load(f)
    for res in res_json:
        image_name = res["image_name"]
        bboxes = res["bbox"]
        score = res["score"]
        pitch = res["pitch"]
        yaw = res["yaw"]
        roll = res["roll"]
        if len(bboxes) == 0:
            continue
        for image in images_list:
            if image_name == image["file_name"]:
                image_id = image["id"]
                break
            
        #bboxes = sorted(bboxes, key=lambda x: x['score'], reverse=True)[:400]
        
        data = dict()
        data['image_id'] = int(image_id)
        data['category_id'] = 1
        data['bbox'] = bboxes
        data['score'] = score
        data['pitch'] = pitch
        data['yaw'] = yaw
        data['roll'] =roll
        temp_json.append(data)
            
    with open('temp.json', 'w') as fid:
        json.dump(temp_json, fid)

def main(args):
    annot = args.gt_path
    json_path = args.result_json
    coco = COCO(annot)
    convert_to_coco_bbox(json_path, coco)
    total_num, pose_matrix, left_num, pose_matrix_b, left_num_b, pose_matrix_f, left_num_f = mean_absolute_error_calculate_v2(annot, 'temp.json', True)
    [pitch_error, yaw_error, roll_error] = pose_matrix
    MAE = np.mean(pose_matrix) 
    #print("bbox number: %d / %d; MAE: %s, [pitch_error, yaw_error, roll_error]: %s, %s, %s"%(left_num, total_num, round(MAE, 4), round(pitch_error, 4), round(yaw_error, 4), round(roll_error, 4)))
    #error_list = [MAE, pitch_error, yaw_error, roll_error]
    coco = COCO("temp_gt_frontal.json")
    result = coco.loadRes("temp_pd_frontal.json")
    eval = COCOeval(coco, result, iouType='bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    logging.info("mAP = {}".format(eval.stats[0]))
    print("bbox number: %d / %d; MAE = %s"%(left_num, total_num, round(MAE, 4)))
    #print("frontal_face==true,mAP:{}".format(eval.stats[0]))

    
if __name__ == '__main__':
    args = argsparser()
    main(args)

