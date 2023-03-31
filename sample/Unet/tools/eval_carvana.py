import os
import numpy as np
import cv2
import argparse
from PIL import Image

def main(args):
    predicted_folder_path = args.pred_path
    mask_folder_path = args.label_path

    if not os.path.exists(predicted_folder_path):
        print('Cannot find predicted folder {}'.format(predicted_folder_path))
        return
    if not os.path.exists(mask_folder_path):
        print('Cannot find mask folder {}'.format(mask_folder_path))
        return
    
    img_name_list = []
    precision_list = []
    recall_list = []
    dice_list = []
    eps = 1e-5

    for root, dir, filenames in os.walk(predicted_folder_path):
        for filename in filenames:
            img_name_list.append(filename)

    img_num = len(img_name_list)
    print('Number of images: {}'.format(img_num))
    valid_num = 0

    for filename in img_name_list:
        predicted_name = os.path.join(predicted_folder_path, filename)
        predicted = np.array(Image.open(predicted_name).convert('L'))
        predicted = predicted.reshape(1,1918 * 1280)
        mask_filename = os.path.join(mask_folder_path, filename.split('.')[0] + '_mask.gif')
        if not os.path.exists(mask_filename):
            continue
        valid_num += 1
        print(valid_num)
        mask = np.array(Image.open(mask_filename).convert('L')).reshape(1, 1918*1280)

        predicted = predicted > 200
        mask = mask > 200

        TP = np.sum(predicted & mask)
        FP = np.sum(predicted & (~mask))
        FN = np.sum((~predicted) & mask)
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        dice = 2 * precision * recall / (precision + recall + eps)
        precision_list.append(precision)
        recall_list.append(recall)
        dice_list.append(dice)
    
    print('image num: {}'.format(valid_num))
    print('precision = {}\nrecall = {}\ndice = {}\n'.format(np.mean(precision_list), np.mean(recall_list), np.mean(dice_list)))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--pred_path', type=str, default='../data/images/test', help='path of predicted images')
    parser.add_argument('--label_path', type=str, default='../data/images/label', help='path of label images')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argsparser()
    main(args)