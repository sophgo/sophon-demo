import time
import os,cv2,sys
import numpy as np
import scipy
import matplotlib as plt
import glob
from scipy import io
import argparse

def get_gt_from_mat(mat_path):
    mat = io.loadmat(mat_path)
    ii = mat['image_info']
    points = []
    for x_cor, y_cor in ii[0][0][0][0][0]:
        x_cor = round(x_cor)
        y_cor = round(y_cor)
        points.append([x_cor, y_cor])
    return points

def getFileNameFromPath(file_path):
    filename = os.path.split(file_path)[-1]
    img_type = filename.split('.')[-1]
    type_len = len(img_type)+1
    filename = filename[0:-type_len]
    return filename

def parse_args():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--gt_path', type=str, default="", help='ground truth path')
    parser.add_argument('--result_path', type=str, default="", help='result path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    gtDir = args.gt_path
    resDir = args.result_path
    print('gtDir:', gtDir)
    print('resDir:', resDir)
    # imageDir = "../ShanghaiTech/part_A/test_data/images"
    # gtDir = "../ShanghaiTech/part_A/test_data/ground-truth"
    # resDir = "../ShanghaiTech/part_A/test_data/result_pc"
    # resDir = "../Release_221201-public/sophon-demo_20221227_085900/sophon-demo_v0.1.2_03507ae_20221227/sample/p2pnet_mq/python/results_trace_pt"
    # resDir = "../Release_221201-public/sophon-demo_20221227_085900/sophon-demo_v0.1.2_03507ae_20221227/sample/p2pnet_mq/python/results_opencv"
    # resDir = "../Release_221201-public/sophon-demo_20221227_085900/sophon-demo_v0.1.2_03507ae_20221227/sample/p2pnet_mq/python/results_opencv_int8"
    # resDir = "../Release_221201-public/sophon-demo_20221227_085900/sophon-demo_v0.1.2_03507ae_20221227/sample/p2pnet_mq/python/results_bmcv"
    # resDir = "../Release_221201-public/sophon-demo_20221227_085900/sophon-demo_v0.1.2_03507ae_20221227/sample/p2pnet_mq/python/results_bmcv_int8"
    gtList = glob.glob(os.path.join(gtDir, '*.mat'))
    if 0 == len(gtList):
        print("gtList is None")
        exit(0)
    i = 0

    gt_num = []
    res_num = []
    for gt_file in gtList:
        i += 1
        print(str(i) + '/' + str(len(gtList)) + '  ' + gt_file)
        gt_name = getFileNameFromPath(gt_file)
        txt_name = gt_name.replace('GT_', '')
        res_txt_file = os.path.join(resDir, txt_name + '.txt')
        print(res_txt_file)
        if not os.path.exists(gt_file) or not os.path.exists(res_txt_file):
            print("gt or res file is not exist")
            continue
        gts = get_gt_from_mat(gt_file)
        res = []
        with open(res_txt_file, 'r') as fp_res:
            res = fp_res.readlines()
        gt_num.append(len(gts))
        res_num.append(len(res))
    print(gt_num)
    print(res_num)
    gt_num = np.array(gt_num)
    res_num = np.array(res_num)
    mae = np.mean(np.abs(gt_num - res_num))
    mse = np.sqrt(np.mean(np.power(gt_num - res_num, 2)))
    print("MAE: {}, MSE: {}.".format(mae, mse))
