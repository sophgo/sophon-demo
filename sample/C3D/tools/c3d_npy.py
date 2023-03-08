#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import time
import os
import numpy as np
import argparse
import json
import math
import cv2
from logging import raiseExceptions

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def inference(input_path):
    # init hyperparams
    max_video_length = 300
    step = 5
    input_shape = [1,3,16,112,112]

    if os.path.isdir(input_path):
        input_directory = os.listdir(input_path)
    count = 0
    for class_idx in range(0, len(input_directory)):   
        print("class: ", input_directory[class_idx])
        class_path = os.path.join(input_path, input_directory[class_idx])
        video_path_list = os.listdir(class_path)
        for video_idx in range(0, int(len(video_path_list)/2)):
            print("reading: ", os.path.join(class_path, video_path_list[video_idx]))
            cap = cv2.VideoCapture(os.path.join(class_path, video_path_list[video_idx]))
            frame_id = 0 
            input_numpy_array = []
            for i in range(0, max_video_length):
                ret, frame = cap.read()
                if ret == 0 or frame_id >= input_shape[2]:
                    break
                if i % step == 0:
                    frame_id += 1
                    tmp_ = center_crop(cv2.resize(frame, (171, 128)))
                    # tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                    # tmp = tmp_ - np.array([[[102.0, 98.0, 90.0]]])
                    tmp = tmp_ - np.array([[[104.0, 117.0, 123.0]]])
                    # tmp = tmp_ - np.array([[[123.0, 117.0, 104.0]]])
                    input_numpy_array.append(tmp)
            while len(input_numpy_array) < input_shape[2]:
                input_numpy_array.append(input_numpy_array[-1])
            input_numpy_array = np.array(input_numpy_array).astype(np.float32)
            input_numpy_array = np.expand_dims(input_numpy_array, axis=0)
            input_numpy_array = np.transpose(input_numpy_array, (0, 4, 1, 2, 3)) # 1,3,16,112,112
            input_dict = {"input":input_numpy_array}
            np.save("../datasets/cali_set_npy/c3d_cali_"+str(count),input_numpy_array)
            count+=1
    
    print(count)
if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--input_path', default='../datasets/UCF_test_01', required=False)
    if not os.path.exists("../datasets"):
        os.mkdir("../datasets")
    if not os.path.exists("../datasets/cali_set_npy"):
        os.mkdir("../datasets/cali_set_npy")
    ARGS = PARSER.parse_args()
    if not (os.path.isdir(ARGS.input_path)):
        raise Exception('{} is not a valid input.'.format(ARGS.input_path))

    inference(ARGS.input_path)
