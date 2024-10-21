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

def inference(input_path):
    # init hyperparams
    max_video_length = 300
    size = 256
    step = 2
    input_shape = [1,3,32,size,size]

    if os.path.isdir(input_path):
        video_name_list = os.listdir(input_path)
    count = 0
    for video_idx in range(0, 128):
        print("reading: ", os.path.join(input_path, video_name_list[video_idx]))
        cap = cv2.VideoCapture(os.path.join(input_path, video_name_list[video_idx]))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width > height:
            newheight = size
            newwidth = int(math.floor(float(width)/height*size))
        else:
            newwidth = size
            newheight = int(math.floor(float(height)/width*size))
        width_start = int(round((newwidth-size)/2))
        height_start = int(round((newheight-size)/2))
        frame_id = 0 
        input_numpy_array = []
        for i in range(0, max_video_length):
            ret, frame = cap.read()
            if ret == 0 or frame_id >= input_shape[2]:
                break
            if i % step == 0:
                frame_id += 1
                frame2 = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                frame3 = (frame2/255.0 - 0.45)/0.225
                frame4 = cv2.resize(frame3, (newwidth,newheight), interpolation=cv2.INTER_LINEAR)
                frame5 = frame4[height_start:height_start+size,width_start:width_start+size,:]
                input_numpy_array.append(frame5)
        while len(input_numpy_array) < input_shape[2]:
            input_numpy_array.append(input_numpy_array[-1])
        input_numpy_array = np.array(input_numpy_array).astype(np.float32)
        input_numpy_array = np.expand_dims(input_numpy_array, axis=0)
        input_numpy_fast = np.transpose(input_numpy_array, (0, 4, 1, 2, 3)) # 1,3,16,112,112
        input_numpy_slow = input_numpy_fast[:,:,::4,:,:]
        input_dict = {"input_fast":input_numpy_fast,"input_slow":input_numpy_slow}
        np.savez("../datasets/cali_set_npy/slowfast_cali_"+str(count)+".npz",**input_dict)
        count+=1
    
    print(count)
if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--input_path', default='../datasets/sampled_k400', required=False)
    if not os.path.exists("../datasets"):
        os.mkdir("../datasets")
    if not os.path.exists("../datasets/cali_set_npy"):
        os.mkdir("../datasets/cali_set_npy")
    ARGS = PARSER.parse_args()
    if not (os.path.isdir(ARGS.input_path)):
        raise Exception('{} is not a valid input.'.format(ARGS.input_path))

    inference(ARGS.input_path)
