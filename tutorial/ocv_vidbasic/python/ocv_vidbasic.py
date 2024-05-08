#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import argparse
import numpy as np
import cv2
import time

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--video_path', type=str, help='input video path')
    parser.add_argument('--device_id', type=int, help='device id', default=0)
    parser.add_argument('--frame_num', type=int, help='encode and decode frame number', default=0)
    parser.add_argument('--width', type=int, help='width of sampler', default=0)
    parser.add_argument('--height', type=int, help='height of sampler', default=0)
    parser.add_argument('--output_name', type=str, help='output name of frame', default='out')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()

    video_path = opt.video_path
    if not os.path.exists(video_path):
        raise FileNotFoundError('{} is not existed!'.format(video_path))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError('video open failed!')
    
    ori_total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ori_fps = cap.get(cv2.CAP_PROP_FPS)
    ori_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ori_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print('orig CAP_PROP_FRAME_HEIGHT: {}'.format(ori_height))
    print('orig CAP_PROP_FRAME_WIDTH: {}'.format(ori_width))
    print('orig total frame number: {}'.format(ori_total_frame_num))
    print('orig FPS: {}'.format(ori_fps))

    frame_num = ori_total_frame_num if opt.frame_num <= 0 else opt.frame_num

    # set resampler
    cap.set(cv2.CAP_PROP_OUTPUT_SRC, 1.0)
    out_src = cap.get(cv2.CAP_PROP_OUTPUT_SRC)
    print('CAP_PROP_OUTPUT_SRC: {}'.format(out_src))

    width = opt.width
    height = opt.height
    if width == 0:
        width = ori_width
    if height == 0:
        height = ori_height
    
    if width > 8192 or height > 8192:
        print('width or height of sampler must be < 8192, but got width: {}, height: {}'.format(width, height))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print('new CAP_PROP_FRAME_HEIGHT: {}'.format(new_width))
    print('new CAP_PROP_FRAME_HEIGHT: {}'.format(new_height))

    tv1 = time.time()

    i_frame_nums = 0
    while(True):
        if i_frame_nums >= frame_num:
            break

        ret, frame = cap.read()
        if (not ret) or (frame is None):
            break
            
        cv2.imencode('.jpg', frame)[1].tofile('{}_{}.jpg'.format(opt.output_name, i_frame_nums))
        
        if (i_frame_nums+1) % 300 ==0:
            tv2 = time.time()
            print('current frame number: {}'.format(i_frame_nums))
            print('current cost time: {}'.format(tv2-tv1))
            print('current process is {} fps!'.format((i_frame_nums / (tv2 - tv1))))

        i_frame_nums += 1

    cap.release()


    print('all done.')