#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
from logging import raiseExceptions
from re import I
import cv2
import os
import sys
from ufwio.io import *
import random
import numpy as np
import argparse

def convertString2Bool(str):
    if str.lower() in {'true'}:
        return True
    elif str.lower() in {'false'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def expand_path(args):
    if args.imageset_rootfolder != '':
        args.imageset_rootfolder = os.path.realpath(args.imageset_rootfolder)
    if args.imageset_lmdbfolder != '':
        args.imageset_lmdbfolder = os.path.realpath(args.imageset_lmdbfolder)
    else:
        args.imageset_lmdbfolder = args.imageset_rootfolder

def gen_imagelist(args):
    pic_path = args.imageset_rootfolder
    file_list = []
    dir_or_files = os.listdir(pic_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(pic_path,dir_file)
        ext_name = os.path.splitext(dir_file)[-1]
        if not os.path.isdir(dir_file_path) and ext_name in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            file_list.append(dir_file_path)
    if len(file_list) == 0:
        print(pic_path+' no pictures')
        exit(1)

    if args.shuffle:
        random.seed(3)
        random.shuffle(file_list)

    return file_list

def read_image_to_cvmat(image_path, height, width, is_gray):
    cv_read_flag = cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR
    cv_img_origin = cv2.imread(image_path.strip(), cv_read_flag)

    if is_gray:
        h, w = cv_img_origin.shape
    else:
        h, w, c = cv_img_origin.shape
    print("original shape:", cv_img_origin.shape)
    
    # if height == 0:
    #     height = h
    
    # if width == 0:
    #     width = w
    
    # cv_img = cv2.resize(cv_img_origin, (width, height))
    h_scale = height / h
    w_scale = width / w
    scale = min(h_scale, w_scale)
    resize_img = cv2.resize(cv_img_origin, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    top = 0 
    bottom = height - resize_img.shape[0]
    left = 0 
    right = width - resize_img.shape[1]
    cv_img = cv2.copyMakeBorder(resize_img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(128,128,128))
    
    return cv_img

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='convert imageset')
    parse.add_argument('--imageset_rootfolder', type=str, required=True, help = 'please setting images source path')
    parse.add_argument('--imageset_lmdbfolder', type=str, default='', help = 'please setting lmdb path')
    parse.add_argument('--shuffle', type=convertString2Bool, default=True, help = 'shuffle order of images')
    parse.add_argument('--resize_height', type=int, default=0, help = 'target height')
    parse.add_argument('--resize_width', type=int, default=0, help = 'target width')
    parse.add_argument('--bgr2rgb', type=convertString2Bool, default=True, help = 'convert bgr to rgb')
    parse.add_argument('--gray', type=convertString2Bool, default=False, help='if True, read image as gray')
    args = parse.parse_args()

    expand_path(args)
    image_list = gen_imagelist(args)

    lmdbfile = os.path.join(args.imageset_lmdbfolder,"data.mdb")
    if os.path.exists(lmdbfile):
        print('remove original lmdb file {}'.format(lmdbfile))
        try:
            os.remove(lmdbfile)
        except IsADirectoryError:
            print('{} is dir, remove it manually for safety'.format(lmdbfile))
            sys.exit(1)
        else:
            print('remove original lmdb file {} Ok!'.format(lmdbfile))

    print(" ")

    lmdb = LMDB_Dataset(args.imageset_lmdbfolder)
    for image_path in image_list:
        print('reading image {}'.format(image_path))
        cv_img = read_image_to_cvmat(image_path, args.resize_height, args.resize_width, args.gray)

        if args.bgr2rgb:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        print('cv_imge after resize {}'.format(cv_img.shape))

        if args.gray:
            cv_img = np.expand_dims(cv_img,2)
            print("gray dimension", cv_img.shape)

        #cv_img = (cv_img-mean_value)*scale_value
        cv_img = cv_img.astype('float32')
        cv_img -= 128.
        cv_img /= 255.
        cv_img = np.transpose(cv_img, (2, 0, 1))
        cv_img = np.expand_dims(cv_img,axis=0)

        lmdb.put(np.ascontiguousarray(cv_img, dtype=np.float32))

        print(" ")

    lmdb.close()

