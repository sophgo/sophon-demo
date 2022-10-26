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

os.chdir(os.path.abspath(os.path.dirname(sys.argv[0])))
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)


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

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img

def preprocess(img, image_size):
        h0, w0 = img.shape[:2]  # orig hw
        r = image_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        
        padded_img_bgr = letterbox(img=img, new_shape=(image_size, image_size), auto=False)
        # cv2.imwrite("/workspace/examples/YOLOv7_object/data/padded_img_bgr_center.jpg", padded_img_bgr)
        
        # BGR => RGB
        padded_img_rgb = cv2.cvtColor(padded_img_bgr, cv2.COLOR_BGR2RGB)

        # to float32
        padded_img_rgb_data = padded_img_rgb.astype(np.float32)

        # Normalize to [0,1]
        padded_img_rgb_data /= 255.0

        # HWC to CHW format:
        padded_img_rgb_data = np.transpose(padded_img_rgb_data, [2, 0, 1])
        
        # CHW to NCHW format
        padded_img_rgb_data = np.expand_dims(padded_img_rgb_data, axis=0)
        # Convert the image to row-major order, also known as "C order":
        padded_img_rgb_data = np.ascontiguousarray(padded_img_rgb_data)        
        return padded_img_rgb_data
    
def preprocess_center(img, image_size):
    
    # 需要的输入尺寸
    h = image_size
    w = image_size    
    print("need img size w={} h={}".format(w, h))

    img_h, img_w, img_c = img.shape
    

    # Calculate widht and height and paddings
    r_w = w / img_w
    r_h = h / img_h
    if r_h > r_w:
        tw = w
        th = int(r_w * img_h)
        tx1 = tx2 = 0
        ty1 = int((h - th) / 2)
        ty2 = h - th - ty1
    else:
        tw = int(r_h * img_w)
        th = h
        tx1 = int((w - tw) / 2)
        tx2 = w - tw - tx1
        ty1 = ty2 = 0

    print("tw={} th={} tx1={} tx2={} ty1={} ty2={} r_w={} r_h={}".format(tw, th, tx1, tx2, ty1, ty2, r_w, r_h))

    # Resize long
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    
    # pad
    padded_img_bgr = cv2.copyMakeBorder(img, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # cv2.imwrite("padded_img_bgr_center.jpg", padded_img_bgr)
    # exit(1)
    
    # BGR => RGB
    padded_img_rgb = cv2.cvtColor(padded_img_bgr, cv2.COLOR_BGR2RGB)

    # to float32
    padded_img_rgb_data = padded_img_rgb.astype(np.float32)

    # Normalize to [0,1]
    padded_img_rgb_data /= 255.0

    # HWC to CHW format:
    padded_img_rgb_data = np.transpose(padded_img_rgb_data, [2, 0, 1])
    
    # CHW to NCHW format
    padded_img_rgb_data = np.expand_dims(padded_img_rgb_data, axis=0)
    # Convert the image to row-major order, also known as "C order":
    padded_img_rgb_data = np.ascontiguousarray(padded_img_rgb_data)        
    return padded_img_rgb_data

def read_image(image_path, image_size):
    cv_img_origin = cv2.imread(image_path.strip())
    print("original shape:", cv_img_origin.shape)    

    # 前处理
    # cv_img = preprocess(cv_img_origin, image_size)
    cv_img = preprocess_center(cv_img_origin, image_size)
    
    print("after preprocess shape:", cv_img.shape)
    
    return cv_img

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='convert imageset')
    parse.add_argument('--imageset_rootfolder', type=str, default='../data/images/coco200', help = 'please setting images source path')
    parse.add_argument('--imageset_lmdbfolder', type=str, default='../data/images', help = 'please setting lmdb path')
    parse.add_argument('--shuffle', type=convertString2Bool, default=False, help = 'shuffle order of images')
    parse.add_argument('--image_size', type=int, default=640, help = 'target size')
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

    flag = True

    lmdb = LMDB_Dataset(args.imageset_lmdbfolder)
    for image_path in image_list:
        print('reading image {}'.format(image_path))
        cv_img = read_image(image_path, args.image_size)        
        lmdb.put(cv_img)
    lmdb.close()

