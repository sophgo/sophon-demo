import imp
import cv2
import os
import argparse
import shutil
import numpy as np

def copy_data(ost_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    file_list = os.listdir(ost_path)
    for image_name in file_list:
        ext_name = os.path.splitext(image_name)[-1]
        if ext_name in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            dst_name = os.path.join(dst_path,image_name)
            ost_name = os.path.join(ost_path,image_name)
            if os.path.exists(dst_name):
                print("Remove: {}".format(dst_name))
                os.remove(dst_name)
            shutil.copy(ost_name,dst_name)
            print("Copy {} to {}".format(ost_name,dst_name))

def resize_padding(ost_path, dst_path, resize_w, resize_h):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    file_list = os.listdir(ost_path)
    for image_name in file_list:
        ext_name = os.path.splitext(image_name)[-1]
        if ext_name in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            dst_name = os.path.join(dst_path,"{}_cv_resize_padding{}".format(image_name[:len(image_name)-len(ext_name)],ext_name))
            ost_name = os.path.join(ost_path,image_name)
            if os.path.exists(dst_name):
                print("Remove: {}".format(dst_name))
                os.remove(dst_name)
        cv_img_origin = cv2.imread(ost_name)
        h, w, c = cv_img_origin.shape
        scale_w = float(resize_w) / float(w)
        scale_h = float(resize_h) / float(h)

        temp_resize_w = resize_w
        temp_resize_h = resize_h

        if scale_w < scale_h:
            temp_resize_h = int(h*scale_w)
        else:
            temp_resize_w = int(w*scale_h)
        
        cv_img = cv2.resize(cv_img_origin, (temp_resize_w, temp_resize_h))
        
        padded_img = np.ones((resize_h,resize_w,c), dtype=np.uint8) * 114
        
        padded_img[: temp_resize_h, : temp_resize_w] = cv_img

        cv2.imwrite(dst_name,padded_img)
        print("Write image: {}".format(dst_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for resize")
    parser.add_argument('--ost_path', type=str, default="/workspace/test/YOLOX/datasets/ost_data")
    parser.add_argument('--dst_path', type=str, default="/workspace/test/YOLOX/datasets/ost_data_enhance")
    parser.add_argument('--dst_width',type=int, default=640)
    parser.add_argument('--dst_height',type=int, default=640)

    opt = parser.parse_args()
    #copy_data(opt.ost_path,opt.dst_path)
    resize_padding(opt.ost_path,opt.dst_path,opt.dst_width,opt.dst_height)
