#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import numpy as np
import cv2

def draw_numpy(image, points):
    # draw the predictions
    size = 2
    for p in points:
        img_to_draw = cv2.circle(image, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image
    # cv2.imwrite(os.path.join('prediction.jpg'.format(predict_cnt)), img_to_draw)
    return img_to_draw

def draw_bmcv(bmcv, bmimg, points):
    # draw the predictions
    for p in points:
        # bmcv.drawPoint(image, (int(p[0]), int(p[1])), (0, 0, 255), 2)
        bmcv.rectangle(bmimg, int(p[0]), int(p[1]), 3, 3, (0, 0, 255), 2)

def is_img(file_name):
    """judge the file is available image or not
    Args:
        file_name (str): input file name
    Returns:
        (bool) : whether the file is available image
    """
    fmt = os.path.splitext(file_name)[-1]
    if isinstance(fmt, str) and fmt.lower() in ['.jpg','.png','.jpeg','.bmp','.jpeg','.webp']:
        return True
    else:
        return False

def decode_image_opencv(image_path):
    try:
        with open(image_path, "rb") as f:
            image = np.array(bytearray(f.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except:
        image = None
    return image

def add_input_img(input_path):
    input_list = []
    if os.path.isdir(input_path):
        for img_name in os.listdir(input_path):
            if is_img(img_name):
                input_list.append(os.path.join(input_path, img_name))
                # image file
    elif is_img(input_path):
        input_list.append(input_path)
    # image list saved in file
    else:
        with open(input_path, 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                line_head = line.strip("\n").split(' ')[0]
                if is_img(line_head):
                    input_list.append(line_head)
    return input_list