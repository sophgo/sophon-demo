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

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--image_path', type=str, help='input image path')
    parser.add_argument('--device_id', type=int, help='device id', default=0)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()

    image_path = opt.image_path
    if not os.path.exists(image_path):
        raise FileNotFoundError('{} is not existed!'.format(image_path))
    
    # Test case 1
    image = cv2.imread(image_path, cv2.IMREAD_COLOR, opt.device_id)
    if image is None:
        raise ValueError('imread: image data is null')
    
    ret = cv2.imwrite('out1.jpg', image)
    if not ret:
        raise ValueError('imwrite failed!')

    # Test case 2
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR, opt.device_id)
    if image is None:
        raise ValueError('imdecode: image data is null')
    
    ret, img_encode = cv2.imencode('.jpg', image)
    if not ret:
        raise ValueError('imencode failed!')
    img_encode.tofile('out2.jpg')

    print('all done.')