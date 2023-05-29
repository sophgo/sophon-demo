#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

import shutil
import numpy as np
import cv2
import argparse
import torch
from p2pnet_utils.preprocess_numpy import PreProcess
from p2pnet_utils.postprocess_numpy import PostProcess
from p2pnet_utils.utils import draw_numpy, is_img

class P2PNet:
    def __init__(self, model_path, batch_size=1):
        if not os.path.exists(model_path):
            raise FileNotFoundError('{} is not existed.'.format(model_path))

        self.net = torch.jit.load(model_path)
        self.net.eval()
        for name, param in self.net.named_parameters():
            print(name,'-->',param.type(),'-->',param.dtype,'-->',param.shape)


        self.batch_size = batch_size
        self.net_c = 3
        self.net_h = 512
        self.net_w = 512
        self.preprocess = PreProcess(self.net_w, self.net_h)
        self.postprocess = PostProcess()

        print('{} is loaded.'.format(model_path))

    @torch.no_grad()
    def predict(self, tensor):
        if tensor.ndim != 4:
            tensor = np.expand_dims(tensor, 0)

        # inp = torch.from_numpy(tensor)
        inp = torch.tensor(tensor, dtype=torch.float)
        # print(inp.size(), inp)
        out = self.net(inp)

        return (out[0].detach().numpy(), out[1].detach().numpy())

def decode_image_opencv(image_path):
    try:
        with open(image_path, "rb") as f:
            image = np.array(bytearray(f.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except:
        image = None
    return image

def main(opt):  
    if not os.path.exists(opt.input):
        raise FileNotFoundError('{} is not existed.'.format(opt.input))
    if not os.path.exists(opt.model):
        raise FileNotFoundError('{} is not existed.'.format(opt.model))
        
    output_dir = os.path.join(os.path.dirname(__file__),"./results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    p2pnet = P2PNet(model_path=opt.model, batch_size=opt.batch_size)
    batch_size = p2pnet.batch_size
    input_path = opt.input

    if os.path.isdir(opt.input) or is_img(opt.input):
        # image directory
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

        img_num = len(input_list)

        inp_batch = []
        images = []
        for ino in range(img_num):
            image = decode_image_opencv(input_list[ino])
            if image is None:
                print('skip: image data is none: {}'.format(input_list[ino]))
                continue
            images.append(image)
            inp_batch.append(input_list[ino])

            if len(images) != batch_size and ino != (img_num - 1):
                continue

            org_size_list = []
            ratio_list = []
            for i in range(len(inp_batch)):
                org_h, org_w = images[i].shape[:2]
                org_size_list.append((org_w, org_h))
                ratio_h, ratio_w = p2pnet.net_h / org_h, p2pnet.net_w / org_w
                ratio_list.append((ratio_w, ratio_h))

            # batch end-to-end inference
            preprocessed_img = p2pnet.preprocess.infer_batch(images)

            out_infer = p2pnet.predict(preprocessed_img)
            points, predict_cnt = p2pnet.postprocess.infer_batch(
                out_infer[0], out_infer[1], ratio_list)
            # print(points, predict_cnt)

            for i, (e_img, p) in enumerate(zip(images, points)):
                vis_image = draw_numpy(e_img, p)
                save_basename = 'res_trace_pt_{}'.format(os.path.basename(inp_batch[i]))
                save_name = os.path.join(output_dir, save_basename.replace('.jpg', ''))
                cv2.imencode('.jpg', vis_image)[1].tofile('{}.jpg'.format(save_name))
                txt_name = os.path.join(output_dir, os.path.basename(inp_batch[i])).replace('.jpg', '.txt')
                with open(txt_name, 'w') as fp:
                    for pt in p:
                        fp.write(str(int(pt[0])) + ' ' + str(int(pt[1])) + '\n')

            images.clear()
            inp_batch.clear()

        print('the results is saved: {}'.format(os.path.abspath(output_dir)))
    else:
        if batch_size != 1:
            raise ValueError(
                'bmodel batch size must be 1 in video inference, but got {}'.format(
                    batch_size)
            )

        cap = cv2.VideoCapture(input_path)
        ret, frame = cap.read()
        id = 0

        while ret and frame is not None:
            org_h, org_w = frame.shape[:2]
            ratio_h, ratio_w = p2pnet.net_h / org_h, p2pnet.net_w / org_w
            ratios_list = [(ratio_w, ratio_h)]
            preprocessed_img = p2pnet.preprocess(frame)
            out_infer = p2pnet.predict(preprocessed_img)
            points, predict_cnt = p2pnet.postprocess.infer_batch(
                out_infer[0], out_infer[1], ratios_list)

            vis_image = draw_numpy(frame.copy(), points[0])
            save_basename = 'res_trace_pt_{}'.format(id)
            save_name = os.path.join(output_dir, save_basename.replace('.jpg', ''))
            cv2.imencode('.jpg', vis_image)[1].tofile('{}.jpg'.format(save_name))
            id += 1
            ret, frame = cap.read()
        cap.release()

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--model', type=str, 
        default='../model_convert/p2pnet_trace.pt', 
        help='pytorch torchsript trace model path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    image_path = os.path.join(os.path.dirname(__file__),"../datasets/video/video.avi")
    parser.add_argument('--input', type=str, default=image_path, help='input image path')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    print('all done.')
