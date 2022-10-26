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
os.chdir(os.path.abspath(os.path.dirname(sys.argv[0])))
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

import shutil
import numpy as np
import cv2
import argparse
from yolov5_utils.sophon_inference import SophonInference
from yolov5_utils.preprocess_numpy import PreProcess
from yolov5_utils.postprocess_numpy import PostProcess
from yolov5_utils.utils import draw_numpy, is_img

class YOLOv5:
    def __init__(self, model_path, device_id, conf_thresh=0.5, nms_thresh=0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError('{} is not existed.'.format(model_path))

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.net = SophonInference(
            model_path=model_path,
            device_id=device_id,
            input_mode=0, # use cv
        )

        self.batch_size = self.net.inputs_shapes[0][0]
        self.net_c = self.net.inputs_shapes[0][1]
        self.net_h = self.net.inputs_shapes[0][2]
        self.net_w = self.net.inputs_shapes[0][3]
        self.preprocess = PreProcess(self.net_w, self.net_h)

        self.agnostic = False
        self.multi_label = True
        self.max_det = 1000
        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )

        self.resort = False
        self.output_order_node = ['147', '148', '149'] # if use resort, please set output_order_node manually

        print('{} is loaded.'.format(model_path))

    def predict(self, tensor):
        if tensor.ndim != 4:
            tensor = np.expand_dims(tensor, 0)
        # feed: [input0]
        out_dict = self.net.infer_numpy([tensor])
        if not self.resort:
            self.output_order_node = self.net.output_names

        # resort
        out_keys = list(out_dict.keys())
        ord = []
        for n in self.output_order_node:
            for i, k in enumerate(out_keys):
                if n in k:
                    ord.append(i)
                    break
        out = [out_dict[out_keys[i]] for i in ord]
        return out
    
    def do_once_proc(self, file_path):
        
        batch_size = self.net.inputs_shapes[0][0]
        input_path = file_path

        if not os.path.exists(input_path):
            raise FileNotFoundError('{} is not existed.'.format(input_path))
        
        # imgage directory
        input_list = []
        assert is_img(input_path), "not correct img path: {}".format(input_path)
        input_list.append(input_path)
        # imgage list saved in file

        inp_batch = []
        images = []
        result = []
        ino = 0        
        image = decode_image_opencv(input_list[ino])
        if image is None:
            print('skip: image data is none: {}'.format(input_list[ino]))
            return None
        
        images.append(image)
        inp_batch.append(input_list[ino])

        org_size_list = []
        for i in range(len(inp_batch)):
            org_h, org_w = images[i].shape[:2]
            org_size_list.append((org_w, org_h))

        # batch end-to-end inference
        preprocessed_img, ratio_list, txy_list = self.preprocess.infer_batch(images)
        out_infer = self.predict(preprocessed_img)
        det_batch = self.postprocess.infer_batch(
            out_infer,
            org_size_list,
            ratio_list,
            txy_list,
        ) 
        
        return det_batch[0]
        
       


def decode_image_opencv(image_path):
    try:
        with open(image_path, "rb") as f:
            image = np.array(bytearray(f.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except:
        image = None
    return image



def main(opt):
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    else:
        shutil.rmtree(opt.output_dir)
        os.makedirs(opt.output_dir)

    yolov5 = YOLOv5(
        model_path=opt.model,
        device_id=opt.dev_id,
        conf_thresh=opt.conf_thresh,
        nms_thresh=opt.nms_thresh,
    )

    batch_size = yolov5.net.inputs_shapes[0][0]
    input_path = opt.input_path

    if not os.path.exists(input_path):
        raise FileNotFoundError('{} is not existed.'.format(input_path))

    if opt.is_video:
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
            preprocessed_img, ratio, txy = yolov5.preprocess(frame)
            out_infer = yolov5.predict(preprocessed_img)
            det_batch = yolov5.postprocess.infer_batch(
                out_infer,
                [(org_w, org_h)],
                [ratio],
                [txy],
            )

            det = det_batch[0]
            vis_image = draw_numpy(frame.copy(), det[:, :4],
                                   masks=None,
                                   classes_ids=det[:, -1],
                                   conf_scores=det[:, -2])
            save_basename, _ = os.path.splitext(os.path.basename(opt.model))
            save_basename = save_basename + '_opencv_python_result_{}'.format(id)
            save_name = os.path.join(opt.output_dir, save_basename)
            cv2.imencode('.jpg', vis_image)[1].tofile('{}.jpg'.format(save_name))
            id += 1
            ret, frame = cap.read()
        cap.release()

    else:

        # imgage directory
        input_list = []
        if os.path.isdir(input_path):
            for img_name in os.listdir(input_path):
                if is_img(img_name):
                    input_list.append(os.path.join(input_path, img_name))
                    # imgage file
        elif is_img(input_path):
            input_list.append(input_path)
        # imgage list saved in file
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
            for i in range(len(inp_batch)):
                org_h, org_w = images[i].shape[:2]
                org_size_list.append((org_w, org_h))

            # batch end-to-end inference
            preprocessed_img, ratio_list, txy_list = yolov5.preprocess.infer_batch(images)

            cur_bs = len(images)
            padding_bs = batch_size - cur_bs
            # padding a batch
            for i in range(padding_bs):
                preprocessed_img = np.concatenate([preprocessed_img,
                                                   preprocessed_img[:1, :, :].copy()],
                                                  axis=0)

            out_infer = yolov5.predict(preprocessed_img)

            # cancel padding data
            if padding_bs != 0:
                out_infer = [e_data[:cur_bs] for e_data in out_infer]

            det_batch = yolov5.postprocess.infer_batch(
                out_infer,
                org_size_list,
                ratio_list,
                txy_list,
            )

            for i, (e_img, det) in enumerate(zip(images,
                                                 det_batch,
                                                 )):
                vis_image = draw_numpy(e_img, det[:,:4],
                                       masks=None, classes_ids=det[:, -1], conf_scores=det[:, -2])
                save_basename, _ = os.path.splitext(os.path.basename(opt.model)) 
                input_name, _ = os.path.splitext(os.path.basename(inp_batch[i])) 
                save_basename = save_basename + '_opencv_python_result_{}'.format(input_name)
                save_name = os.path.join(opt.output_dir, save_basename)
                cv2.imencode('.jpg', vis_image)[1].tofile('{}.jpg'.format(save_name))

            images.clear()
            inp_batch.clear()

        print('the results is saved: {}'.format(os.path.abspath(opt.output_dir)))


def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--model', type=str, default="../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel", help='bmodel path')
    parser.add_argument('--dev_id', type=int, default=0, help='device id')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    parser.add_argument('--is_video', default=0, type=int, help="input is video?")
    parser.add_argument('--input_path', type=str, default="../data/images/zidane.jpg", help='input image path')
    parser.add_argument('--output_dir', type=str, default="results", help='output image directory')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    print('all done.')