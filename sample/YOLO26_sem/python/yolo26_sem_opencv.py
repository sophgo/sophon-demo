#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import time
import json
import logging
logging.basicConfig(level=logging.INFO)

import cv2
import argparse
import numpy as np
import sophon.sail as sail

from postprocess_numpy import PostProcess
from utils import blend_seg, is_img


class Yolo26Sem:
    def __init__(self, args):
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.postprocess = PostProcess(net_w=self.net_w, net_h=self.net_h)

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def letterbox(self, im, new_shape=(1024, 2048), color=(114, 114, 114)):
        """等比缩放 + padding，与 ultralytics 一致."""
        shape = im.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = (r, r)
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio

    def preprocess(self, ori_img):
        img, ratio = self.letterbox(ori_img, new_shape=(self.net_h, self.net_w), color=(114, 114, 114))
        img = img.transpose((2, 0, 1))[::-1]        # HWC -> CHW, BGR -> RGB
        img = img.astype(np.float32)
        img = np.ascontiguousarray(img / 255.0)
        return img, ratio

    def predict(self, input_img, img_num):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)
        out_keys = list(outputs.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n == k:
                    ord.append(i)
                    break
        out = [outputs[out_keys[i]][:img_num] for i in ord]
        return out

    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        ratio_list = []
        for ori_img in img_list:
            ori_size_list.append((ori_img.shape[0], ori_img.shape[1]))
            start_time = time.time()
            preprocessed_img, ratio = self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)

        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)

        start_time = time.time()
        outputs = self.predict(input_img, img_num)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        segmaps = self.postprocess(outputs, ori_size_list, ratio_list)
        self.postprocess_time += time.time() - start_time

        return segmaps


def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))

    output_dir = "./results"
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    segmap_dir = os.path.join(output_dir, "segmaps")
    os.makedirs(segmap_dir, exist_ok=True)

    net = Yolo26Sem(args)
    batch_size = net.batch_size
    net.init()
    decode_time = 0.0
    results_list = []
    cn = 0

    if os.path.isdir(args.input):
        img_list = []
        filename_list = []
        files = []
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if is_img(filename):
                    files.append(os.path.join(root, filename))
        files.sort()
        for img_file in files:
            filename = os.path.basename(img_file)
            cn += 1
            logging.info("{}, img_file: {}".format(cn, img_file))
            start_time = time.time()
            src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
            if src_img is None:
                logging.error("{} imdecode is None.".format(img_file))
                continue
            if len(src_img.shape) != 3:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
            decode_time += time.time() - start_time

            img_list.append(src_img)
            filename_list.append(filename)
            if (len(img_list) == batch_size or img_file == files[-1]) and len(img_list):
                segmaps = net(img_list)
                for i, name in enumerate(filename_list):
                    save_basename = os.path.splitext(name)[0]
                    vis = blend_seg(img_list[i], segmaps[i])
                    cv2.imwrite(os.path.join(output_dir, "images", save_basename + ".png"), vis)
                    cv2.imwrite(os.path.join(segmap_dir, save_basename + ".png"), segmaps[i])
                    results_list.append({
                        "image_name": name,
                        "segmap": os.path.join(segmap_dir, save_basename + ".png"),
                    })
                img_list.clear()
                filename_list.clear()
    else:
        cap = cv2.VideoCapture()
        if not cap.open(args.input):
            raise Exception("can not open the video")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        save_video = os.path.join(output_dir, os.path.splitext(os.path.basename(args.input))[0] + '.avi')
        out = cv2.VideoWriter(save_video, fourcc, fps, size)
        frame_list = []
        end_flag = False
        while not end_flag:
            start_time = time.time()
            ret, frame = cap.read()
            decode_time += time.time() - start_time
            if not ret or frame is None:
                end_flag = True
            else:
                frame_list.append(frame)
            if (len(frame_list) == batch_size or end_flag) and len(frame_list):
                segmaps = net(frame_list)
                for i in range(len(frame_list)):
                    cn += 1
                    logging.info("frame {}".format(cn))
                    out.write(blend_seg(frame_list[i], segmaps[i]))
                frame_list.clear()
        cap.release()
        out.release()
        logging.info("result saved in {}".format(save_video))

    if results_list:
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.basename(args.input.rstrip('/')) + "_opencv_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            json.dump({"img_info": results_list}, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

    logging.info("------------------ Predict Time Info ----------------------")
    if cn > 0:
        decode_time = decode_time / cn
        preprocess_time = net.preprocess_time / cn
        inference_time = net.inference_time / cn
        postprocess_time = net.postprocess_time / cn
        logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
        logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
        logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
        logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/yolo26s_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    return parser.parse_args()


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')