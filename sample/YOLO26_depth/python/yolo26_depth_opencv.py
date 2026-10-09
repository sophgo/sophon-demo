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
import argparse
import logging
logging.basicConfig(level=logging.INFO)

import cv2
import numpy as np
import sophon.sail as sail

from postprocess_numpy import PostProcess
from utils import is_img, is_video, depth_colorize


class Yolo26Depth:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.debug("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]

        # get input
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}

        # get output
        self.output_names = self.net.get_output_names(self.graph_name)

        # check batch size / input shape
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        # 归一化：alpha = input_scale / 255 由 bmodel input_scale 决定（无烤入 scale 时为 1/255）
        self.alpha = self.input_scale / 255.0

        # postprocess
        self.postprocess = PostProcess(net_w=self.net_w, net_h=self.net_h)

        # time
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess_opencv(self, img_bgr):
        """stretch resize → BGR→RGB → /255 → CHW，与官方 val（LetterBox scale_fill=True + DepthFormat）一致."""
        img = cv2.resize(img_bgr, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1]                      # BGR -> RGB
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        img = (img.astype(np.float32) * self.alpha)
        return img

    def predict(self, input_img):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)
        out = [outputs[n] for n in self.output_names]
        return out

    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = [(img.shape[0], img.shape[1]) for img in img_list]

        start_time = time.time()
        preprocessed = [self.preprocess_opencv(img) for img in img_list]
        self.preprocess_time += time.time() - start_time

        if img_num == self.batch_size:
            input_img = np.stack(preprocessed, axis=0)
        else:
            input_img = np.zeros(self.input_shape, dtype=np.float32)
            input_img[:img_num] = np.stack(preprocessed, axis=0)

        start_time = time.time()
        outputs = self.predict(input_img)
        out = [o[:img_num] for o in outputs]
        self.inference_time += time.time() - start_time

        start_time = time.time()
        depth_maps = self.postprocess(out, ori_size_list)
        self.postprocess_time += time.time() - start_time

        return depth_maps


def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))

    output_dir = "./results"
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    depth_dir = os.path.join(output_dir, "depth_maps")
    os.makedirs(depth_dir, exist_ok=True)

    net = Yolo26Depth(args)
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
            img_bgr = cv2.imread(img_file, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logging.error("{} decode failure.".format(img_file))
                continue
            decode_time += time.time() - start_time
            img_list.append(img_bgr)
            filename_list.append(filename)

            if (len(img_list) == batch_size or img_file == files[-1]) and len(img_list):
                depth_maps = net(img_list)
                for i, name in enumerate(filename_list):
                    save_basename = os.path.splitext(name)[0]
                    depth = depth_maps[i]  # [net_h, net_w] float32 米
                    # 保存 raw 深度图供精度评测（与官方口径一致：768x768）
                    np.save(os.path.join(depth_dir, save_basename + ".npy"), depth)
                    # 保存伪彩色可视化（缩回原图尺寸）
                    vis = depth_colorize(cv2.resize(depth, (img_list[i].shape[1], img_list[i].shape[0]),
                                                    interpolation=cv2.INTER_LINEAR))
                    cv2.imwrite(os.path.join(output_dir, "images", save_basename + ".png"), vis)
                    results_list.append({
                        "image_name": name,
                        "depth_map": os.path.join(depth_dir, save_basename + ".npy"),
                    })
                img_list.clear()
                filename_list.clear()
    elif is_video(args.input):
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
                depth_maps = net(frame_list)
                for i in range(len(frame_list)):
                    cn += 1
                    logging.info("frame {}".format(cn))
                    depth = depth_maps[i]
                    vis = depth_colorize(cv2.resize(depth, (frame_list[i].shape[1], frame_list[i].shape[0]),
                                                    interpolation=cv2.INTER_LINEAR))
                    out.write(vis)
                frame_list.clear()
        cap.release()
        out.release()
        logging.info("result saved in {}".format(save_video))
    else:
        img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise Exception("can not read image: {}".format(args.input))
        depth_maps = net([img_bgr])
        depth = depth_maps[0]
        save_basename = os.path.splitext(os.path.basename(args.input))[0]
        np.save(os.path.join(depth_dir, save_basename + ".npy"), depth)
        vis = depth_colorize(cv2.resize(depth, (img_bgr.shape[1], img_bgr.shape[0]),
                                        interpolation=cv2.INTER_LINEAR))
        cv2.imwrite(os.path.join(output_dir, "images", save_basename + ".png"), vis)
        cn = 1

    # 保存评测用 json
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
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/yolo26s_depth_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    return parser.parse_args()


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')