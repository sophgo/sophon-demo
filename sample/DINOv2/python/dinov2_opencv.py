#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# -*- coding: utf-8 -*-
"""
DINOv2 (dinov2_vitb14_reg) 图像分类推理（OpenCV 前处理 + SAIL）。

模型输出为官方线性探针特征 concat([CLS, mean(patch_tokens)]) = [B, 1536]，
再接官方线性头 dinov2_vitb14_reg4_linear_head (weight[1000,1536], bias[1000])
得到 1000 类 logits。

前处理对齐官方 dinov2/data/transforms.py: make_classification_eval_transform() 的默认参数
（官方 ImageNet-1k linear-probe 精度口径）:
  Resize(短边=256, BICUBIC) -> CenterCrop(224) -> (x/255 - mean) / std
"""
import os
import time
import json
import glob
import argparse
import logging

import cv2
import numpy as np
import sophon.sail as sail

logging.basicConfig(level=logging.INFO)

# 官方 make_classification_eval_transform() 的 resize 短边；CenterCrop 尺寸取自
# bmodel 的输入 H/W（= 224）。
RESIZE_SIZE = 256


class DINOv2(object):
    def __init__(self, args):
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        logging.debug(
            "input_name: {}, input_shape: {}, output_name: {}, output_shape: {}".format(
                self.input_name,
                self.input_shape,
                self.output_name,
                self.net.get_output_shape(self.graph_name, self.output_name),
            )
        )

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 线性头 (numpy)
        head = np.load(args.head)
        self.fc_weight = head["weight"].astype(np.float32)  # [1000, 1536]
        self.fc_bias = head["bias"].astype(np.float32)      # [1000]

        # 类别名
        self.class_names = [l.strip() for l in open(args.classnames) if l.strip()]
        assert len(self.class_names) == self.fc_weight.shape[0], "class names 与线性头类别数不一致"

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]
        # Resize 短边=RESIZE_SIZE 保比例（双三次）；长边 int() 截断，与官方 torchvision Resize 一致
        if w < h:
            nw = RESIZE_SIZE
            nh = int(RESIZE_SIZE * h / w)
        else:
            nh = RESIZE_SIZE
            nw = int(RESIZE_SIZE * w / h)
        img = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
        # Center crop self.net_h x self.net_w = 224x224，偏移 round，与官方 CenterCrop 一致
        left = int(round((nw - self.net_w) / 2.0))
        top = int(round((nh - self.net_h) / 2.0))
        img = img[top:top + self.net_h, left:left + self.net_w]
        # BGR -> RGB, /255, normalize, CHW
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))  # [3, H, W]
        return np.ascontiguousarray(img)

    def predict(self, input_arr):
        input_data = {self.input_name: input_arr}
        outputs = self.net.process(self.graph_name, input_data)
        return list(outputs.values())[0]

    def postprocess(self, feat):
        # feat: [B, 1536] -> logits [B, 1000] -> softmax -> topk
        logits = feat @ self.fc_weight.T + self.fc_bias
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        idx = np.argsort(prob, axis=1)[:, ::-1]
        res = []
        for i in range(prob.shape[0]):
            topk = [(int(j), float(prob[i, j])) for j in idx[i][: self.topk]]
            res.append(topk)
        return res

    def __call__(self, img_list):
        self.topk = min(self.topk, 1000)
        img_input_list = []
        for img in img_list:
            t0 = time.time()
            img_input_list.append(self.preprocess(img))
            self.preprocess_time += time.time() - t0

        img_num = len(img_list)
        if img_num == self.batch_size:
            input_arr = np.stack(img_input_list).astype(np.float32)
        else:
            input_arr = np.zeros(self.input_shape, dtype=np.float32)
            input_arr[:img_num] = np.stack(img_input_list)

        t0 = time.time()
        feat = self.predict(input_arr)[:img_num]
        self.inference_time += time.time() - t0

        t0 = time.time()
        res = self.postprocess(feat)
        self.postprocess_time += time.time() - t0
        return res


def main(args):
    dinov2 = DINOv2(args)
    dinov2.topk = args.topk
    batch_size = dinov2.batch_size

    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(args.input):
        raise Exception('{} is not a directory.'.format(args.input))

    img_list, filename_list, results_list = [], [], []
    decode_time = 0.0
    for filename in sorted(glob.glob(args.input + '/*')):
        if os.path.splitext(filename)[-1] not in ['.jpg', '.png', '.jpeg', '.bmp', '.JPEG', '.JPG', '.BMP']:
            continue
        t0 = time.time()
        src = cv2.imread(filename, cv2.IMREAD_COLOR)
        if src is None:
            logging.error("{} imread is None.".format(filename))
            continue
        decode_time += time.time() - t0
        img_list.append(src)
        filename_list.append(filename)
        if len(img_list) == batch_size:
            results = dinov2(img_list)
            for i, fn in enumerate(filename_list):
                pred = results[i][0][0]
                topk_names = [(dinov2.class_names[k], s) for k, s in results[i]]
                logging.info("{}: {}".format(fn, topk_names))
                results_list.append({
                    'filename': fn,
                    'prediction': pred,
                    'score': results[i][0][1],
                    'topk': [{'class_id': k, 'class_name': dinov2.class_names[k], 'score': s} for k, s in results[i]],
                })
            img_list, filename_list = [], []

    if img_list:
        results = dinov2(img_list)
        for i, fn in enumerate(filename_list):
            pred = results[i][0][0]
            topk_names = [(dinov2.class_names[k], s) for k, s in results[i]]
            logging.info("{}: {}".format(fn, topk_names))
            results_list.append({
                'filename': fn,
                'prediction': pred,
                'score': results[i][0][1],
                'topk': [{'class_id': k, 'class_name': dinov2.class_names[k], 'score': s} for k, s in results[i]],
            })

    if args.input[-1] == '/':
        args.input = args.input[:-1]
    json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_opencv_python_result.json"
    with open(os.path.join(output_dir, json_name), 'w') as jf:
        json.dump(results_list, jf, indent=4, ensure_ascii=False)
    logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

    cn = len(results_list)
    logging.info("------------------ Inference Time Info ----------------------")
    logging.info("decode_time(ms): {:.2f}".format(decode_time / cn * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(dinov2.preprocess_time / cn * 1000))
    logging.info("inference_time(ms): {:.2f}".format(dinov2.inference_time / cn * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(dinov2.postprocess_time / cn * 1000))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/imagenet_val_1k/img', help='path of input, image directory')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/dinov2_vitb14_reg_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--head', type=str, default='../datasets/dinov2_vitb14_reg4_linear_head.npz', help='linear head npz')
    parser.add_argument('--classnames', type=str, default='../datasets/imagenet_classes.txt', help='class names file')
    parser.add_argument('--topk', type=int, default=5, help='top-k classes to output')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    return parser.parse_args()


if __name__ == '__main__':
    args = argsparser()
    main(args)