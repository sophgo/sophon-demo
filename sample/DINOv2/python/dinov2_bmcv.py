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
DINOv2 (dinov2_vitb14_reg) 图像分类推理（BMCV 前处理 + SAIL）。

模型输出官方线性探针特征 [B, 1536]，再接官方线性头 (weight[1000,1536], bias[1000])
得 1000 类 logits。

BMCV 前处理对齐官方 dinov2/data/transforms.py: make_classification_eval_transform()
的默认参数（官方 ImageNet-1k linear-probe 精度口径）：

  Resize(短边=256, 与官方同为 BICUBIC, 但 VPP 只支持 LINEAR) ->
  CenterCrop(224) -> (x/255 - mean) / std   (mean/std 见下)

即模型输入 224x224。
"""
import os
import time
import json
import glob
import argparse
import logging

import numpy as np
import sophon.sail as sail

logging.basicConfig(level=logging.INFO)

# 官方 make_classification_eval_transform() 的 resize 短边；CenterCrop 尺寸取自
# bmodel 的输入 H/W（= 224）。
RESIZE_SIZE = 256


class DINOv2(object):
    def __init__(self, args):
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_name)
        # 输出反量化系数：F32/F16 输出为 1.0，INT8 输出需乘上标定 scale
        self.output_scale = self.net.get_output_scale(self.graph_name, self.output_name)
        self.handle = self.net.get_handle()
        self.bmcv = sail.Bmcv(self.handle)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.a = (1.0 / (255.0 * self.std)).tolist()
        self.b = (-self.mean / self.std).tolist()
        self.ab = [(self.a[i], self.b[i]) for i in range(3)]

        head = np.load(args.head)
        self.fc_weight = head["weight"].astype(np.float32)  # [1000, 1536]
        self.fc_bias = head["bias"].astype(np.float32)      # [1000]
        self.class_names = [l.strip() for l in open(args.classnames) if l.strip()]

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

        self.output_tensor = sail.Tensor(self.handle, self.output_shape, self.output_dtype, True, True)
        self.output_tensors = {self.output_name: self.output_tensor}

    def preprocess(self, input_bmimg):
        if input_bmimg.format() == sail.Format.FORMAT_YUV420P:
            bgr = self.bmcv.yuv2bgr(input_bmimg)
        else:
            bgr = input_bmimg
        h, w = bgr.height(), bgr.width()
        # Resize 短边=RESIZE_SIZE 保比例；长边 int() 截断，与官方 torchvision Resize 一致
        if w < h:
            nw = RESIZE_SIZE
            nh = int(RESIZE_SIZE * h / w)
        else:
            nh = RESIZE_SIZE
            nw = int(RESIZE_SIZE * w / h)
        # VPP 不支持 BICUBIC，这里与 C++ bmcv 例程一致采用 LINEAR
        resized = self.bmcv.resize(bgr, nw, nh, sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)
        # Center crop 224x224，偏移 round，与官方 CenterCrop 一致
        left = int(round((nw - self.net_w) / 2.0))
        top = int(round((nh - self.net_h) / 2.0))
        cropped = self.bmcv.crop(resized, left, top, self.net_w, self.net_h)
        # BGR -> RGB planar
        rgb_planar = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, cropped.dtype())
        self.bmcv.convert_format(cropped, rgb_planar)
        # normalize
        out = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(rgb_planar, out, tuple(self.ab))
        return out

    def postprocess(self, feat):
        logits = feat @ self.fc_weight.T + self.fc_bias
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        idx = np.argsort(prob, axis=1)[:, ::-1]
        res = []
        for i in range(prob.shape[0]):
            res.append([(int(j), float(prob[i, j])) for j in idx[i][: self.topk]])
        return res

    def __call__(self, bmimg_list):
        self.topk = min(self.topk, 1000)
        img_num = len(bmimg_list)
        assert img_num <= self.batch_size

        if self.batch_size == 1:
            for bmimg in bmimg_list:
                t0 = time.time()
                out = self.preprocess(bmimg)
                self.preprocess_time += time.time() - t0
                input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
                self.bmcv.bm_image_to_tensor(out, input_tensor)
                t0 = time.time()
                self.net.process(self.graph_name, {self.input_name: input_tensor}, self.output_tensors)
                feat = self.output_tensor.asnumpy() * self.output_scale
                self.inference_time += time.time() - t0
        else:
            out_list = []
            for b in bmimg_list:
                t0 = time.time()
                out_list.append(self.preprocess(b))
                self.preprocess_time += time.time() - t0
            BMImageArray = getattr(sail, 'BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(len(out_list)):
                bmimgs[i] = out_list[i].data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)
            t0 = time.time()
            self.net.process(self.graph_name, {self.input_name: input_tensor}, self.output_tensors)
            feat = self.output_tensor.asnumpy()[:img_num] * self.output_scale
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

    bmimg_list, filename_list, results_list = [], [], []
    decode_time = 0.0
    for filename in sorted(glob.glob(args.input + '/*')):
        if os.path.splitext(filename)[-1] not in ['.jpg', '.png', '.jpeg', '.bmp', '.JPEG', '.JPG', '.BMP']:
            continue
        t0 = time.time()
        decoder = sail.Decoder(filename, True, args.dev_id)
        bmimg = sail.BMImage()
        ret = decoder.read(dinov2.handle, bmimg)
        if ret != 0:
            logging.error("{} decode failure.".format(filename))
            continue
        decode_time += time.time() - t0
        bmimg_list.append(bmimg)
        filename_list.append(filename)
        if len(bmimg_list) == batch_size:
            results = dinov2(bmimg_list)
            for i, fn in enumerate(filename_list):
                logging.info("{}: {}".format(fn, [(dinov2.class_names[k], s) for k, s in results[i]]))
                results_list.append({
                    'filename': fn,
                    'prediction': results[i][0][0],
                    'score': results[i][0][1],
                })
            bmimg_list, filename_list = [], []

    if bmimg_list:
        results = dinov2(bmimg_list)
        for i, fn in enumerate(filename_list):
            logging.info("{}: {}".format(fn, [(dinov2.class_names[k], s) for k, s in results[i]]))
            results_list.append({
                'filename': fn,
                'prediction': results[i][0][0],
                'score': results[i][0][1],
            })

    if args.input[-1] == '/':
        args.input = args.input[:-1]
    json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_bmcv_python_result.json"
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