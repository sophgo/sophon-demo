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
from utils import palette_map, blend_seg, is_img


class Yolo26Sem:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]

        # get input
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}

        # get output
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_tensors = {}
        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            output_dtype = self.net.get_output_dtype(self.graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.output_tensors[output_name] = output

        # check batch size
        self.batch_size = self.input_shape[0]
        support_batch_size = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
        if self.batch_size not in support_batch_size:
            raise ValueError('batch_size must be {} for bmcv, but got {}'.format(support_batch_size, self.batch_size))
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        # preprocess
        self.use_resize_padding = True
        self.use_vpp = False
        self.ab = [x * self.input_scale / 255. for x in [1, 0, 1, 0, 1, 0]]

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

    def preprocess_bmcv(self, input_bmimg):
        rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                      sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)
        resized_img_rgb, ratio, txy = self.resize_bmcv(rgb_planar_img)
        preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w,
                                          sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, ((self.ab[0], self.ab[1]),
                                                                   (self.ab[2], self.ab[3]),
                                                                   (self.ab[4], self.ab[5])))
        return preprocessed_bmimg, ratio, txy

    def resize_bmcv(self, bmimg):
        """letterbox resize (pad 114) 到网络输入尺寸."""
        img_w = bmimg.width()
        img_h = bmimg.height()
        if self.use_resize_padding:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            r = min(r_w, r_h)
            tw = int(round(r * img_w))
            th = int(round(r * img_h))
            tx1 = (self.net_w - tw) / 2
            ty1 = (self.net_h - th) / 2
            ratio = (r, r)
            txy = (tx1, ty1)
            attr = sail.PaddingAtrr()
            attr.set_stx(int(round(tx1 - 0.1)))
            attr.set_sty(int(round(ty1 - 0.1)))
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)
            preprocess_fn = self.bmcv.vpp_crop_and_resize_padding if self.use_vpp else self.bmcv.crop_and_resize_padding
            resized_img_rgb = preprocess_fn(bmimg, 0, 0, img_w, img_h, self.net_w, self.net_h,
                                            attr, sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)
        else:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            ratio = (r_w, r_h)
            txy = (0, 0)
            preprocess_fn = self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
            resized_img_rgb = preprocess_fn(bmimg, self.net_w, self.net_h)
        return resized_img_rgb, ratio, txy

    def predict(self, input_tensor, img_num):
        input_tensors = {self.input_name: input_tensor}
        self.net.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)
        outputs_dict = {}
        for name in self.output_names:
            outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num]
        out_keys = list(outputs_dict.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n in k:
                    ord.append(i)
                    break
        out = [outputs_dict[out_keys[i]] for i in ord]
        return out

    def __call__(self, bmimg_list):
        img_num = len(bmimg_list)
        ori_size_list = []
        ratio_list = []
        if self.batch_size == 1:
            ori_size_list.append((bmimg_list[0].height(), bmimg_list[0].width()))
            start_time = time.time()
            preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(bmimg_list[0])
            self.preprocess_time += time.time() - start_time
            ratio_list.append(ratio)
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimg, input_tensor)
        else:
            BMImageArray = getattr(sail, 'BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                ori_size_list.append((bmimg_list[i].height(), bmimg_list[i].width()))
                start_time = time.time()
                preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(bmimg_list[i])
                self.preprocess_time += time.time() - start_time
                ratio_list.append(ratio)
                bmimgs[i] = preprocessed_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)

        start_time = time.time()
        outputs = self.predict(input_tensor, img_num)
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
        bmimg_list = []
        filename_list = []
        handle = sail.Handle(args.dev_id)
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
            decoder = sail.Decoder(img_file, True, args.dev_id)
            bmimg = sail.BMImage()
            ret = decoder.read(handle, bmimg)
            if ret != 0:
                logging.error("{} decode failure.".format(img_file))
                continue
            decode_time += time.time() - start_time
            bmimg_list.append(bmimg)
            filename_list.append(filename)

            if (len(bmimg_list) == batch_size or img_file == files[-1]) and len(bmimg_list):
                segmaps = net(bmimg_list)
                for i, name in enumerate(filename_list):
                    save_basename = os.path.splitext(name)[0]
                    img = bmimg_list[i].asmat()
                    # 保存可视化叠加图
                    vis = blend_seg(img, segmaps[i])
                    cv2.imwrite(os.path.join(output_dir, "images", save_basename + ".png"), vis)
                    # 保存类别图（灰度 png，像素值为类别 id），供精度评测
                    cv2.imwrite(os.path.join(segmap_dir, save_basename + ".png"), segmaps[i])
                    results_list.append({
                        "image_name": name,
                        "segmap": os.path.join(segmap_dir, save_basename + ".png"),
                    })
                bmimg_list.clear()
                filename_list.clear()
    else:
        # 视频：逐帧推理并输出分割叠加视频
        decoder = sail.Decoder(args.input, True, args.dev_id)
        if not decoder.is_opened():
            raise Exception("can not open the video")
        enc_params = "width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25"
        encoder = sail.Encoder("results/output.mp4", args.dev_id, 'h264_bm', 'NV12', enc_params, 10)
        frame_list = []
        handle = sail.Handle(args.dev_id)
        end_flag = False
        while not end_flag:
            frame = sail.BMImage()
            start_time = time.time()
            ret = decoder.read(handle, frame)
            decode_time += time.time() - start_time
            if ret:
                end_flag = True
            else:
                frame_list.append(frame)
            if (len(frame_list) == batch_size or end_flag) and len(frame_list):
                segmaps = net(frame_list)
                for i, frame in enumerate(frame_list):
                    cn += 1
                    logging.info("frame {}".format(cn))
                    img = frame_list[i].asmat()
                    vis = blend_seg(img, segmaps[i])
                    vis_bmimg = net.bmcv.mat_to_bm_image(vis)
                    encoder.video_write(vis_bmimg)
                frame_list.clear()
        decoder.release()
        encoder.release()
        logging.info("result saved in results/output.mp4")

    # 保存评测用 json
    if results_list:
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.basename(args.input.rstrip('/')) + "_bmcv_python_result.json"
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