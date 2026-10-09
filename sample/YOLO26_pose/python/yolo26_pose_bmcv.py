#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import cv2
import time
import json
import argparse
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
from utils import skeleton, pose_limb_color, is_img, KPT_OFFSET, KPT_DRAW_THRESH
import logging
logging.basicConfig(level=logging.INFO)

# 画框与保存结果 json 的置信度阈值
DRAW_THRESH = 0.25


class YOLO26_pose:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.info("load {} success!".format(args.bmodel))
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
        suppoort_batch_size = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
        if self.batch_size not in suppoort_batch_size:
            raise ValueError('batch_size must be {} for bmcv, but got {}'.format(suppoort_batch_size, self.batch_size))
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        # init preprocess
        self.use_resize_padding = True
        self.use_vpp = False
        self.ab = [x * self.input_scale / 255. for x in [1, 0, 1, 0, 1, 0]]

        # init postprocess
        self.conf_thresh = args.conf_thresh
        self.postprocess = PostProcess(conf_thres=self.conf_thresh)

        # init time
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
        preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, ((self.ab[0], self.ab[1]),
                                                                   (self.ab[2], self.ab[3]),
                                                                   (self.ab[4], self.ab[5])))
        return preprocessed_bmimg, ratio, txy

    def resize_bmcv(self, bmimg):
        """
        resize for single sail.BMImage
        :param bmimg:
        :return: a resize image of sail.BMImage
        """
        img_w = bmimg.width()
        img_h = bmimg.height()
        if self.use_resize_padding:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            r = min(r_w, r_h)
            tw = int(round(r * img_w))
            th = int(round(r * img_h))
            tx1, ty1 = self.net_w - tw, self.net_h - th  # wh padding

            tx1 /= 2  # divide padding into 2 sides
            ty1 /= 2

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
            resized_img_rgb = preprocess_fn(bmimg, 0, 0, img_w, img_h, self.net_w, self.net_h, attr, sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)
        else:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            ratio = (r_w, r_h)
            txy = (0, 0)
            preprocess_fn = self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
            resized_img_rgb = preprocess_fn(bmimg, self.net_w, self.net_h)
        return resized_img_rgb, ratio, txy

    def predict(self, input_tensor, img_num):
        """
        ensure output order: output0
        Args:
            input_tensor:
        Returns:
        """
        input_tensors = {self.input_name: input_tensor}
        self.net.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)
        outputs_dict = {}
        for name in self.output_names:
            outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num]
        # resort
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
        txy_list = []
        if self.batch_size == 1:
            ori_h, ori_w = bmimg_list[0].height(), bmimg_list[0].width()
            ori_size_list.append((ori_h, ori_w))
            start_time = time.time()
            preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(bmimg_list[0])
            self.preprocess_time += time.time() - start_time
            ratio_list.append(ratio)
            txy_list.append(txy)
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimg, input_tensor)
        else:
            BMImageArray = getattr(sail, 'BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                ori_h, ori_w = bmimg_list[i].height(), bmimg_list[i].width()
                ori_size_list.append((ori_h, ori_w))
                start_time = time.time()
                preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(bmimg_list[i])
                self.preprocess_time += time.time() - start_time
                ratio_list.append(ratio)
                txy_list.append(txy)
                bmimgs[i] = preprocessed_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)

        start_time = time.time()
        outputs = self.predict(input_tensor, img_num)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results


def draw_bmcv(bmcv, image, det_draw, output_img_dir, file_name, cn, isvideo=False):
    img_bgr_planar = bmcv.vpp_convert_format(image, sail.FORMAT_YUV444P)
    boxes = det_draw[:, :4]
    kpts = det_draw[:, KPT_OFFSET:]
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        color = (0, 0, 255)
        # draw boxs
        bmcv.rectangle(img_bgr_planar, x1, y1, (x2 - x1), (y2 - y1), color, thickness=2)

        # draw keypoints
        # for i in range(0, len(kpts[idx]), 3):
        #     x, y, conf = kpts[idx, i:i + 3]
        #     if conf > KPT_DRAW_THRESH:
        #         bmcv.drawPoint(img_bgr_planar, (int(x), int(y)), color, 5)

        # draw skeleton
        for sk_id, sk in enumerate(skeleton):
            r, g, b = pose_limb_color[sk_id]
            pos1 = (int(kpts[idx, (sk[0]-1)*3]), int(kpts[idx, (sk[0]-1)*3+1]))
            pos2 = (int(kpts[idx, (sk[1]-1)*3]), int(kpts[idx, (sk[1]-1)*3+1]))
            conf1 = kpts[idx, (sk[0]-1)*3+2]
            conf2 = kpts[idx, (sk[1]-1)*3+2]
            if conf1 > KPT_DRAW_THRESH and conf2 > KPT_DRAW_THRESH:  # 对于肢体，相连的两个关键点置信度 必须同时大于阈值
                bmcv.polylines(img_bgr_planar, [[pos1, pos2]], True, [int(r), int(g), int(b)], thickness=2)

    if isvideo:
        bmcv.imwrite(os.path.join(output_img_dir, file_name + '_' + str(cn) + '.jpg'), img_bgr_planar)
    else:
        bmcv.imwrite(os.path.join(output_img_dir, file_name), img_bgr_planar)


def save_results(det_list, filename_list, img_list, bmcv, output_img_dir, results_list):
    """draw results and collect the per-person keypoints for the json"""
    for i, filename in enumerate(filename_list):
        det = det_list[i]
        det_draw = det[det[:, 4] > DRAW_THRESH]
        save_name = 'res_bmcv_{}'.format(os.path.basename(filename))
        draw_bmcv(bmcv, img_list[i], det_draw, output_img_dir, save_name, 0)

        kpts = det_draw[:, KPT_OFFSET:]
        for n in range(kpts.shape[0]):
            res_dict = dict()
            res_dict['image_name'] = filename
            res_dict['score'] = float(det_draw[n, 4])
            res_dict['keypoints'] = []
            for m in range(0, len(kpts[n]), 3):
                res_dict['keypoints'].append(float(kpts[n, m]))
                res_dict['keypoints'].append(float(kpts[n, m + 1]))
                res_dict['keypoints'].append(float(kpts[n, m + 2]))
            results_list.append(res_dict)


def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))

    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)

    yolo26_pose = YOLO26_pose(args)
    batch_size = yolo26_pose.batch_size

    yolo26_pose.init()
    decode_time = 0.0
    cn = 0
    # test images
    if os.path.isdir(args.input):
        bmimg_list = []
        filename_list = []
        results_list = []
        for root, dirs, filenames in os.walk(args.input):
            filenames.sort()
            for filename in filenames:
                if not is_img(filename):
                    continue
                img_file = os.path.join(root, filename)
                # decode
                start_time = time.time()
                decoder = sail.Decoder(img_file, True, args.dev_id)
                bmimg = sail.BMImage()
                ret = decoder.read(yolo26_pose.handle, bmimg)
                decode_time += time.time() - start_time
                if ret != 0:
                    logging.error("{} decode failure.".format(img_file))
                    continue

                bmimg_list.append(bmimg)
                filename_list.append(filename)
                if len(bmimg_list) == batch_size:
                    logging.info("{}, img_file: {}".format(cn + len(bmimg_list), img_file))
                    results = yolo26_pose(bmimg_list)
                    save_results(results, filename_list, bmimg_list, yolo26_pose.bmcv, output_img_dir, results_list)
                    cn += len(bmimg_list)
                    bmimg_list.clear()
                    filename_list.clear()

        # the last batch
        if len(bmimg_list):
            logging.info("{}, img_file: {}".format(cn + len(bmimg_list), img_file))
            results = yolo26_pose(bmimg_list)
            save_results(results, filename_list, bmimg_list, yolo26_pose.bmcv, output_img_dir, results_list)
            cn += len(bmimg_list)

        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_bmcv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

    # test video
    else:
        decoder = sail.Decoder(args.input, True, args.dev_id)
        if not decoder.is_opened():
            raise Exception("can not open the video")
        frame_list = []
        end_flag = False
        while not end_flag:
            frame = sail.BMImage()
            start_time = time.time()
            ret = decoder.read(yolo26_pose.handle, frame)
            decode_time += time.time() - start_time
            if ret:  # differ from cv
                end_flag = True
            else:
                frame_list.append(frame)
            if (len(frame_list) == batch_size or end_flag) and len(frame_list):
                results = yolo26_pose(frame_list)
                for i, frame in enumerate(frame_list):
                    det_draw = results[i][results[i][:, 4] > DRAW_THRESH]
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det_draw.shape[0]))
                    draw_bmcv(yolo26_pose.bmcv, frame_list[i], det_draw, output_img_dir, 'video', cn, isvideo=True)
                frame_list.clear()
        decoder.release()
        logging.info("result saved in {}".format(output_img_dir))

    # calculate speed
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = yolo26_pose.preprocess_time / cn
    inference_time = yolo26_pose.inference_time / cn
    postprocess_time = yolo26_pose.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/yolo26s-pose_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='confidence threshold')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
