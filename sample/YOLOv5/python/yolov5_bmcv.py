# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
import argparse
import ast
import json
import logging
import os
import time

import cv2
import numpy as np
import sophon.sail as sail

from postprocess_numpy import PostProcess
from utils import COCO_CLASSES, COLORS

logging.basicConfig(level=logging.INFO)
# sail.set_print_flag(1)


class YOLOv5:
    def __init__(self, args):
        # load bmodel
        self.net = sail.nn.Engine(args.bmodel, args.dev_id)
        logging.info("load {} success!".format(args.bmodel))
        self.dev_id = args.dev_id
        self.stream = sail.nn.Stream(args.dev_id)
        self.bmcv = sail.cv.Functional(self.dev_id)

        self.net_name = self.net.get_net_names()[0]
        self.input_name = self.net.get_input_names(self.net_name)[0]
        self.input_shape = self.net.get_input_shapes(self.net_name, 0)[0]
        self.input_scale = self.net.get_input_scales(self.net_name)[0]

        self.output_names = self.net.get_output_names(self.net_name)
        self.output_shapes = self.net.get_output_shapes(self.net_name)
        self.output_dtypes = self.net.get_output_dtypes(self.net_name)
        if len(self.output_names) not in [1, 3]:
            raise ValueError(
                "only suport 1 or 3 outputs, but got {} outputs bmodel".format(
                    len(self.output_names)
                )
            )

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh

        if "use_cpu_opt" in getattr(args, "__dict__", {}):
            self.use_cpu_opt = args.use_cpu_opt
        else:
            self.use_cpu_opt = False
        self.agnostic = False
        self.multi_label = True
        self.max_det = 1000
        self.ab = [x * self.input_scale / 255.0 for x in [1, 0, 1, 0, 1, 0]]
        if self.use_cpu_opt:
            logging.error("Not support cpu_opt now!")
        else:
            self.postprocess = PostProcess(
                conf_thresh=self.conf_thresh,
                nms_thresh=self.nms_thresh,
                agnostic=self.agnostic,
                multi_label=self.multi_label,
                max_det=self.max_det,
            )

        self.output_arrays = [
            sail.nn.Tensor(
                self.output_shapes[i], self.output_dtypes[i], self.dev_id
            )
            for i in range(len(self.output_shapes))
        ]
        self.outputs = {i: array for i, array in enumerate(self.output_arrays)}

    def init(self):
        self.preprocess_time = 0.0
        self.resize_time = 0.0
        self.convert_to_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess_bmcv(self, bmimg):
        start_time = time.time()
        img_w = bmimg.width()
        img_h = bmimg.height()

        resized_img_rgb = sail.cv.Image(
            self.net_h,
            self.net_w,
            sail.cv.ImgFormat.FORMAT_RGB_PLANAR,
            sail.cv.ImgDataType.DATA_TYPE_EXT_1N_BYTE,
            self.dev_id,
        )

        rect = sail.cv.Rect(0, 0, img_w, img_h)

        r_w = self.net_w / img_w
        r_h = self.net_h / img_h
        if r_h > r_w:
            tw = self.net_w
            th = int(r_w * img_h)
            tx1 = tx2 = 0
            ty1 = int((self.net_h - th) / 2)
            ty2 = self.net_h - th - ty1
        else:
            tw = int(r_h * img_w)
            th = self.net_h
            tx1 = int((self.net_w - tw) / 2)
            tx2 = self.net_w - tw - tx1
            ty1 = ty2 = 0

        ratio = (min(r_w, r_h), min(r_w, r_h))
        txy = (tx1, ty1)
        attr = sail.cv.PaddingAttr(tx1, ty1, tw, th, 114, 114, 114, 1)
        self.bmcv.vpp_convert_padding(
            bmimg, resized_img_rgb, rect, attr, sail.cv.ResizeAlgo.BMCV_INTER_NEAREST
        )
        self.resize_time += time.time() - start_time
        start_time = time.time()
        preprocessed_bmimg = sail.cv.Image(
            self.net_h,
            self.net_w,
            sail.cv.ImgFormat.FORMAT_RGB_PLANAR,
            sail.cv.ImgDataType.DATA_TYPE_EXT_FLOAT32,
            self.dev_id,
        )

        attr = sail.cv.ConvertoAttr(
            self.ab[0], self.ab[1], self.ab[2], self.ab[3], self.ab[4], self.ab[5]
        )
        self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, attr)
        self.convert_to_time += time.time() - start_time

        return preprocessed_bmimg, ratio, txy

    def predict(self, input_img, img_num):
        input_data = {0: input_img[0]}

        for i in self.output_arrays:
            if not i.on_device():
                i.to_("device")
        start_time = time.time()
        ret = self.net.process(input_data, self.outputs, self.stream, self.net_name)
        self.inference_time += time.time() - start_time
        if self.use_cpu_opt:
            out = {}
        else:
            # resort
            out_keys = list(self.outputs.keys())
            ord = []
            for i, k in enumerate(out_keys):
                ord.append(k)
            for i in ord:
                self.outputs[out_keys[i]].to_("host")
            out = [self.outputs[out_keys[i]].asnumpy() for i in ord]

        return out

    def __call__(self, bmimg_list):
        img_num = len(bmimg_list)
        ori_size_list = []
        ori_w_list = []
        ori_h_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in bmimg_list:
            ori_h, ori_w = ori_img.height(), ori_img.width()

            ori_size_list.append((ori_w, ori_h))
            ori_w_list.append(ori_w)
            ori_h_list.append(ori_h)
            start_time = time.time()
            preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_bmimg.to_tensor())
            ratio_list.append(ratio)
            txy_list.append(txy)

        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype="float32")
            input_img[:img_num] = np.stack(preprocessed_img_list)

        outputs = self.predict(input_img, img_num)

        start_time = time.time()
        if self.use_cpu_opt:
            self.cpu_opt_process = sail.algo_yolov5_post_cpu_opt(
                self.output_shapes, self.net_w, self.net_h
            )
            results = self.cpu_opt_process.process(
                outputs,
                ori_w_list,
                ori_h_list,
                [self.conf_thresh] * self.batch_size,
                [self.nms_thresh] * self.batch_size,
                True,
                self.multi_label,
            )
            results = [np.array(result) for result in results]
        else:
            results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results


def draw_numpy(image, boxes, masks=None, classes_ids=None, conf_scores=None):
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        logging.debug(
            "class id={}, score={}, (x1={},y1={},x2={},y2={})".format(
                classes_ids[idx], conf_scores[idx], x1, y1, x2, y2
            )
        )
        if conf_scores[idx] < 0.25:
            continue
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) + 1]
        else:
            color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            cv2.putText(
                image,
                COCO_CLASSES[classes_ids[idx] + 1]
                + ":"
                + str(round(conf_scores[idx], 2)),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                thickness=2,
            )
        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5

    return image


def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError("{} is not existed.".format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError("{} is not existed.".format(args.bmodel))

    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, "images")
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)

    # initialize net
    yolov5 = YOLOv5(args)
    batch_size = yolov5.batch_size

    # warm up
    # for i in range(10):
    #     results = yolov5([np.zeros((640, 640, 3))])
    yolov5.init()

    decode_time = 0.0

    # test images
    if os.path.isdir(args.input):
        logging.error("Not support images dir now!")
        return 0
        img_list = []
        filename_list = []
        results_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in [
                    ".jpg",
                    ".png",
                    ".jpeg",
                    ".bmp",
                    ".webp",
                ]:
                    continue
                img_file = os.path.join(root, filename)
                cn += 1
                logging.info("{}, img_file: {}".format(cn, img_file))
                # decode
                start_time = time.time()
                opts: dict = {}
                dec = sail.cv.Decoder(img_file, opts, args.dev_id)
                src_img = sail.cv.Image()
                ret = dec.read(src_img)
                if src_img is None:
                    logging.error("{} imdecode is None.".format(img_file))
                    continue
                # if len(src_img.shape) != 3:
                #     src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                decode_time += time.time() - start_time
                img_list.append(src_img)
                filename_list.append(filename)
                if (len(img_list) == batch_size or cn == len(filenames)) and len(
                    img_list
                ):
                    # predict
                    results = yolov5(img_list)

                    for i, filename in enumerate(filename_list):
                        det = results[i]
                        # save image
                        if args.use_cpu_opt:
                            if det.shape[0] >= 1:
                                res_img = draw_numpy(
                                    img_list[i].asnumpy(),
                                    det[:, :4],
                                    masks=None,
                                    classes_ids=det[:, -2],
                                    conf_scores=det[:, -1],
                                )
                            else:
                                res_img = img_list[i]
                        else:
                            if det.shape[0] >= 1:
                                res_img = draw_numpy(
                                    img_list[i].asnumpy(),
                                    det[:, :4],
                                    masks=None,
                                    classes_ids=det[:, -1],
                                    conf_scores=det[:, -2],
                                )
                            else:
                                res_img = img_list[i].asnumpy()
                        cv2.imwrite(os.path.join(output_img_dir, filename), res_img)

                        # save result
                        res_dict = dict()
                        res_dict["image_name"] = filename
                        res_dict["bboxes"] = []
                        for idx in range(det.shape[0]):
                            bbox_dict = dict()
                            if args.use_cpu_opt:
                                x1, y1, x2, y2, category_id, score = det[idx]
                            else:
                                x1, y1, x2, y2, score, category_id = det[idx]
                            bbox_dict["bbox"] = [
                                float(round(x1, 3)),
                                float(round(y1, 3)),
                                float(round(x2 - x1, 3)),
                                float(round(y2 - y1, 3)),
                            ]
                            bbox_dict["category_id"] = int(category_id)
                            bbox_dict["score"] = float(round(score, 5))
                            res_dict["bboxes"].append(bbox_dict)
                        results_list.append(res_dict)

                    img_list.clear()
                    filename_list.clear()

        # save results
        if args.input[-1] == "/":
            args.input = args.input[:-1]
        json_name = (
            os.path.split(args.bmodel)[-1]
            + "_"
            + os.path.split(args.input)[-1]
            + "_opencv"
            + "_python_result.json"
        )
        with open(os.path.join(output_dir, json_name), "w") as jf:
            # json.dump(results_list, jf)
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

    # test video
    else:
        opts: dict = {sail.cv.DecOpt.QUEUE_CAPACITY: 1000, sail.cv.DecOpt.READ_TIMEOUT: 1000000}
        dec = sail.cv.Decoder(args.input, opts, args.dev_id)
        video_name = os.path.splitext(os.path.split(args.input)[1])[0]
        cn = 0
        frame_list = []
        end_flag = False
        while not end_flag:
            start_time = time.time()
            src_img = sail.cv.Image()
            ret = dec.read(src_img)
            if ret != 0:
                break
            decode_time += time.time() - start_time
            if not ret or src_img is None:
                end_flag = True
            else:
                frame_list.append(src_img)
            if (len(frame_list) == batch_size or end_flag) and len(frame_list):
                results = yolov5(frame_list)
                for i, frame in enumerate(frame_list):
                    det = results[i]
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                
                    save_path = os.path.join(output_img_dir, video_name + '_' + str(cn) + '.jpg')
                    if det.shape[0] <= 0:
                        continue
                    if args.use_cpu_opt:
                        res_frame = draw_numpy(
                            frame_list[i].asnumpy(),
                            det[:, :4],
                            masks=None,
                            classes_ids=det[:, -2],
                            conf_scores=det[:, -1],
                        )
                    else:
                        res_frame = draw_numpy(
                            frame_list[i].asnumpy(),
                            det[:, :4],
                            masks=None,
                            classes_ids=det[:, -1],
                            conf_scores=det[:, -2],
                        )
                    cv2.imwrite(save_path,res_frame)
                frame_list.clear()
        dec.release()
     
        logging.info("result saved in {}".format(output_img_dir))

    # calculate speed
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = yolov5.preprocess_time / cn
    resize_time = yolov5.resize_time / cn
    convert_to_time = yolov5.convert_to_time / cn
    inference_time = yolov5.inference_time / cn
    postprocess_time = yolov5.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("resize_time(ms): {:.2f}".format(resize_time * 1000))
    logging.info("convert_to_time(ms): {:.2f}".format(convert_to_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    # average_latency = decode_time + preprocess_time + inference_time + postprocess_time
    # qps = 1 / average_latency
    # logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument(
        "--input", type=str, default="../datasets/test", help="path of input"
    )
    parser.add_argument(
        "--bmodel",
        type=str,
        default="../models/BM1690/yolov5s_v6.1_3output_fp32_1b.bmodel",
        help="path of bmodel",
    )
    parser.add_argument("--dev_id", type=int, default=0, help="dev id")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.001, help="confidence threshold"
    )
    parser.add_argument("--nms_thresh", type=float, default=0.6, help="nms threshold")
    parser.add_argument(
        "--use_cpu_opt",
        action="store_true",
        default=False,
        help="accelerate cpu postprocess",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    sail.set_loglevel(sail.LogLevel.ERROR)

    args = argsparser()
    main(args)
    print("all done.")
