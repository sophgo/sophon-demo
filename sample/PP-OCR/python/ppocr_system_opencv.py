#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# -*- coding: utf-8 -*- 
import os
import cv2
import numpy as np
import argparse
import sophon.sail as sail
import math
import copy
from PIL import Image, ImageDraw, ImageFont
import logging
import json
import time
logging.basicConfig(level=logging.INFO)

import ppocr_det_opencv as predict_det
import ppocr_rec_opencv as predict_rec
import ppocr_cls_opencv as predict_cls

class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.PPOCRv2Det(args)
        self.text_recognizer = predict_rec.PPOCRv2Rec(args)
        self.use_angle_cls = args.use_angle_cls
        if self.use_angle_cls:
            self.text_classifier = predict_cls.PPOCRv2Cls(args)
        self.rec_thresh = args.rec_thresh
        self.crop_num = 0
        self.crop_time = 0.0
    def __call__(self, img_list, cls=True):
        ori_img_list = img_list.copy()
        results_list = [{"dt_boxes":[], "text":[], "score":[]} for i in range(len(img_list))]
        ori_img_list = img_list.copy()
        dt_boxes_list = self.text_detector(img_list)
        img_dict = {"imgs":[], "dt_boxes":[], "pic_ids":[]}
        for id, dt_boxes in enumerate(dt_boxes_list):
            self.crop_num += len(dt_boxes)
            start_crop = time.time()
            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop = get_rotate_crop_image(ori_img_list[id], tmp_box)
                img_dict["imgs"].append(img_crop)
                img_dict["dt_boxes"].append(dt_boxes[bno])
                img_dict["pic_ids"].append(id)
            self.crop_time += time.time() - start_crop

        if self.use_angle_cls and cls:
            img_dict["imgs"], cls_res = self.text_classifier(img_dict["imgs"])

        rec_res = self.text_recognizer(img_dict["imgs"])

        results_list = [{"dt_boxes":[], "text":[], "score":[]} for i in range(len(img_list))]
        for i, id in enumerate(rec_res.get("ids")):
            text, score = rec_res["res"][i]
            if score >= self.rec_thresh:
                pic_id = img_dict["pic_ids"][id]
                results_list[pic_id]["dt_boxes"].append(img_dict["dt_boxes"][id])
                results_list[pic_id]["text"].append(text)
                results_list[pic_id]["score"].append(score)

        return results_list

def get_rotate_crop_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    
    # align with cpp
    img_crop_width = max(16, img_crop_width)
    img_crop_height = max(16, img_crop_height)
    
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img   

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     rec_thresh=0.5,
                     font_path="../datasets/fonts/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < rec_thresh:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)

def main(opt):
    draw_img_save = "./results/inference_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    ppocrv2_sys = TextSystem(opt)
    
    img_file_list = []
    batch_size = opt.batch_size
    file_list = sorted(os.listdir(opt.input))
    for img_name in file_list:
        img_file = os.path.join(opt.input, img_name)
        img_file_list.append(img_file)
    
    decode_time = 0.0
    result_json = dict()
    for batch_idx in range(0, len(img_file_list), batch_size):
        img_list = []
        # 不是整batch的，转化为1batch进行处理
        if batch_idx + batch_size >= len(img_file_list):
            batch_size = len(img_file_list) - batch_idx
        for idx in range(batch_size):
            start_time = time.time()
            src_img = cv2.imdecode(np.fromfile(img_file_list[batch_idx+idx], dtype=np.uint8), -1)
            decode_time += time.time() - start_time
            img_list.append(src_img)

        results_list = ppocrv2_sys(img_list)    

        for i, result in enumerate(results_list):
            img_name = file_list[batch_idx+i]
            logging.info(img_name)
            logging.info(result["text"])
            image_file = os.path.join(opt.input, img_name)
            image = Image.fromarray(cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB))
            
            img_name_splited = img_name.split('.')[0]
            result_json[img_name_splited] = []
            for j in range(0, len(result["text"])):
                result_json_per_box = dict()
                result_json_per_box["illegibility"] = bool(result["score"][j] < opt.rec_thresh)
                result_json_per_box["points"] = result["dt_boxes"][j].tolist()
                result_json_per_box["score"] = float(result["score"][j])
                result_json_per_box["transcription"] = result["text"][j]
                result_json[img_name_splited].append(result_json_per_box)
            draw_img = draw_ocr_box_txt(
                    image,
                    result["dt_boxes"],
                    result["text"],
                    result["score"],
                    rec_thresh=opt.rec_thresh)
            img_name_pure = os.path.split(image_file)[-1]
            img_path = os.path.join(draw_img_save,
                                    "ocr_res_{}".format(img_name_pure))
            cv2.imwrite(img_path, draw_img[:, :, ::-1])
            logging.info("The visualized image saved in {}".format(img_path))
    save_json = "results/ppocr_system_results_b" + str(opt.batch_size) + ".json"
    with open(save_json, 'w') as jf:
        json.dump(result_json, jf, indent=4, ensure_ascii=False)
    logging.info("result saved in {}".format(save_json))
    
    # calculate speed  
    logging.info("------------------ Det Predict Time Info ----------------------")
    decode_time = decode_time / len(img_file_list)
    preprocess_time = ppocrv2_sys.text_detector.preprocess_time / len(img_file_list)
    inference_time = ppocrv2_sys.text_detector.inference_time / len(img_file_list)
    postprocess_time = ppocrv2_sys.text_detector.postprocess_time / len(img_file_list)
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    if opt.use_angle_cls == True:
        logging.info("------------------ Cls Predict Time Info ----------------------")
        preprocess_time = ppocrv2_sys.text_classifier.preprocess_time / ppocrv2_sys.crop_num
        inference_time = ppocrv2_sys.text_classifier.inference_time / ppocrv2_sys.crop_num
        postprocess_time = ppocrv2_sys.text_classifier.postprocess_time / ppocrv2_sys.crop_num
        logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
        logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
        logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    logging.info("------------------ Rec Predict Time Info ----------------------")
    crop_time = ppocrv2_sys.crop_time / ppocrv2_sys.crop_num
    preprocess_time = ppocrv2_sys.text_recognizer.preprocess_time / ppocrv2_sys.crop_num
    inference_time = ppocrv2_sys.text_recognizer.inference_time / ppocrv2_sys.crop_num
    postprocess_time = ppocrv2_sys.text_recognizer.postprocess_time / ppocrv2_sys.crop_num
    logging.info("crop_time(ms): {:.2f}".format(crop_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

def img_size_type(arg):
    # 将字符串解析为列表类型
    img_sizes = arg.strip('[]').split('],[')
    img_sizes = [size.split(',') for size in img_sizes]
    img_sizes = [[int(width), int(height)] for width, height in img_sizes]
    return img_sizes

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/cali_set_det', help='input image directory path')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu card id')
    parser.add_argument("--batch_size", type=int, default=4, help='img num for a ppocr system process launch.')
    # params for text detector
    parser.add_argument('--bmodel_det', type=str, default='../models/BM1684X/ch_PP-OCRv3_det_fp32.bmodel', help='detector bmodel path')
    parser.add_argument('--det_limit_side_len', type=int, default=[640])
    # params for text recognizer
    parser.add_argument('--bmodel_rec', type=str, default='../models/BM1684X/ch_PP-OCRv3_rec_fp32.bmodel', help='recognizer bmodel path')
    parser.add_argument('--img_size', type=img_size_type, default=[[320, 48],[640, 48]], help='You should set inference size [width,height] manually if using multi-stage bmodel.')
    parser.add_argument("--char_dict_path", type=str, default="../datasets/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=bool, default=True)
    parser.add_argument("--rec_thresh", type=float, default=0.5)
    # params for text classifier
    parser.add_argument("--use_angle_cls", action='store_true')
    parser.add_argument('--bmodel_cls', type=str, default='../models/BM1684X/ch_PP-OCRv3_cls_fp32.bmodel', help='classifier bmodel path')
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
