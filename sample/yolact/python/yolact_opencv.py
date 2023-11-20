# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
import os
import time
import json
import argparse
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
from utils import COLORS, COCO_CLASSES
import logging
import cv2

logging.basicConfig(level=logging.INFO)


class Yolact():
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.debug("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        if len(self.output_names) != 4:
            raise ValueError('only suport 4 outputs, but got {} outputs bmodel'.format(len(self.output_names)))

        # check batch size
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        self.std = np.array([57.38, 57.12, 58.40], dtype=np.float32)

        # image info
        self.width = 550
        self.height = 550

        # init postprocess
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.keep_top_k = args.keep_top_k

        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            keep_top_k=self.keep_top_k,
        )

        # init time
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, img):
        preprocessed_img = cv2.resize(img.astype(np.float32), (self.width, self.height))
        preprocessed_img = (preprocessed_img - self.mean) / self.std
        chw_img = preprocessed_img[:, :, ::-1].transpose((2, 0, 1))

        return chw_img.astype(np.float32)

    def predict(self, input_img, img_num):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)

        # resort
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

        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()
            preprocessed_img = self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_img)

        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)

        start_time = time.time()
        outputs = self.predict(input_img, img_num)
        self.inference_time += time.time() - start_time

        start_time = time.time()

        results = self.postprocess.infer_batch(outputs, ori_size_list)
        self.postprocess_time += time.time() - start_time

        return results

def draw_numpy(image, boxes, masks=None, classes_ids=None, conf_scores=None, save_path = None, videos=False):
    for idx in range(len(boxes)):
        left, top, width, height = boxes[idx, :].astype(np.int32).tolist()

        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) % len(COLORS)]
        else:
            color = (0, 0, 255)

        thickness = 2  # Bounding box line thickness

        cv2.rectangle(image, (left, top), (left + width, top + height), color, thickness=thickness)

        if masks is not None:
            mask = masks[:, :, idx]

            class_id = int(classes_ids[idx]) % len(COLORS)
            color = COLORS[class_id]

            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5

        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)

            text = COCO_CLASSES[classes_ids[idx] + 1] + ':' + str(round(conf_scores[idx], 2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8  # Text size
            text_thickness = 2  # Text line thickness

            # Calculate text position
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
            text_position = (left, top + height - 10)

            # Ensure text does not go beyond image boundary
            if text_position[0] + text_width > image.shape[1]:
                text_position = (left, top - 5)

            cv2.putText(image, text, text_position, font, font_scale, (0, 255, 0), thickness=text_thickness)
            
    save_name = save_path.split('/')[-1]
    cv2.imencode('.jpg', image)[1].tofile('{}.jpg'.format(save_path))

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

        # initialize net
    yolact = Yolact(args)
    batch_size = yolact.batch_size

    # warm up
    # for i in range(10):
    #     results = yolact([np.zeros((550, 550, 3))])
    yolact.init()

    decode_time = 0.0
    # test images
    if os.path.isdir(args.input):
        img_list = []
        filename_list = []
        results_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg', '.png', '.jpeg', '.bmp', '.webp']:
                    continue
                img_file = os.path.join(root, filename)
                cn += 1
                logging.info("{}, img_file: {}".format(cn, img_file))
                # decode
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

                if (len(img_list) == batch_size or cn == len(filenames)) and len(img_list):
                    # predict
                    results = yolact(img_list)

                    for i, filename in enumerate(filename_list):
                        det = [results[j][i] for j in range(4)]
                        if len(det[-1]) == 0:
                            print("No objects in {}".format(filename))
                            continue
  
                        # save image
                        save_path = os.path.join(output_img_dir, filename)
                        
                        # image,bbox,masksclass_id,conf_score,save_path
                        draw_numpy(img_list[i], 
                                   boxes=det[2],
                                   masks=det[3], 
                                   classes_ids=det[0], 
                                   conf_scores=det[1],
                                   save_path=save_path)
                        
                        # save result
                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['bboxes'] = []
                        for idx in range(det[0].shape[0]):
                            bbox_dict = dict()
                            x1, y1, x2, y2 = det[2][idx]
                            score = det[1][idx]
                            category_id = det[0][idx]
                            bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2, 3)),
                                                 float(round(y2, 3))]
                            bbox_dict['category_id'] = int(category_id)
                            bbox_dict['score'] = float(round(score, 5))
                            res_dict['bboxes'].append(bbox_dict)
                        results_list.append(res_dict)
                        ##############################################

                    img_list.clear()
                    filename_list.clear()

        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[
            -1] + "_opencv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

    # test video
    else:
        cap = cv2.VideoCapture()
        if not cap.open(args.input):
            raise Exception("can not open the video")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        save_video = os.path.join(output_dir, os.path.splitext(os.path.split(args.input)[1])[0] + '.avi')
        video_name = os.path.splitext(os.path.split(args.input)[1])[0]
        out = cv2.VideoWriter(save_video, fourcc, fps, size)
        cn = 0
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
                results = yolact(frame_list)
                for i, frame in enumerate(frame_list):
                    det = [results[j][i] for j in range(4)]
                    if len(det[-1]) == 0:
                        print("None object in this frame.")
                        continue
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det[0].shape[0]))
                    save_path = os.path.join(output_img_dir, video_name + '_' + str(cn) + '.jpg')
                    draw_numpy(frame_list[i], det[2], det[3], classes_ids=det[0], conf_scores=det[1],
                                save_path=save_path)
                frame_list.clear()
        cap.release()
        out.release()
        logging.info("result saved in {}".format(save_video))

    # calculate speed
    if cn > 0:
        logging.info("------------------ Predict Time Info ----------------------")
        decode_time = decode_time / cn
        preprocess_time = yolact.preprocess_time / cn
        inference_time = yolact.inference_time / cn
        postprocess_time = yolact.postprocess_time / cn
        logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
        logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
        logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
        logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/yolact_bm1684x_fp32_1b.bmodel',
                        help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.15, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    parser.add_argument('--keep_top_k', type=int, default=100, help='keep top k candidate boxs')
    
    args = parser.parse_args()
    return args
  
if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')