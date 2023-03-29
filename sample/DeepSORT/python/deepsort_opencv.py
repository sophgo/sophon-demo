#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sys
sys.path.append("../../YOLOv5/python")
from yolov5_opencv import YOLOv5
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
import cv2
import argparse
import os
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np

def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 255, 0)
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(str(cls_id), 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image


def update_tracker(target_detector, deepsort, img_batch, frame_id, mot_saver):
    new_faces = []
    detector_results = target_detector(img_batch)
    bbox_xywh_batch = []
    confs_batch = []
    clss_batch = []
    for i in range(len(detector_results)):
        det = detector_results[i]
        bbox_xywh = [] # center x, y
        confs = []
        clss = []
        for idx in range(det.shape[0]):
            x1, y1, x2, y2, conf, cls = det[idx]
            obj = [
                int((x1+x2)/2), int((y1+y2)/2),
                x2-x1, y2-y1
            ]
            bbox_xywh.append(obj)
            confs.append(float(conf))
            clss.append(int(cls))            
        bbox_xywh_batch.append(bbox_xywh)
        confs_batch.append(confs)
        clss_batch.append(clss)
    imgdraw_batch = []
    for i in range(len(img_batch)):
        detect_num = len(bbox_xywh_batch[i])
        outputs = deepsort.update(bbox_xywh_batch[i], confs_batch[i], clss_batch[i], img_batch[i], frame_id + i)
        logging.info("{}, detect_nums: {}; track_nums: {}".format(frame_id + i, detect_num, len(outputs)))
        bboxes2draw = []
        current_ids = []
        for value in list(outputs):
            x1, y1, x2, y2, cls_, track_id = value
            bboxes2draw.append(
                (x1, y1, x2, y2, cls_, track_id)
            )
            current_ids.append(track_id)
            save_str = "{},{},{},{},{},{},1,-1,-1,-1\n".format(frame_id+i, track_id, x1, y1, x2 - x1, y2 - y1)
            mot_saver.write(save_str)
        image = plot_bboxes(img_batch[i], bboxes2draw)
        imgdraw_batch.append(image)
    return imgdraw_batch


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test_car_person_1080P.mp4', help='path of input video or image folder')
    parser.add_argument('--bmodel_detector', type=str, default='../models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel', help='path of detector bmodel')
    parser.add_argument('--bmodel_extractor', type=str, default='../models/BM1684X/extractor_fp16_4b.bmodel', help='path of extractor bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()    
    return args

class yolov5_arg:
    def __init__(self, bmodel, dev_id, conf_thresh, nms_thresh):
        self.bmodel = bmodel
        self.dev_id = dev_id
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

def main():
    args = argsparser()
    cfg = get_config()
    cfg.merge_from_file("configs/deep_sort.yaml")
    yolov5_args = yolov5_arg(args.bmodel_detector, args.dev_id, cfg.DETECTOR.CONF_THRE, cfg.DETECTOR.NMS_THRE)
    #initialize detector yolov5.
    detector = YOLOv5(yolov5_args)
    #initialize deepsort tracker.
    deepsort = DeepSort(args.bmodel_extractor, args.dev_id,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET)
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("./results/images"):
        os.mkdir("./results/images")
    if not os.path.exists("./results/video"):
        os.mkdir("./results/video")
    if not os.path.exists("./results/mot_eval"):
        os.mkdir("./results/mot_eval")
    
    decode_time = 0.0
    encode_time = 0.0
    img_batch = []
    frame_num = 0
    if args.input[-1] == '/':
        args.input = args.input[:-1]
    if os.path.isdir(args.input):
        if args.input.split('/')[-1] == 'img1':
            save_name = args.input.split('/')[-2]
        else:
            save_name = args.input.split('/')[-1]
    else:
        save_name = args.input.split('/')[-1].split('.')[0]
    mot_saver = open("results/mot_eval/{}_{}.txt".format(save_name, args.bmodel_extractor.split("/")[-1]), "w") 
    if os.path.isdir(args.input): #test image folder
        image_paths = []
        for fpathe,dirs,fs in os.walk(args.input):
            for f in fs:
                if f.split(".")[-1] in ['jpg']:
                    image_paths.append(os.path.join(fpathe,f))
        image_paths.sort(key = lambda x: int(x.split(".jpg")[0][-6:]))
        save_id = 0
        #inference flow
        for image_path in image_paths:
            # try:
            start_decode = time.time()
            im = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            decode_time += time.time() - start_decode
            frame_num += 1
            img_batch.append(im)
            if (frame_num % detector.batch_size == 0 or frame_num == len(image_paths))and len(img_batch): 
                result_batch = update_tracker(detector, deepsort, img_batch, frame_num, mot_saver)
                img_batch = []
                start_encode = time.time()
                for result in result_batch:
                    save_id += 1
                    cv2.imwrite("results/images/"+str(save_id)+".jpg", result)
                encode_time += time.time() - start_encode
    else: #test video
        #initialize video capture
        cap = cv2.VideoCapture(args.input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        cap_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        videoWriter = cv2.VideoWriter('results/video/result.mp4', fourcc, fps, cap_size)
        save_id = 0
        #inference flow
        flag = True
        while flag:
            # try:
            start_decode = time.time()
            _, im = cap.read()
            decode_time += time.time() - start_decode
            if im is None:
                flag = False
            else:
                frame_num += 1
                img_batch.append(im)
            if (frame_num % detector.batch_size == 0 or flag == False) and len(img_batch): 
                result_batch = update_tracker(detector, deepsort, img_batch, frame_num, mot_saver)
                img_batch = []
                start_encode = time.time()
                for result in result_batch:
                    cv2.imwrite("results/video/"+str(save_id)+".jpg", result)
                    save_id += 1
                    videoWriter.write(result)
                encode_time += time.time() - start_encode
        cap.release()
        videoWriter.release()
        mot_saver.close()
    # calculate speed  
    decode_time = decode_time / frame_num
    encode_time = encode_time / frame_num
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000)) 
    logging.info("encode_time(ms): {:.2f}".format(encode_time * 1000))
    logging.info("------------------Detector Predict Time Info ----------------------")
    detector_preprocess_time = detector.preprocess_time / frame_num
    detector_inference_time = detector.inference_time / frame_num
    detector_postprocess_time = detector.postprocess_time / frame_num
    logging.info("preprocess_time(ms): {:.2f}".format(detector_preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(detector_inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(detector_postprocess_time * 1000))
    logging.info("-------------------------------------------------------------------")
    logging.info("------------------Deepsort Tracker Time Info ----------------------")
    deepsort_preprocess_time = deepsort.extractor.preprocess_time / deepsort.crop_num
    deepsort_inference_time = deepsort.extractor.inference_time / deepsort.crop_num
    deepsort_postprocess_time = deepsort.postprocess_time / frame_num
    logging.info("preprocess_time(ms): {:.2f}".format(deepsort_preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(deepsort_inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(deepsort_postprocess_time * 1000))
    logging.info("avg crop num: {:.2f}".format(deepsort.crop_num / frame_num))
    logging.info("-------------------------------------------------------------------")

if __name__ == '__main__':
    main()