#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import json
import time
import cv2
import argparse
import numpy as np
import sophon.sail as sail
import logging

from utils import *
logging.basicConfig(level=logging.INFO)

class YOLOv8:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
              
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.agnostic = False
        self.multi_label = False
        self.max_det = 300

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, ori_img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        letterbox_img, ratio, (tx1, ty1) = self.letterbox(
            ori_img,
            new_shape=(self.net_h, self.net_w),
            color=(114, 114, 114),
            auto=False,
            scaleFill=False,
            scaleup=True,
            stride=32
        )

        img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.astype(np.float32)
        # input_data = np.expand_dims(input_data, 0)
        img = np.ascontiguousarray(img / 255.0)
        return img, ratio, (tx1, ty1) 

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return im, ratio, (dw, dh)

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
    
    def postprocess(self, preds, ori_size_list, ratio_list, txy_list):
        """
        post-processing
        Args:
            preds: numpy.ndarray -- (n,8400,56) [cx,cy,w,h,conf,17*3]

        Returns: 
            results: list of numpy.ndarray -- (n, 56) [x1, y1, x2, y2, conf, 17*3]

        """
        results = []
        preds = preds[0]
        for i, pred in enumerate(preds):
            # Transpose and squeeze the output to match the expected shape
            pred = np.transpose(pred, (1, 0))   # [8400,56]

            pred = pred[pred[:, 4] > self.conf_thresh]

            if len(pred) == 0:
                print("none detected")
                results.append(np.zeros((0, 56)))
            else:
                # Box (center x, center y, width, height) to (x1, y1, x2, y2)
                pred = self.xywh2xyxy(pred)
                results.append(self.nms_boxes(pred, self.nms_thresh))


        # Rescale boxes and keypoints from img_size to im0 size
        for det, (org_w, org_h), ratio, (tx1, ty1) in zip(results, ori_size_list, ratio_list, txy_list):
            if len(det):
                # Rescale boxes from img_size to im0 size
                coords = det[:, :4]
                coords[:, [0, 2]] -= tx1  # x padding
                coords[:, [1, 3]] -= ty1  # y padding
                coords[:, [0, 2]] /= ratio[0]
                coords[:, [1, 3]] /= ratio[1]

                coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, org_w - 1)  # x1, x2
                coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, org_h - 1)  # y1, y2

                det[:, :4] = coords

                # Rescale keypoints from img_size to im0 size
                num_kpts = (det.shape[1] - 5) // 3
                for kid in range(num_kpts):
                    det[:, 5 + kid * 3] -= tx1
                    det[:, 5 + kid * 3 + 1] -= ty1
                    det[:, 5 + kid * 3] /= ratio[0]
                    det[:, 5 + kid * 3 + 1] /= ratio[1]
                    det[:, 5 + kid * 3] = det[:, 5 + kid * 3].clip(0, org_w - 1)
                    det[:, 5 + kid * 3 + 1] = det[:, 5 + kid * 3 + 1].clip(0, org_h - 1)

        return results

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.copy() if isinstance(x, np.ndarray) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def nms_boxes(self, pred, iou_thres):
        x = pred[:, 0]
        y = pred[:, 1]
        w = pred[:, 2] - pred[:, 0]
        h = pred[:, 3] - pred[:, 1]

        scores = pred[:, 4]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        output = []
        for i in keep:
            output.append(pred[i].tolist())
        return np.array(output)

    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
        
        start_time = time.time()
        outputs = self.predict(input_img, img_num)

        self.inference_time += time.time() - start_time

        start_time = time.time()
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)

        self.postprocess_time += time.time() - start_time

        return results


def draw_numpy(image, det_draw, masks=None, classes_ids=None, conf_scores=None):
    boxes = det_draw[:, :4]
    kpts = det_draw[:, 5:]
    num_kpts = (kpts.shape[1]) // 3
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) + 1]
        else:
            color = (0, 0, 255)
        # draw boxs
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            cv2.putText(image, COCO_CLASSES[classes_ids[idx] + 1] + ':' + str(round(conf_scores[idx], 2)),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
        
        # draw keypoints
        for i in range(0, len(kpts[idx]), 3):
            x, y, conf = kpts[idx, i:i + 3]
            if conf > 0.5:
                cv2.circle(image, (int(x), int(y)), 5, color, -1)
        
        # draw skeleton
        for sk_id, sk in enumerate(skeleton):
            r, g, b = pose_limb_color[sk_id]
            pos1 = (int(kpts[idx, (sk[0]-1)*3]), int(kpts[idx, (sk[0]-1)*3+1]))
            pos2 = (int(kpts[idx, (sk[1]-1)*3]), int(kpts[idx, (sk[1]-1)*3+1]))
            conf1 = kpts[idx, (sk[0]-1)*3+2]
            conf2 = kpts[idx, (sk[1]-1)*3+2]
            if conf1 >0.5 and conf2 >0.5:
                cv2.line(image, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

        
        logging.debug("score={}, (x1={},y1={},x2={},y2={})".format(conf_scores[idx], x1, y1, x2, y2))
    return image

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

    yolov8 = YOLOv8(args)
    batch_size = yolov8.batch_size
    
    yolov8.init()

    decode_time = 0.0
    # test images
    if os.path.isdir(args.input): 
        img_list = []
        filename_list = []
        results_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            filenames.sort()
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
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
                if len(img_list) == batch_size:
                    # predict
                    results = yolov8(img_list)
                    
                    for i, filename in enumerate(filename_list):
                        det = results[i]
                        # save image
                        det_draw = det[det[:, 4] > 0.25]
                        res_img = draw_numpy(img_list[i], det_draw, masks=None, classes_ids=None, conf_scores=det_draw[:, 4])
                        cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
                        
                        kpts = det_draw[:, 5:]

                        for n in range(kpts.shape[0]):
                            res_dict = dict()
                            res_dict['image_name'] = filename
                            res_dict['score'] = det[n, 4]
                            res_dict['keypoints'] = []
                            for m in range(0, len(kpts[n]), 3):
                                x, y, score = kpts[n, m:m + 3]
                                res_dict['keypoints'].append(x)
                                res_dict['keypoints'].append(y)
                                res_dict['keypoints'].append(score)
                            results_list.append(res_dict)
                        
                    img_list.clear()
                    filename_list.clear()

        if len(img_list):
            # predict
            results = yolov8(img_list)
            for i, filename in enumerate(filename_list):
                det = results[i]
                # save image
                det_draw = det[det[:, 4] > 0.25]
                res_img = draw_numpy(img_list[i], det_draw, masks=None, classes_ids=None, conf_scores=det_draw[:, 4])
                cv2.imwrite(os.path.join(output_img_dir, filename), res_img)

                kpts = det_draw[:, 5:]
                # save result
                res_dict = dict()
                res_dict['image_name'] = filename
                # res_dict['bboxes'] = []
                # for idx in range(det.shape[0]):
                #     bbox_dict = dict()
                #     x1, y1, x2, y2, score = det[idx][:5]
                #     bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
                #     bbox_dict['score'] = float(round(score,5))
                #     res_dict['bboxes'].append(bbox_dict)
                res_dict['keypoints'] = []
                for n in range(kpts.shape[0]):
                    for m in range(0, len(kpts[n]), 3):
                        x, y, score = kpts[n, i:i + 3]
                        res_dict['keypoints'].append(x)
                        res_dict['keypoints'].append(y)
                        res_dict['keypoints'].append(score)
                results_list.append(res_dict)
            img_list.clear()
            filename_list.clear()   

        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_opencv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            # json.dump(results_list, jf)
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

    # test video
    else:
        cap = cv2.VideoCapture()
        if not cap.open(args.input):
            raise Exception("can not open the video")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print(fps, size)
        save_video = os.path.join(output_dir, os.path.splitext(os.path.split(args.input)[1])[0] + '.avi')
        out = cv2.VideoWriter(save_video, fourcc, fps, size)
        cn = 0
        frame_list = []
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            decode_time += time.time() - start_time
            if not ret or frame is None:
                break
            frame_list.append(frame)
            if len(frame_list) == batch_size:
                results = yolov8(frame_list)
                for i, frame in enumerate(frame_list):
                    det = results[i]
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                    det_draw = det[det[:, -2] > 0.25]
                    res_frame = draw_numpy(frame_list[i], det_draw, masks=None, classes_ids=None, conf_scores=det_draw[:, -2])
                    out.write(res_frame)
                frame_list.clear()
        if len(frame_list):
            results = yolov8(frame_list)
            for i, frame in enumerate(frame_list):
                det = results[i]
                cn += 1
                logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                det_draw = det[det[:, -2] > 0.25]
                res_frame = draw_numpy(frame_list[i], det_draw, masks=None, classes_ids=None, conf_scores=det_draw[:, -2])
                out.write(res_frame)
        cap.release()
        out.release()
        logging.info("result saved in {}".format(save_video))

    
    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = yolov8.preprocess_time / cn
    inference_time = yolov8.inference_time / cn
    postprocess_time = yolov8.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/yolov8s_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='nms threshold')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
