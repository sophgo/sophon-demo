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
import cv2
import argparse
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
import logging

logging.basicConfig(level=logging.INFO)


class SCRFD:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        if len(self.output_names) not in [9]:
            raise ValueError('only suport 9 outputs, but got {} outputs bmodel'.format(len(self.output_names)))

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        self.keep_ratio = True
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        
        self.max_det = 1000
        
        self.agnostic = False
        self.multi_label = True
        self.postprocess = PostProcess(
                conf_thresh=self.conf_thresh,
                nms_thresh=self.nms_thresh,
                agnostic=self.agnostic,
                multi_label=self.multi_label,
                max_det=self.max_det,
            )
                       
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
           
    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def distance2kps(self, points, distance, max_shape=None):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)
    
    
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
    
    def postprocess_scrfd(self, outs_batch, ratio, pad_list):
        bboxes_batch = []
        scores_batch = []
        for batch_size_idx in range(len(ratio)):
            (ratioh_rec, ratiow_rec) = ratio[batch_size_idx]
            (padw, padh) = pad_list[batch_size_idx]
            scores_list, bboxes_list, kpss_list = [], [], []
            for idx, stride in enumerate(self._feat_stride_fpn):
                scores = outs_batch[idx][batch_size_idx]
                bbox_preds = outs_batch[idx + self.fmc * 1][batch_size_idx] * stride
                kps_preds = outs_batch[idx + self.fmc * 2][batch_size_idx] * stride
                height = self.input_shape[2] // stride
                width = self.input_shape[3] // stride
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                pos_inds = np.where(scores >= self.conf_thresh)[0]
                bboxes = self.distance2bbox(anchor_centers, bbox_preds)
                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]
                scores_list.append(pos_scores)
                bboxes_list.append(pos_bboxes)

                kpss = self.distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
            scores = np.vstack(scores_list)
            bboxes = np.vstack(bboxes_list)
            kpss = np.vstack(kpss_list)
            bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
            ratioh, ratiow = 1 / ratioh_rec, 1 / ratiow_rec
            bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow
            bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
            bboxes[:, 2] = bboxes[:, 2] * ratiow
            bboxes[:, 3] = bboxes[:, 3] * ratioh
            kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
            kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
            bboxes_batch.append(bboxes)
            scores_batch.append(scores)
        return bboxes_batch, scores_batch
     
    
    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        ori_w_list = []
        ori_h_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            ori_w_list.append(ori_w)
            ori_h_list.append(ori_h)
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
        outputs_scrfd, scores_scrfd = self.postprocess_scrfd(outputs, ratio_list, txy_list)
        results_batch = []
        for i,(outputs_scrfd_i,scores_scrfd_i) in enumerate(zip(outputs_scrfd, scores_scrfd)):
            outputs_scrfd_i = np.expand_dims(outputs_scrfd_i, axis=0)
            scores_scrfd_i = np.expand_dims(scores_scrfd_i, axis=0)
            results = self.postprocess(outputs_scrfd_i, scores_scrfd_i, ori_size_list[i], ratio_list[i], txy_list[i])
            results_batch.append(results[0])
        self.postprocess_time += time.time() - start_time
        return results_batch
    
def draw_numpy(image, boxes, masks=None, conf_scores=None):
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        logging.debug("score={}, (x1={},y1={},x2={},y2={})".format(conf_scores[idx], x1, y1, x2, y2))
        if conf_scores[idx] < 0.25:
            continue
        color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if conf_scores is not None:
            cv2.putText(image, str(round(conf_scores[idx], 2)), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
        
    return image
    
    
    
def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    
    # creat save path
    output_dir = "./results"
    results_txt_dir = "./tools/prediction_dir"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(results_txt_dir) and args.eval:
        os.mkdir(results_txt_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)
    output_txt_dir = os.path.join(output_dir, 'txt_results')
    if not os.path.exists(output_txt_dir):
        os.mkdir(output_txt_dir)
    
    # initialize net
    scrfd = SCRFD(args)
    batch_size = scrfd.batch_size
    
    scrfd.init()
    
    decode_time = 0.0
    # test images
    if os.path.isdir(args.input): 
        img_list = []
        filename_list = []
        results_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                img_file = os.path.join(root, filename)
                img_dir = os.path.basename(root)
                if not os.path.exists(os.path.join(results_txt_dir, img_dir)) and args.eval:
                    os.mkdir(os.path.join(results_txt_dir, img_dir))
                elif not os.path.exists(os.path.join(output_txt_dir, img_dir)):
                    os.mkdir(os.path.join(output_txt_dir, img_dir))
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
                    results = scrfd(img_list)
                    
                    for i, filename in enumerate(filename_list):
                        det = results[i]
                        if args.eval:
                            # save result
                            img_name = filename.rsplit('.', 1)[0]
                            txt_name = img_name + '.txt'
                            with open(os.path.join(results_txt_dir, img_dir, txt_name), 'w') as txt_f:
                                txt_f.write(filename + '\n')
                                txt_f.write(str(det.shape[0]) + '\n')
                                for idx in range(det.shape[0]):
                                    x1, y1, x2, y2, score, _ = det[idx]
                                    width = abs(x2 - x1)
                                    height = abs(y2 - y1)
                                    txt_f.write("{:} {:} {:} {:} {:}\n".format(x1, y1, width, height, score))
                        else:
                            # save image
                            res_img = draw_numpy(img_list[i], det[:,:4], masks=None, conf_scores=det[:, -2])
                            cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
                            
                            # save result
                            img_name = filename.rsplit('.', 1)[0]
                            txt_name = img_name + '.txt'
                            with open(os.path.join(output_txt_dir, img_dir, txt_name), 'w') as txt_f:
                                txt_f.write(filename + '\n')
                                txt_f.write(str(det.shape[0]) + '\n')
                                for idx in range(det.shape[0]):
                                    x1, y1, x2, y2, score, _ = det[idx]
                                    width = abs(x2 - x1)
                                    height = abs(y2 - y1)
                                    txt_f.write("{:} {:} {:} {:} {:}\n".format(x1, y1, width, height, score))
                        
                    img_list.clear()
                    filename_list.clear()

        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        if args.eval:
            logging.info("result saved in {} ".format(results_txt_dir))
        else:
            logging.info("result saved in {}".format(os.path.join(output_txt_dir)))
        
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
                results = scrfd(frame_list)
                for i, frame in enumerate(frame_list):
                    det = results[i]
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                    res_frame = draw_numpy(frame_list[i], det[:,:4], masks=None, conf_scores=det[:, -2])
                    out.write(res_frame)
                frame_list.clear()
        cap.release()
        out.release()
        logging.info("result saved in {}".format(save_video))
        
    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = scrfd.preprocess_time / cn
    inference_time = scrfd.inference_time / cn
    postprocess_time = scrfd.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    # average_latency = decode_time + preprocess_time + inference_time + postprocess_time
    # qps = 1 / average_latency
    # logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/scrfd_fp32_1684x.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.02, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.45, help='nms threshold')
    parser.add_argument('--eval', type=bool, default=False, help='if true then gen result_txt')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
    