# -*- coding: utf-8 -*- 
import os
import cv2
import numpy as np
import argparse
import sophon.sail as sail
import math
import logging
logging.basicConfig(level=logging.DEBUG)

from shapely.geometry import Polygon
import pyclipper

class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                   src_w, src_h)

            boxes_batch.append({'points': boxes})
        return boxes_batch

class PPOCRv2Det(object):
    def __init__(self, args):
        self.det_batch_size = args.det_batch_size
        # load bmodel
        model_path = args.det_model
        logging.info("using model {}".format(model_path))
        self.net = sail.Engine(model_path, args.tpu_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        logging.info("load bmodel success!")
        #self.input_shape = self.net.get_max_input_shapes(self.graph_name)[self.input_name]
        # preprocess
        self.det_limit_side_len = sorted(args.det_limit_side_len)
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)).astype('float32') * 255.0
        self.scale = np.array([1/0.229, 1/0.224, 1/0.225]).reshape((1, 1, 3)).astype('float32') * 1 / 255.0

        # postprocess
        self.postprocess_op = DBPostProcess(thresh=0.3,
                                            box_thresh=0.6,
                                            max_candidates=1000,
                                            unclip_ratio=1.5,
                                            use_dilation=False,
                                            score_mode="fast")

    def preprocess(self, img):
        h, w, _ = img.shape
        size_max = max(h, w)
        if size_max >= self.det_limit_side_len[-1]:
            limit_side_len = self.det_limit_side_len[-1]
            ratio = float(limit_side_len) / size_max
        else:
            for side_len in self.det_limit_side_len:
                if size_max <= side_len:
                    limit_side_len = side_len
                    ratio = 1.
                    break
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
            
        if h != resize_h or w != resize_w:
            img = cv2.resize(img, (resize_w, resize_h))

        img = img.astype('float32')
        img = img - self.mean
        img = img * self.scale
        img = np.transpose(img, (2, 0, 1))

        padding_im = np.zeros((3, limit_side_len, limit_side_len), dtype=np.float32)
        padding_im[:, 0 : resize_h, 0 : resize_w] = img
        
        # CHW to NCHW format
        #padding_im = np.expand_dims(padding_im, axis=0)

        #ratio_h = resize_h / float(h)
        #ratio_w = resize_w / float(w)

        return padding_im, [h, w, resize_h, resize_w]

    def predict(self, tensor):
        input_data = {self.input_name: np.array(tensor, dtype=np.float32)}
        outputs = self.net.process(self.graph_name, input_data)
        return list(outputs.values())[0]

    def postprocess(self, outputs, src_h, src_w, resize_h, resize_w):
        preds = {}
        preds['maps'] = np.expand_dims(outputs[:, 0 : resize_h, 0 : resize_w], axis=0)
        shape_list = np.array([src_h, src_w, resize_h / float(src_h), resize_w / float(src_w)])
        shape_list = np.expand_dims(shape_list, axis=0)
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, (src_h, src_w, 3))

        return dt_boxes

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes
    
    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def __call__(self, img_list):
        img_num = len(img_list)
        img_input_list = []
        img_size_list = []
        # 对每张图片进行预处理
        for img in img_list:
            img, [src_h, src_w, resize_h, resize_w] = self.preprocess(img)
            img_input_list.append(img)
            img_size_list.append([src_h, src_w, resize_h, resize_w])
        # 组batch操作
        outputs_list = []
        for beg_img_no in range(0, img_num, self.det_batch_size):
            end_img_no = min(img_num, beg_img_no + self.det_batch_size)
            if beg_img_no + self.det_batch_size > img_num:
                for ino in range(beg_img_no, end_img_no):
                    img_input = np.expand_dims(img_input_list[ino], axis=0)
                    outputs = self.predict(img_input)
                    outputs_list.extend(outputs)
            else:
                img_input = np.stack(img_input_list[beg_img_no:end_img_no])
                outputs = self.predict(img_input)
                outputs_list.extend(outputs)
        # 对输出进行后处理
        dt_boxes_list = []
        for id, outputs in enumerate(outputs_list):
            src_h, src_w, resize_h, resize_w = img_size_list[id]
            dt_boxes = self.postprocess(outputs, src_h, src_w, resize_h, resize_w)
            dt_boxes_list.append(dt_boxes)
        return dt_boxes_list

def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im

def main(opt):
    draw_img_save = "./results/det_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    ppocrv2_det = PPOCRv2Det(opt)
    # 读取得到的图片存放在这个list中
    img_list = []
    for img_name in os.listdir(opt.img_path):
        logging.info(img_name)
        #label = img_name.split('.')[0]
        img_file = os.path.join(opt.img_path, img_name)
        #print(img_file, label)
        # src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
        src_img = cv2.imread(img_file)
        img_list.append(src_img)
    # 检测得到的结果
    dt_boxes_list = ppocrv2_det(img_list)

    for img_name, dt_boxes in zip(os.listdir(opt.img_path), dt_boxes_list):
        #logging.info(img_name)
        #logging.info(dt_boxes)
        image_file = os.path.join(opt.img_path, img_name)
        draw_im = draw_text_det_res(dt_boxes, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save,
                                "det_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, draw_im)
        logging.info("The visualized image saved in {}".format(img_path))

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    #parser.add_argument('--img-size', type=int, default=[48, 192], help='inference size (pixels)')
    parser.add_argument('--tpu_id', type=int, default=0, help='tpu id')
    parser.add_argument('--img_path', type=str, default='../data/images/ppocr_img/test', help='input image path')
    parser.add_argument('--det_model', type=str, default='../data/models/BM1684/ch_PP-OCRv2_det_fp32_b1b4.bmodel', help='bmodel path')
    parser.add_argument("--det_batch_size", type=int, default=4)
    parser.add_argument('--det_limit_side_len', type=int, default=[960])
    #parser.add_argument("--cls_thresh", type=float, default=0.9)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
