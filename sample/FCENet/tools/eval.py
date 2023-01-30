#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import argparse
import json
import logging
import numpy as np
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)
from eval_det_iou import DetectionIoUEvaluator

logging.basicConfig(level=logging.DEBUG)


class DetFCEMetric(object):
    def __init__(self, main_indicator='hmean', **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        '''
       batch: a list produced by dataloaders.
           image: np.ndarray  of shape (N, C, H, W).
           ratio_list: np.ndarray  of shape(N,2)
           polygons: np.ndarray  of shape (N, K, P, 2), the polygons of objective regions.
           ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
       preds: a list of dict produced by post process
            points: np.ndarray of shape (N, K, P, 2), the polygons of objective regions.
       '''
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]

        for pred, gt_polyons, ignore_tags in zip(preds, gt_polyons_batch,
                                                 ignore_tags_batch):
            # prepare gt
            gt_info_list = [{
                'points': gt_polyon,
                'text': '',
                'ignore': ignore_tag
            } for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)]
            # prepare det
            det_info_list = [{
                'points': det_polyon,
                'text': '',
                'score': score
            } for det_polyon, score in zip(pred['points'], pred['scores'])]

            for score_thr in self.results.keys():
                det_info_list_thr = [
                    det_info for det_info in det_info_list
                    if det_info['score'] >= score_thr
                ]
                result = self.evaluator.evaluate_image(gt_info_list,
                                                       det_info_list_thr)
                self.results[score_thr].append(result)

    def get_metric(self):
        """
        return metrics {'heman':0,
            'thr 0.3':'precision: 0 recall: 0 hmean: 0',
            'thr 0.4':'precision: 0 recall: 0 hmean: 0',
            'thr 0.5':'precision: 0 recall: 0 hmean: 0',
            'thr 0.6':'precision: 0 recall: 0 hmean: 0',
            'thr 0.7':'precision: 0 recall: 0 hmean: 0',
            'thr 0.8':'precision: 0 recall: 0 hmean: 0',
            'thr 0.9':'precision: 0 recall: 0 hmean: 0',
            }
        """
        metrics = {}
        hmean = 0
        for score_thr in self.results.keys():
            metric = self.evaluator.combine_results(self.results[score_thr])
            # for key, value in metric.items():
            #     metrics['{}_{}'.format(key, score_thr)] = value
            metric_str = 'precision:{:.5f} recall:{:.5f} hmean:{:.5f}'.format(
                metric['precision'], metric['recall'], metric['hmean'])
            metrics['thr {}'.format(score_thr)] = metric_str
            hmean = max(hmean, metric['hmean'])
        metrics['hmean'] = hmean

        self.reset()
        return metrics

    def reset(self):
        self.results = {
            0.3: [],
            0.4: [],
            0.5: [],
            0.6: [],
            0.7: [],
            0.8: [],
            0.9: []
        }  # clear results


class DetLabelEncode(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=bool)

        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        return data

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes



# 
# def main(args):
# 
#     d_gt = dict([l[:-1].split('\t') for l in open(args.gt_path, 'r').readlines()])
#     
#     d_pred = {}
#     for l in open(args.pred_path, 'r').readlines():
#         name, prediction, score = l[:-1].split('\t')
#         name = name.split('/')[-1]
#         d_pred[name]  = prediction
# 
#     correct = 0
#     for k, gt in d_gt.items():
#         prediction = d_pred[k]
#         if gt==prediction:
#             correct += 1
#     acc = correct / float(len(d_gt))
# 
#     logging.info('gt_path: {}'.format(args.gt_path))
#     logging.info('pred_path: {}'.format(args.pred_path))
#     logging.info('ACC: {:.5f}%'.format(acc*100))
# 
# 
# if __name__ == '__main__':
#     args = argsparser()
#     main(args)
# 


def parse_args():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--gt_path', type=str, default='datasets/ctw1500/imgs/test.txt', help='path of label')
    parser.add_argument('--pred_path', type=str, default='python/results/fcenet_fp32_b1.bmodel_test_opencv_read_write_opencv_python_result.json', help='path of result')
    args = parser.parse_args()
    return args

def main():
    args = parse_args() 

    det_label_encode_op = DetLabelEncode() 
    det_fc_metric = DetFCEMetric()
    
    
    d_pred = {}
    for l in open(args.pred_path, 'r').readlines():
        filename, pred = l[:-1].split('\t')
        pred = json.loads(pred)
        pred['points'] = np.array(pred['points'])
        d_pred[filename] = pred

    data_lines = open(args.gt_path, 'r').readlines()
    for data_line in data_lines:
        filename, label = data_line.strip('\n').split('\t')
        filename = filename.split('/')[-1]
        data = {}
        data['label'] = label
        data = det_label_encode_op(data)
        gt_polygons_batch = np.expand_dims(data['polys'], axis=0)
        ignore_tags_batch = np.expand_dims(data['ignore_tags'], axis=0)
        batch = (None, None, gt_polygons_batch, ignore_tags_batch)
        if filename in d_pred:
            preds = [d_pred[filename]] 
            det_fc_metric(preds, batch)
        else:
            logging.warning('{} result is missed!'.format(filename))
    metric = det_fc_metric.get_metric()
    for k, v in metric.items():
        logging.info('{}:{}'.format(k, v))

if __name__ == '__main__':
    main()

