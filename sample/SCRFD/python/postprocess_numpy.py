#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
import cv2
import pdb
# import scipy.special

class PostProcess:
    def __init__(self, conf_thresh=0.5, nms_thresh=0.5, agnostic=False, multi_label=False, max_det=1000):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.agnostic_nms = agnostic
        self.multi_label = multi_label
        self.max_det = max_det
        self.nms = pseudo_torch_nms()


    def  __call__(self, preds_batch, scores_batch, org_size_batch, ratios_batch, txy_batch):
        """
        post-processing
        :param preds_batch:     list of predictions in a batch
        :para scores_batch:     list of scores in a batch
        :param org_size_batch:  list of (org_img_w, org_img_h) in a batch
        :param ratios_batch:    list of (ratio_x, ratio_y) in a batch when resize-and-center-padding
        :return:
        """
        if isinstance(preds_batch, list) and isinstance(scores_batch, list) and len(scores_batch) != len(preds_batch):
            print('preds_batch type: '.format(type(preds_batch)))
            raise NotImplementedError
        dets = np.concatenate((preds_batch, scores_batch, scores_batch), axis=2)
        outs = self.nms.non_max_suppression(
            dets,
            conf_thres=self.conf_thresh,
            iou_thres=self.nms_thresh,
            classes=None,
            agnostic=self.agnostic_nms,
            multi_label=self.multi_label,
            labels=(),
            max_det=self.max_det,
        )
        return outs


# numpy multiclass nms implementation from original yolov5 repo torch implementation
class pseudo_torch_nms:
    def nms_boxes(self, boxes, scores, iou_thres):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

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
        keep = np.array(keep)
        return keep

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.copy() if isinstance(x, np.ndarray) else np.copy(x)
        # y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        # y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 0] = x[:, 0]                # top left x
        y[:, 1] = x[:, 1]                # top left y
        y[:, 2] = x[:, 0] + x[:, 2]      # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3]      # bottom right y
        return y

    def box_area(self, box):
        # box = xyxy(4,n)
        return (box[2] - box[0]) * (box[3] - box[1])

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
        # inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
        inter = (np.min([a2, b2], 0) - np.max([a1, b1], 0)).clip(min=0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / (self.box_area(box1.T)[:, None] + self.box_area(box2.T) - inter)

    def nms(self, pred, conf_thres=0.25, iou_thres=0.5, agnostic=False, max_det=1000):
        return self.non_max_suppression(pred, conf_thres, iou_thres,
                                        classes=None,
                                        agnostic=agnostic,
                                        multi_label=False,
                                        max_det=max_det)

    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.25,
                            iou_thres=0.5,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - 5 # number of classes
        xc = prediction[:,:,4] > conf_thres  # candidates

        # Checks
        # assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        # assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        # output = [torch.zeros((0, 6), device=prediction.device)] * bs
        output = [np.zeros((0, 6))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero()
                x = np.concatenate([box[i], x[i, j + 5, None], j[:, None].astype(np.float32)], 1)
            else:  # best class only
                conf = x[:, 5:].max(1, keepdims=True)
                j_argmax = x[:, 5:].argmax(1)
                j = j_argmax if j_argmax.shape == x[:, 5:].shape else \
                    np.expand_dims(j_argmax, 1)  # for argmax(axis, keepdims=True)
                x = np.concatenate([box, conf, j.astype(np.float32)], 1)[conf.reshape(-1) > conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x_argsort = np.argsort(x[:, 4])[:max_nms] # sort by confidence
                x = x[x_argsort]

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

            #############################################
            # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = self.nms_boxes(boxes, scores, iou_thres)
            ############################################

            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]

            output[xi] = x[i]

        return output
