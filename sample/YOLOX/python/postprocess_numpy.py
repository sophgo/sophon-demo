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

class PostProcess:
    def __init__(self, input_h, input_w, conf_thresh=0.001, nms_thresh=0.7, agnostic=False, multi_label=True, max_det=300, p6=False):
        self.input_h = input_h
        self.input_w = input_w
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.agnostic_nms = agnostic
        self.multi_label = multi_label
        self.max_det = max_det
        self.nms = pseudo_torch_nms()

        self.grids = []
        self.expanded_strides = []

        if not p6:
            strides = [8,16,32]
        else:
            strides = [8,16,32,64]

        hsizes = [input_h // stride for stride in strides]
        wsizes = [input_w // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize),np.arange(hsize))
            grid = np.stack((xv,yv),2).reshape(1,-1,2)
            self.grids.append(grid)
            shape = grid.shape[:2]
            self.expanded_strides.append(np.full((*shape,1),stride))

        self.grids = np.concatenate(self.grids,1)
        self.expanded_strides = np.concatenate(self.expanded_strides,1)
    

    def  __call__(self, preds_batch, input_size, org_size_batch, ratios_batch, txy_batch):
        """
        post-processing
        :param preds_batch:     list of predictions in a batch
        :param org_size_batch:  list of (org_img_w, org_img_h) in a batch
        :param input_size:      (input_w,input_h)
        :param ratios_batch:    list of (ratio_x, ratio_y) in a batch when resize-and-center-padding
        :param txy_batch:       list of (tx, ty) in a batch when resize-and-center-padding
        :return:
        """
        if isinstance(preds_batch, list) and len(preds_batch) == 1:
            # 1 output
            dets = np.concatenate(preds_batch)
        else:
            print('preds_batch type: '.format(type(preds_batch)))
            raise NotImplementedError
        
        dets = self.decode(preds_batch[0])
        
        

        outs = self.nms.non_max_suppression(dets,
                                            self.conf_thresh,
                                            self.nms_thresh,
                                            agnostic=False,
                                            max_det=300,
                                            multi_label=self.multi_label,
                                            classes=None)

        # Rescale boxes from img_size to im0 size
        for det, (org_w, org_h), ratio, (tx1, ty1) in zip(outs, org_size_batch, ratios_batch, txy_batch):
            if len(det):
                # Rescale boxes from img_size to im0 size
                coords = det[:, :4]
                # coords[:, [0, 2]] -= tx1  # x padding
                # coords[:, [1, 3]] -= ty1  # y padding
                coords[:, [0, 2]] /= ratio[0]
                coords[:, [1, 3]] /= ratio[1]

                coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, org_w - 1)  # x1, x2
                coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, org_h - 1)  # y1, y2

                det[:, :4] = coords

        return outs


    def decode(self, outputs):
        for i in range(len(outputs)):
            valid_indices = np.where(outputs[..., 4] > self.conf_thresh)[1]
            expanded_strides = self.expanded_strides[:, valid_indices, :]
            grids = self.grids[:, valid_indices, :]
            outputs = outputs[:, valid_indices, :]
            outputs[i][..., :2] = (outputs[i][..., :2] + grids) * expanded_strides
            outputs[i][..., 2:4] = np.exp(outputs[i][..., 2:4]) * expanded_strides
            outputs[i][..., 5:] *= outputs[i][..., 4:5]
        
        return outputs


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
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def nms(self, pred, conf_thres=0.001, iou_thres=0.5, agnostic=False, max_det=1000):
        return self.non_max_suppression(pred, conf_thres, iou_thres,
                                        classes=None,
                                        agnostic=agnostic,
                                        multi_label=True,
                                        max_det=max_det)
    
    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.25,
                            iou_thres=0.5,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300,
                            nm=0):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 4  # number of classes
        mi = 4 + nc  # mask start index
        # xc = prediction[:, 4:mi].max(1) > conf_thres  # candidates
        xc = prediction[..., 5:].max(2) > conf_thres

        # Checks
        # assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        # assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        # output = [torch.zeros((0, 6), device=prediction.device)] * bs
        output = [np.zeros((0, 6 + nm))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf


            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:,:4])

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
