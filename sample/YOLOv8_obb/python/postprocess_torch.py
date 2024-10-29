import numpy as np
import torch
import cv2
import math

class PostProcess:
    def __init__(self, conf_thresh=0.001, nms_thresh=0.7, agnostic=False, multi_label=True, max_det=300):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.agnostic_nms = agnostic
        self.multi_label = multi_label
        self.max_det = max_det
        self.nms = torch_nms()
        self.ops = ops()
        self.count = 0

    def __call__(self, preds_batch, org_size_batch, ratios_batch, txy_batch):
        """Post-processes predictions and returns a list of Results objects."""
        if isinstance(preds_batch, list) and len(preds_batch) == 1:
            # 1 output
            dets = np.concatenate(preds_batch)
        class_num = dets.shape[-1] - 5
        outs = self.nms.non_max_suppression(
            dets,
            self.conf_thresh,
            self.nms_thresh,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=class_num,
            classes=None,
            rotated=True,
            multi_label=True
        )
        results = []
        for pred, (org_w, org_h), ratio, (tx1, ty1) in zip(outs, org_size_batch, ratios_batch, txy_batch):
            # rboxes为[x,y,w,h,r]
            rboxes = self.ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
            # rboxes[:, :4] = self.ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
            coords = rboxes[:, :4]
            coords[:, 0] -= tx1  # x padding
            coords[:, 1] -= ty1  # y padding
            coords[:, [0, 2]] /= ratio[0]   # x, w
            coords[:, [1, 3]] /= ratio[1]   # y, h

            # coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, org_w - 1)  # x1
            # coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, org_h - 1)  # y1

            rboxes[:, :4] = coords.round()

            # xywh, r, conf, cls
            obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1).numpy()
            results.append(np.ascontiguousarray(obb))
        self.count += 1 
        return results


class torch_nms:
    def _get_covariance_matrix(self, boxes):

        # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
        gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
        a, b, c = gbbs.split(1, dim=-1)
        cos = c.cos()
        sin = c.sin()
        cos2 = cos.pow(2)
        sin2 = sin.pow(2)
        return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

    def batch_probiou(self, obb1, obb2, eps=1e-7):
        
        obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
        obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

        x1, y1 = obb1[..., :2].split(1, dim=-1)
        x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
        a1, b1, c1 = self._get_covariance_matrix(obb1)
        a2, b2, c2 = (x.squeeze(-1)[None] for x in self._get_covariance_matrix(obb2))

        t1 = (
            ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        ) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
        t3 = (
            ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
            / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
            + eps
        ).log() * 0.5
        bd = (t1 + t2 + t3).clamp(eps, 100.0)
        hd = (1.0 - (-bd).exp() + eps).sqrt()
        return 1 - hd

    def nms_rotated(self, boxes, scores, threshold=0.45):
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.int8)
        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = self.batch_probiou(boxes, boxes).triu_(diagonal=1)
        pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
        return sorted_idx[pick]
    
    def xywh2xyxy(self, x):

        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        dw = x[..., 2] / 2  # half-width
        dh = x[..., 3] / 2  # half-height
        y[..., 0] = x[..., 0] - dw  # top left x
        y[..., 1] = x[..., 1] - dh  # top left y
        y[..., 2] = x[..., 0] + dw  # bottom right x
        y[..., 3] = x[..., 1] + dh  # bottom right y
        return y
    

    def non_max_suppression(self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_nms=30000,
        max_wh=7680,
        in_place=True,
        rotated=False,
    ):
        # Checks
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
                
        # if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        #     prediction = prediction[0]  # select only inference output

        if isinstance(prediction, np.ndarray):
            prediction = torch.from_numpy(prediction)
        
        # prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)

        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[2] - 4)  # number of classes
        nm = prediction.shape[2] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, :, 4:mi].amax(2) > conf_thres  # candidates
        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        if not rotated:
            if in_place:
                prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])  # xywh to xyxy
            else:
                prediction = torch.cat((self.xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]) and not rotated:
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
                v[:, :4] = self.xywh2xyxy(lb[:, 1:5])  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)

            if multi_label:
                i, j = torch.where(cls > conf_thres)
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            scores = x[:, 4]  # scores
            if rotated:
                boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
                i = self.nms_rotated(boxes, scores, iou_thres)
            else:
                boxes = x[:, :4] + c  # boxes (offset by class)
                i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            output[xi] = x[i]

        return output
    

class ops:
    def regularize_rboxes(self, rboxes):
        x, y, w, h, t = rboxes.unbind(dim=-1)
        # Swap edge and angle if h >= w
        w_ = torch.where(w > h, w, h)
        h_ = torch.where(w > h, h, w)
        t = torch.where(w > h, t, t + math.pi / 2) % math.pi
        return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes
    
    def clip_boxes(self, boxes, shape):
        if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
            boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
            boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
            boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
            boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes
    
    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
        """
        Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
        specified in (img1_shape) to the shape of a different image (img0_shape).

        Args:
            img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
            boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
            img0_shape (tuple): the shape of the target image, in the format of (height, width).
            ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                calculated based on the size difference between the two images.
            padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
                rescaling.
            xywh (bool): The box format is xywh or not, default=False.

        Returns:
            boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (
                round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
                round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
            )  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        if padding:
            boxes[..., 0] -= pad[0]  # x padding
            boxes[..., 1] -= pad[1]  # y padding
            if not xywh:
                boxes[..., 2] -= pad[0]  # x padding
                boxes[..., 3] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        return self.clip_boxes(boxes, img0_shape)