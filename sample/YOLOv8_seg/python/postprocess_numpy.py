#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import math
import time
import cv2
import numpy as np
from pycocotools.mask import encode
from utils import *



class PostProcess:

    def __init__(self, conf_thres=0.7, iou_thres=0.5, num_masks=32):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks
        self.nms = pseudo_torch_nms()
  
    def __call__(self, outputs,im0_shape,ratio, txy):
        results=[]
        for i in range(outputs[0].shape[0]):
            output=[outputs[0][i][np.newaxis,:],outputs[1][i][np.newaxis,:]]
            results.append(self.postprocess(output,im0_shape[i],ratio[i], txy[i][0], txy[i][1],self.conf_threshold,self.iou_threshold,self.num_masks))
        return results

  

    # def postprocess(self, preds, im0_shape, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
    #     """
    #     Post-process the prediction.

    #     Args:
    #         preds (Numpy.ndarray): predictions come from ort.session.run().
    #         im0 (Numpy.ndarray): [h, w, c] original input image.
    #         ratio (tuple): width, height ratios in letterbox.
    #         pad_w (float): width padding in letterbox.
    #         pad_h (float): height padding in letterbox.
    #         conf_threshold (float): conf threshold.
    #         iou_threshold (float): iou threshold.
    #         nm (int): the number of masks.

    #     Returns:
    #         boxes (List): list of bounding boxes.
    #         segments (List): list of segments.
    #         masks (np.ndarray): [N, H, W], output masks.
    #     """
    #     x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

    #     # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
    #     x = np.einsum('bcn->bnc', x)

    #     # Predictions filtering by conf-threshold
    #     x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

    #     # Create a new matrix which merge these(box, score, cls, nm) into one
    #     # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
    #     x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

    #     # NMS filtering
    #     if(x.shape[0]):
    #         x = x[self.nms.nms_boxes(x[:, :4], x[:, 4], iou_threshold)]
    #     if(x.shape[0]>512):x=x[:512]
    #     # Decode and return
    #     if len(x) > 0:

    #         # Bounding boxes format change: cxcywh -> xyxy
    #         x[..., [0, 1]] -= x[..., [2, 3]] / 2
    #         x[..., [2, 3]] += x[..., [0, 1]]

    #         # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
    #         x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
    #         x[..., :4] /= min(ratio)

    #         # Bounding boxes boundary clamp
    #         x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0_shape[1])
    #         x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0_shape[0])
    #         x = x[np.bitwise_and(x[..., 2] > x[..., 0], x[..., 3] > x[..., 1])]
    #         # Process masks
    #         masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0_shape)

    #         # Masks -> Segments(contours)
    #         segments = self.masks2segments(masks)
    #         return x[..., :6], segments, masks  # boxes, segments, masks
    #     else:
    #         return [], [], []
    def postprocess(self, preds, im0_shape, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum('bcn->bnc', x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        if(x.shape[0]):
            x = x[self.nms.nms_boxes(x[:, :4], x[:, 4], iou_threshold)]
       
        ans1,ans2,ans3=[],[],[]
        post_batch_size = 1
        for i in range((int(x.shape[0]/post_batch_size)+1)):
            X=x[i*post_batch_size:min((i+1)*post_batch_size,x.shape[0])]
            X=self.get_mask_distrubute(X,im0_shape, ratio, pad_w, pad_h,protos)
            ans1.extend(X[0])
            ans2.extend(X[1])
            ans3.extend(X[2])
        return ans1,ans2,ans3
        
    def get_mask_distrubute(self,x,im0_shape, ratio, pad_w, pad_h,protos):
        if len(x) > 0:

            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0_shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0_shape[0])

            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0_shape)

            # Masks -> Segments(contours)
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []
    @staticmethod
    def masks2segments(masks):
        """
        It takes a list of masks(n,h,w) and returns a list of segments(n,xy) (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L750)

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype('uint8'):
            # c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # CHAIN_APPROX_SIMPLE
            # if c:
            #     c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            # else:
            #     c = np.zeros((0, 2))  # no segments found
            contours, _ = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if(contours):
                contours = np.array(contours[np.array([len(x) for x in contours]).argmax()])
                coco_segmentation = [contours.flatten().astype('float32')]                            
                segments.append(coco_segmentation)
            else:
                segments.append([])

        return segments

    @staticmethod
    def crop_mask(masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L599)

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
        but is slower. (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L618)

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum('HWN -> NHW', masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))#,
                           #interpolation=cv2.INTER_CUBIC)  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    def draw_and_visualize(self, filename,im, bboxes, segments, vis=False, save=True):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
            segments (List): list of segment masks.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """

        # Draw rectangles and polygons
        im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            if conf < 0.25 :continue
            color=colors[int(cls_)]
            #draw contour and fill mask
            if(len(segment)):
                for seg in segment:
                    cv2.polylines(im, np.int32([np.int32([seg]).reshape(-1,1,2)]), True, color, 2)  # white borderline
                    cv2.fillPoly(im_canvas, np.int32([np.int32([seg]).reshape(-1,1,2)]), color)

            # draw bbox rectangle
            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          color, 1, cv2.LINE_AA)
            cv2.putText(im, f'{class_names[int(cls_)]}: {conf:.3f}', (int(box[0]), int(box[1] - 9)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Mix image
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # # Show image
        # if vis:
        #     cv2.imshow('demo', im)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        #Save image
        if save:
            cv2.imwrite(filename+".jpg", im)
        return im
    
class pseudo_torch_nms:
    def nms_boxes(self, boxes, scores, iou_thres):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

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

    

if __name__ == '__main__':
    
    p=PostProcess(conf_thres=0.5, iou_thres=0.3)
    