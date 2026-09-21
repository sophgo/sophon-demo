
#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import numpy as np
import cv2 as cv2
import sophon.sail as sail

COCO_CLASSES = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

def _as_fp32(arr):
    """sail returns fp16 tensor data as raw uint16 bits; convert to float32."""
    if arr.dtype == np.uint16:
        return arr.view(np.float16).astype(np.float32)
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    return arr


def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)

def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)

def draw_and_visualize(filename, im, bboxes, segments, vis=False, save=True, using_COCO_name=True, draw_thresh=0.25):

    # Draw rectangles and polygons
    im_canvas = im.copy()
    for (*box, confidence, label), segment in zip(bboxes, segments):

        if confidence < draw_thresh :continue
        color=random_color(int(label))
        #draw contour and fill mask
        if(len(segment)):
            for seg in segment:
                # cv2.polylines(im, np.int32([np.int32([seg]).reshape(-1,1,2)]), True, color, 2)  # white borderline
                cv2.fillPoly(im_canvas, np.int32([np.int32([seg]).reshape(-1,1,2)]), color)

        # draw bbox rectangle
        left, top, right, bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        cv2.rectangle(im, (left, top), (right, bottom), color = color ,thickness=2, lineType=cv2.LINE_AA)
        if using_COCO_name:
            caption = f"{COCO_CLASSES[int(label)]} {confidence:.3f}"
        else:
            caption = f"class:{int(label)} {confidence:.3f}"
        w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
        cv2.rectangle(im, (left - 3, top - 33), (left + w + 10, top), color, -1)
        cv2.putText(im, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

    # Mix image
    im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

    if save:
        cv2.imwrite(filename, im)
        print(f"output been saved as {filename}")
    return im

class PostProcess:
    def __init__(self, handle, bmcv):
        self.handle = handle
        self.bmcv = bmcv
    def __call__(self, outputs, im0_shape, ratio, txy):
        results=[]
        masks_uncrop, seg_out = None, None
        for out in outputs.values():
            if(len(out.shape()) == 3):
                masks_uncrop = out
            if(len(out.shape()) == 2):
                out.sync_d2s()
                seg_out = _as_fp32(out.asnumpy())
        for i in range(len(txy)):
            boxes, masks, roi = self.postprocess(masks_uncrop, seg_out, im0_shape[i], ratio[i], txy[i][0], txy[i][1])
            results.append([boxes, masks, roi])
        return results
    
    def postprocess(self, masks_uncrop:sail.Tensor, seg_out, im0_shape, ratio, pad_w, pad_h):
        if(seg_out is None or seg_out.shape[0] <= 0):
            return [],[],()
        im0_h, im0_w = im0_shape[:2]

        # bbox: model -> original-image coords (float), same numerics as before
        boxes = seg_out.copy()
        boxes[:, :4] -= [pad_w, pad_h, pad_w, pad_h]
        boxes[:, :4] /= min(ratio)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, im0_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, im0_h)

        # integer bbox ROI: std::round (floor(x+0.5) for x>=0) to match C++ yolov8_seg_fuse.cpp:414-419
        x1 = np.floor(boxes[:, 0] + 0.5).astype(np.int32)
        y1 = np.floor(boxes[:, 1] + 0.5).astype(np.int32)
        x2 = np.floor(boxes[:, 2] + 0.5).astype(np.int32)
        y2 = np.floor(boxes[:, 3] + 0.5).astype(np.int32)
        keep = (x2 > x1 + 1) & (y2 > y1 + 1)
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            return [], [], ()

        boxes = boxes[idx]
        x1, y1, x2, y2 = x1[idx], y1[idx], x2[idx], y2[idx]

        # 3: resize / upsample only the surviving mask channels
        masks_scaled = self.scale_mask(masks=masks_uncrop, im0_shape=im0_shape[:2], indices=idx)

        # 2: crop each mask to its bbox ROI and threshold (replaces full-res broadcast gate)
        n = idx.size
        masks = np.zeros((n, im0_h, im0_w), dtype=bool)
        for k in range(n):
            masks[k, y1[k]:y2[k], x1[k]:x2[k]] = masks_scaled[k, y1[k]:y2[k], x1[k]:x2[k]] > 127.5

        # ROI returned so contours can be computed outside the timed region (draw-only)
        return boxes, masks, (x1, y1, x2, y2)

    @staticmethod
    def masks2segments(masks, x1, y1, x2, y2):
        segments = []
        for m, xs, ys, xe, ye in zip(masks, x1, y1, x2, y2):
            roi = m[ys:ye, xs:xe].astype('uint8')  # bbox-area only, replaces full-res astype
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=len).reshape(-1, 2)
                contour += [xs, ys]
                segments.append([contour.astype('float32').flatten()])
            else:
                segments.append([])
        return segments

    def scale_mask(self, masks:sail.Tensor, im0_shape, ratio_pad=None, indices=None):
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
        im1_shape = masks.shape()[1:]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        width = right - left
        height = bottom - top
        
        # bmcv solution
        if masks.dtype() == sail.Dtype.BM_UINT8:
            mask_size = im1_shape[0] * im1_shape[1]
            masks_resized = []
            if indices is None:
                indices = range(masks.shape()[0])
            for i in indices:
                mask_tensor = sail.Tensor(masks, [1,1,im1_shape[1],im1_shape[0]], int(i) * mask_size)
                mask_bmimg = sail.BMImage(self.handle, im1_shape[1], im1_shape[0], sail.Format.FORMAT_GRAY, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
                self.bmcv.tensor_to_bm_image(mask_tensor, mask_bmimg, sail.Format.FORMAT_GRAY)
                mask_bmimg_resized = self.bmcv.crop_and_resize(mask_bmimg, left, top, width, height, im0_shape[1], im0_shape[0], sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)
                mask_bmimg_resized.unalign()
                mask_tensor_resized = self.bmcv.bm_image_to_tensor(mask_bmimg_resized)
                masks_resized.append(mask_tensor_resized.asnumpy().squeeze())
            return np.stack(masks_resized)
        else:
            # opencv solution
            masks.sync_d2s()
            masks = _as_fp32(masks.asnumpy()).transpose(1,2,0)
            if indices is not None:
                masks = masks[:, :, indices]
            masks = masks[top:bottom, left:right]
            masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))#,
                            #interpolation=cv2.INTER_CUBIC)  # INTER_CUBIC would be better
            if len(masks.shape) == 2:
                masks = masks[:, :, None]
            # Note: TPU fuse model already outputs sigmoid*255, no need to scale again
            return masks.transpose(2, 0, 1)
