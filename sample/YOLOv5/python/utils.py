#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import sys
import numpy as np
import cv2

COCO_CLASSES = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COLORS = [
        [56, 0, 255],
        [226, 255, 0],
        [0, 94, 255],
        [0, 37, 255],
        [0, 255, 94],
        [255, 226, 0],
        [0, 18, 255],
        [255, 151, 0],
        [170, 0, 255],
        [0, 255, 56],
        [255, 0, 75],
        [0, 75, 255],
        [0, 255, 169],
        [255, 0, 207],
        [75, 255, 0],
        [207, 0, 255],
        [37, 0, 255],
        [0, 207, 255],
        [94, 0, 255],
        [0, 255, 113],
        [255, 18, 0],
        [255, 0, 56],
        [18, 0, 255],
        [0, 255, 226],
        [170, 255, 0],
        [255, 0, 245],
        [151, 255, 0],
        [132, 255, 0],
        [75, 0, 255],
        [151, 0, 255],
        [0, 151, 255],
        [132, 0, 255],
        [0, 255, 245],
        [255, 132, 0],
        [226, 0, 255],
        [255, 37, 0],
        [207, 255, 0],
        [0, 255, 207],
        [94, 255, 0],
        [0, 226, 255],
        [56, 255, 0],
        [255, 94, 0],
        [255, 113, 0],
        [0, 132, 255],
        [255, 0, 132],
        [255, 170, 0],
        [255, 0, 188],
        [113, 255, 0],
        [245, 0, 255],
        [113, 0, 255],
        [255, 188, 0],
        [0, 113, 255],
        [255, 0, 0],
        [0, 56, 255],
        [255, 0, 113],
        [0, 255, 188],
        [255, 0, 94],
        [255, 0, 18],
        [18, 255, 0],
        [0, 255, 132],
        [0, 188, 255],
        [0, 245, 255],
        [0, 169, 255],
        [37, 255, 0],
        [255, 0, 151],
        [188, 0, 255],
        [0, 255, 37],
        [0, 255, 0],
        [255, 0, 170],
        [255, 0, 37],
        [255, 75, 0],
        [0, 0, 255],
        [255, 207, 0],
        [255, 0, 226],
        [255, 245, 0],
        [188, 255, 0],
        [0, 255, 18],
        [0, 255, 75],
        [0, 255, 151],
        [255, 56, 0],
        [245, 255, 0],
    ]

def draw_numpy(image, boxes, masks=None, classes_ids=None, conf_scores=None):
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) + 1]
        else:
            color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=1)
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            cv2.putText(image, COCO_CLASSES[classes_ids[idx] + 1] + ':' + str(round(conf_scores[idx], 2)),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=1)
        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
        print(" class id=", classes_ids[idx], " score = ", conf_scores[idx], "   (x1={},y1={},x2={},y2={})".format(x1, y1, x2, y2))
    return image

def draw_bmcv(bmcv, image, boxes, masks=None, classes_ids=None, conf_scores=None):
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        if classes_ids is not None:
            color = np.array(COLORS[int(classes_ids[idx]) + 1]).astype(np.uint8).tolist()
        else:
            color = (0, 0, 255)
        bmcv.rectangle(image, x1, y1, (x2 - x1), (y2 - y1), color, 2)
        print(" class id=", int(classes_ids[idx]), " score = ", conf_scores[idx], "   (x1={},y1={},w={},h={})".format(x1, y1, x2-x1, y2-y1))

def is_img(file_name):
    """judge the file is available image or not
    Args:
        file_name (str): input file name
    Returns:
        (bool) : whether the file is available image
    """
    fmt = os.path.splitext(file_name)[-1]
    if isinstance(fmt, str) and fmt.lower() in ['.jpg','.png','.jpeg','.bmp','.jpeg','.webp']:
        return True
    else:
        return False

# scipy: https://zhuanlan.zhihu.com/p/375344904
def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
        .. versionadded:: 0.11.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.
        .. versionadded:: 0.15.0
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.
        .. versionadded:: 0.12.0
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).
        .. versionadded:: 0.16.0
    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.

    """
    # a = _asarray_validated(a, check_finite=False)
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out

# scipy.special.softmax: https://zhuanlan.zhihu.com/p/375344904
def softmax(x, axis=None):
    r"""
    Softmax function
    The softmax function transforms each element of a collection by
    computing the exponential of each element divided by the sum of the
    exponentials of all the elements. That is, if `x` is a one-dimensional
    numpy array::
        softmax(x) = np.exp(x)/sum(np.exp(x))
    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.
    Returns
    -------
    s : ndarray
        An array the same shape as `x`. The result will sum to 1 along the
        specified axis.

    """
    # compute in log space for numerical stability
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))