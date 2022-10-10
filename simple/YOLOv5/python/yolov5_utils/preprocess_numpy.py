import numpy as np
import cv2

class PreProcess:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
                  stride=32):
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

    def __call__(self, img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (1,3,h,w) numpy.ndarray after pre-processing

        """
        letterbox_data, ratio, (tx1, ty1) = self.letterbox(
            img,
            new_shape=(self.height, self.width),
            color=(114, 114, 114),
            auto=False,
            scaleFill=False,
            scaleup=True,
            stride=32
        )

        letterbox_data = letterbox_data.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        input_data = letterbox_data.astype(np.float32)
        input_data = np.expand_dims(input_data, 0)
        inp = np.ascontiguousarray(input_data / 255.0)
        return inp, ratio, (tx1, ty1)


    def infer_batch(self, img_list):
        """
        batch pre-processing
        Args:
            img_list: a list of (h,w,3) numpy.ndarray or numpy.ndarray with (n,h,w,3)

        Returns: (n,3,h,w) numpy.ndarray after pre-processing

        """
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for img in img_list:
            preprocessed_img, ratio, (tx1, ty1) = self(img)
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
        return np.concatenate(preprocessed_img_list), ratio_list, txy_list