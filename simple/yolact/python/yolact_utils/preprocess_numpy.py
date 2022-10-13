import numpy as np
import cv2

class PreProcess:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        self.std = np.array([57.38, 57.12, 58.40], dtype=np.float32)

        self.normalize = cfg['normalize']
        self.subtract_means = cfg['subtract_means']
        self.to_float = cfg['to_float']

        self.width = cfg['width']
        self.height = cfg['height']

    def __call__(self, img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        preprocessed_img = cv2.resize(img, (self.width, self.height)).astype(np.float32)
        if self.normalize:
            preprocessed_img = (preprocessed_img - self.mean) / self.std
        elif self.subtract_means:
            preprocessed_img = (preprocessed_img - self.mean)
        elif self.to_float:
            preprocessed_img /= 255.
        chw_img = preprocessed_img[:, :, ::-1].transpose((2, 0, 1))
        return chw_img.astype(np.float32)

    def infer_batch(self, img_list):
        """
        batch pre-processing
        Args:
            img_list: a list of (h,w,3) numpy.ndarray or numpy.ndarray with (n,h,w,3)

        Returns: (n,3,h,w) numpy.ndarray after pre-processing

        """
        preprocessed_img_list = []
        for img in img_list:
            preprocessed_img = self(img)
            preprocessed_img_list.append(preprocessed_img)
        return np.array(preprocessed_img_list)