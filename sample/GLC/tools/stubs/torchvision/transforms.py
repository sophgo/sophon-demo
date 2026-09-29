"""只实现 GLC metric 用到的 ToTensor。"""
import numpy as np
import torch


class ToTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            if pic.ndim == 2:
                pic = pic[:, :, None]
            return torch.from_numpy(pic.transpose((2, 0, 1)).copy()).float().div(255.0)
        raise TypeError("ToTensor stub 仅支持 numpy 数组")
