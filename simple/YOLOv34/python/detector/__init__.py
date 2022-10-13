'''
Date: 1970-01-01 08:00:00
LastEditors: zhangjin
LastEditTime: 2020-12-08 11:22:03
FilePath: /inference_service_by_trt_python_on_django/detector/__init__.py
'''
import sys; sys.path.append("..")
from detector.sophon.yolov3.model import YOLOV3

detect_dict = {
        "yolov34": YOLOV3,
    }

class DetectFactory(object):
    def __new__(cls, name, **kwargs):
        if name in detect_dict.keys():
            return detect_dict[name](**kwargs)
        else:
            raise KeyError("Invalid detector, got '{}'," \
                    "but expected to be one of {}".format(name, detect_dict.keys))