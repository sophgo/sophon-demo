'''
Date: 1970-01-01 08:00:00
LastEditors: zhangjin
LastEditTime: 2020-12-09 10:55:09
FilePath: /inference_service_by_trt_python_on_django/configs/config.py
'''
import yaml
from pprint import pprint
from easydict import EasyDict as edict
import time

args = edict()

args.DEV_ID = [0,] 

args.IMG_DIR = ""
args.VIDEO_FILE = ""
args.RTSP_URL = ""

args.SRC_TYPE = 3 # 0-IMGS 1-VIDEO 2-RTSP 3-HTTP POST IMG
args.RUN_MODE = "service" # "single" "batch" "batch_thread" "service"

args.DEBUG = True
args.PRINT = True
args.WRITE_VIDEO = True
args.SAVE_IMAGE = True # False  True
args.SHOW_IMAGE = True

args.TIME_PREFIX= time.strftime('%Y-%m%d-%H%M-%S',time.localtime(time.time()))

## general model config
args.DETECTOR = edict()
args.DETECTOR.TYPE = ""
args.DETECTOR.NAME = "yolov3"  # "yolov3"
args.DETECTOR.PRODUCER_NAME = ""

## tensorrt model config
args.DETECTOR.EIGIEN_FILE = ""
args.DETECTOR.LABEL_FILE = ""
args.DETECTOR.DEV_ID = 0

args.DETECTOR.YOLO_MASKS = [(6, 7, 8), (3, 4, 5), (0, 1, 2)] 
args.DETECTOR.YOLO_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                    (59, 119), (116, 90), (156, 198), (373, 326)] 

args.DETECTOR.OUTPUT_TENSOR_CHANNELS = []
args.DETECTOR.MIN_CONFIDENCE = 0.5
args.DETECTOR.NMS_MAX_OVERLAP = 0.45
args.DETECTOR.MIN_WIDTH = 0
args.DETECTOR.MIN_HEIGHT = 0
args.DETECTOR.MIN_IMAGE_SIZE = args.DETECTOR.MIN_HEIGHT * args.DETECTOR.MIN_WIDTH

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in args:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        args[k][vk] = vv
                else:
                    args[k] = v
            else:
                raise ValueError("key {} must exist in configs.py".format(k))
    return args

if __name__ == '__main__':
    args = update_config("./detector.yml")
    pprint(args)
