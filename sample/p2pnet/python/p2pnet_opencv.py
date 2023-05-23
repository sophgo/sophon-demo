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
os.chdir(os.path.abspath(os.path.dirname(sys.argv[0])))
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
import time
import shutil
import numpy as np
import cv2
import argparse
from p2pnet_utils.sophon_inference import SophonInference
from p2pnet_utils.preprocess_numpy import PreProcess
from p2pnet_utils.postprocess_numpy import PostProcess
from p2pnet_utils.utils import draw_numpy, is_img
import logging
logging.basicConfig(level=logging.INFO)

class P2PNet:
    def __init__(self, model_path, device_id):
        if not os.path.exists(model_path):
            raise FileNotFoundError('{} is not existed.'.format(model_path))

        self.net = SophonInference(
            model_path=model_path,
            device_id=device_id,
            input_mode=0, # use cv
        )

        self.batch_size = self.net.inputs_shapes[0][0]
        self.net_c = self.net.inputs_shapes[0][1]
        self.net_h = self.net.inputs_shapes[0][2]
        self.net_w = self.net.inputs_shapes[0][3]
        self.preprocess = PreProcess(self.net_w, self.net_h)
        self.postprocess = PostProcess()

        print('{} is loaded.'.format(model_path))

    def predict(self, tensor):
        if tensor.ndim != 4:
            tensor = np.expand_dims(tensor, 0)
        # feed: [input0]
        out_dict = self.net.infer_numpy([tensor])
        # print(out_dict)

        out_keys = list(out_dict.keys())
        # print(out_keys)
        out = [out_dict[key] for key in out_keys]
        return out[0], out[1]
    
    def do_once_proc(self, file_path):
        
        batch_size = self.net.inputs_shapes[0][0]
        input_path = file_path

        if not os.path.exists(input_path):
            raise FileNotFoundError('{} is not existed.'.format(input_path))
        
        # image directory
        input_list = []
        assert is_img(input_path), "not correct img path: {}".format(input_path)
        input_list.append(input_path)
        # image list saved in file

        input_batch = []
        images = []
        result = []
        ino = 0        
        image = decode_image_opencv(input_list[ino])
        if image is None:
            print('skip: image data is none: {}'.format(input_list[ino]))
            return None
        
        images.append(image)
        input_batch.append(input_list[ino])

        org_size_list = []
        ratios_list = []
        for i in range(len(input_batch)):
            org_h, org_w = images[i].shape[:2]
            org_size_list.append((org_w, org_h))
            org_h, org_w = images[i].shape[:2]
            ratio_h, ratio_w = self.net_h / org_h, self.net_w / org_w
            ratios_list.append((ratio_w, ratio_h))

        # batch end-to-end inference
        preprocessed_img = self.preprocess.infer_batch(images)
        out_infer = self.predict(preprocessed_img)
        points, predict_cnt = self.postprocess.infer_batch(
            out_infer[0], out_infer[1], ratios_list)
        
        return points
        
    
def decode_image_opencv(image_path):
    try:
        with open(image_path, "rb") as f:
            image = np.array(bytearray(f.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except:
        image = None
    return image

def main(opt):
    if not os.path.exists(opt.input):
        raise FileNotFoundError('{} is not existed.'.format(opt.input))
    if not os.path.exists(opt.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(opt.bmodel))

    output_dir = './results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    p2pnet = P2PNet(model_path=opt.bmodel, device_id=opt.dev_id)
    batch_size = p2pnet.net.inputs_shapes[0][0]
    input_path = opt.input
    
    decode_time = 0.0
    preprocess_time = 0.0
    inference_time = 0.0
    postprocess_time = 0.0
    if os.path.isdir(opt.input) or is_img(opt.input):
        # image directory
        input_list = []
        if os.path.isdir(input_path):
            for img_name in os.listdir(input_path):
                if is_img(img_name):
                    input_list.append(os.path.join(input_path, img_name))
                    # image file
        elif is_img(input_path):
            input_list.append(input_path)
        # image list saved in file
        else:
            with open(input_path, 'r', encoding='utf-8') as fin:
                for line in fin.readlines():
                    line_head = line.strip("\n").split(' ')[0]
                    if is_img(line_head):
                        input_list.append(line_head)

        img_num = len(input_list)

        input_batch = []
        images = []
        for ino in range(img_num):
            # print(ino, input_list[ino])
            start_time = time.time()
            image = decode_image_opencv(input_list[ino])
            decode_time += time.time() - start_time
            if image is None:
                print('skip: image data is none: {}'.format(input_list[ino]))
                continue
            images.append(image)
            input_batch.append(input_list[ino])

            if len(images) != batch_size and ino != (img_num - 1):
                continue

            org_size_list = []
            ratio_list = []
            for i in range(len(input_batch)):
                org_h, org_w = images[i].shape[:2]
                org_size_list.append((org_w, org_h))
                ratio_h, ratio_w = p2pnet.net_h / org_h, p2pnet.net_w / org_w
                ratio_list.append((ratio_w, ratio_h))

            # batch end-to-end inference
            start_time = time.time()
            preprocessed_img = p2pnet.preprocess.infer_batch(images)
            preprocess_time += time.time() - start_time
            start_time = time.time()
            out_infer = p2pnet.predict(preprocessed_img)
            inference_time += time.time() - start_time
            # # cancel padding data
            # if padding_bs != 0:
            #     out_infer = [e_data[:cur_bs] for e_data in out_infer]
            start_time = time.time()
            points, predict_cnt = p2pnet.postprocess.infer_batch(
                out_infer[0], out_infer[1], ratio_list)
            postprocess_time += time.time() - start_time
            # print(points, predict_cnt)

            for i, (e_img, p) in enumerate(zip(images, points)):
                vis_image = draw_numpy(e_img, p)
                save_basename, _ = os.path.splitext(os.path.basename(opt.bmodel)) 
                input_name, _ = os.path.splitext(os.path.basename(input_batch[i])) 
                save_basename = save_basename + '_opencv_python_result_{}'.format(input_name)
                save_name = os.path.join(output_dir, save_basename)
                # print(save_name)
                cv2.imencode('.jpg', vis_image)[1].tofile('{}.jpg'.format(save_name))
                txt_name = os.path.join(output_dir, os.path.basename(input_batch[i])).replace('.jpg', '.txt')
                with open(txt_name, 'w') as fp:
                    for pt in p:
                        fp.write(str(int(pt[0])) + ' ' + str(int(pt[1])) + '\n')

            images.clear()
            input_batch.clear()
        print('the results is saved: {}'.format(os.path.abspath(output_dir)))
        logging.info("decode_time(ms): {:.2f}".format(decode_time / img_num * 1000))
        logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time / img_num * 1000))
        logging.info("inference_time(ms): {:.2f}".format(inference_time / img_num * 1000))
        logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time / img_num * 1000))
    else:
        if batch_size != 1:
            raise ValueError(
                'bmodel batch size must be 1 in video inference, but got {}'.format(
                    batch_size)
            )

        cap = cv2.VideoCapture(input_path)
        start_time = time.time()
        ret, frame = cap.read()
        decode_time += time.time() - start_time
        id = 0

        while ret and frame is not None:
            id += 1
            org_h, org_w = frame.shape[:2]
            ratio_h, ratio_w = p2pnet.net_h / org_h, p2pnet.net_w / org_w
            ratios_list = [(ratio_w, ratio_h)]
            start_time = time.time()
            preprocessed_img = p2pnet.preprocess(frame)
            preprocess_time += time.time() - start_time
            start_time = time.time()
            out_infer = p2pnet.predict(preprocessed_img)
            inference_time += time.time() - start_time
            start_time = time.time()
            points, predict_cnt = p2pnet.postprocess.infer_batch(
                out_infer[0], out_infer[1], ratios_list)
            postprocess_time += time.time() - start_time
            vis_image = draw_numpy(frame.copy(), points[0])
            save_basename, _ = os.path.splitext(os.path.basename(opt.bmodel))
            save_basename = save_basename + '_opencv_python_result_{}'.format(id)
            save_name = os.path.join(output_dir, save_basename)
            cv2.imencode('.jpg', vis_image)[1].tofile('{}.jpg'.format(save_name))
            start_time = time.time()
            ret, frame = cap.read()
            decode_time += time.time() - start_time
        cap.release()
        logging.info("decode_time(ms): {:.2f}".format(decode_time / id * 1000))
        logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time / id * 1000))
        logging.info("inference_time(ms): {:.2f}".format(inference_time / id * 1000))
        logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time / id * 1000))

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default="../datasets/calibration_table", help='input image path')
    parser.add_argument('--bmodel', type=str, default="../models/BM1684X/p2pnet_bm1684x_int8_1b.bmodel", help='bmodel path')
    parser.add_argument('--dev_id', type=int, default=0, help='device id')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    print('all done.')