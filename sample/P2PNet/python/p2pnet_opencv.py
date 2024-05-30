#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import os
import time
import shutil
import numpy as np
import cv2
import argparse
import sophon.sail as sail
from scipy.special import softmax
from utils import draw_numpy, is_img, decode_image_opencv, add_input_img
import logging
logging.basicConfig(level=logging.INFO)

class P2PNet:
    def __init__(self, args):
        if not os.path.exists(args.bmodel):
            raise FileNotFoundError('{} is not existed.'.format(args.bmodel))

        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)) * 255.0
        self.scale = np.array([1/0.229, 1/0.224, 1/0.225]).reshape((1, 1, 3)) * 1 / 255.0

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, img_list):
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
            letterbox_img, ratio, (tx1, ty1) = self.letterbox(
                img,
                new_shape=(self.net_h, self.net_w),
                color=(114, 114, 114),
                auto=False,
                scaleFill=False,
                scaleup=True,
                stride=32
            )
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
            # HWC to CHW, BGR to RGB
            img = cv2.cvtColor(letterbox_img, cv2.COLOR_BGR2RGB).astype('float32')
            img = (img - self.mean) * self.scale
            img = np.transpose(img, (2, 0, 1))
            preprocessed_img = np.ascontiguousarray(np.expand_dims(img, 0)).astype(np.float32)
            preprocessed_img_list.append(preprocessed_img)
        return np.concatenate(preprocessed_img_list), ratio_list, txy_list

    def letterbox(self, im, new_shape=(512, 512), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
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

    def predict(self, input_img, img_num):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)
        # resort
        out_keys = list(outputs.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n == k:
                    ord.append(i)
                    break
        out = [outputs[out_keys[i]][:img_num] for i in ord]
        return out

    def postprocess(self, pred_logits_batch, pred_points_batch, ratios_batch, txy_list, conf_thresh = 0.5):
        """
        post-processing using single post-processing for loop
        :param pred_logits_batch:
        :param pred_points_batch:
        :return:
        """
        outputs_scores_batch = softmax(pred_logits_batch, axis=-1)
        outputs_points_batch = pred_points_batch
        points_batch = []

        for i in range(len(ratios_batch)):
            pred_scores = outputs_scores_batch[0][i]
            pred_points = outputs_points_batch[0][i]
            ratios = ratios_batch[i]
            outputs_scores = pred_scores[:, 1]
            outputs_points = pred_points
            # filter the predictions
            scores_above_thresh = outputs_scores > conf_thresh
            points = outputs_points[scores_above_thresh]
            # back to original image size
            tx1, ty1 = txy_list[i]
            points[:, 0] = (points[:, 0] - tx1) / ratios[0]
            points[:, 1] = (points[:, 1] - ty1) / ratios[1]
            points_batch.append(points)

        return points_batch

    def __call__(self, img_list):
        start_time = time.time()
        preprocessed_img, ratio_list, txy_list = self.preprocess(img_list)
        self.preprocess_time += time.time() - start_time

        start_time = time.time()
        input_num = len(preprocessed_img)
        if input_num != self.batch_size:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:input_num] = np.stack(preprocessed_img)
            preprocessed_img = input_img

        infer_results = self.predict(preprocessed_img, input_num)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        points = self.postprocess(
            infer_results[0], infer_results[1], ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return points


def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    output_dir = './results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    p2pnet = P2PNet(args)
    decode_time = 0.0
    frame_num = 0
    if os.path.isdir(args.input) or is_img(args.input):
        output_dir = './results/images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        input_list = add_input_img(args.input)
        img_num = len(input_list)
        filename_list = []
        img_list = []
        for index in range(img_num):
            start_time = time.time()
            image = decode_image_opencv(input_list[index])
            if image is None:
                logging.error('imdecode is None: {}'.format(input_list[index]))
                continue
            decode_time += time.time() - start_time
            frame_num += 1
            img_list.append(image)
            filename_list.append(input_list[index])
            if len(img_list) != p2pnet.batch_size and index != (img_num - 1):
                continue

            points = p2pnet(img_list)
            for i, (e_img, p) in enumerate(zip(img_list, points)):
                logging.info("{}, point nums: {}".format(frame_num - len(img_list) + i + 1, len(p)))
                vis_image = draw_numpy(e_img, p)
                save_name = os.path.join(output_dir, os.path.basename(filename_list[i]))
                cv2.imencode('.jpg', vis_image)[1].tofile(save_name)
                txt_name = os.path.join(output_dir, os.path.basename(filename_list[i])).replace('.jpg', '.txt')
                with open(txt_name, 'w') as fp:
                    for pt in p:
                        fp.write(str(int(pt[0])) + ' ' + str(int(pt[1])) + '\n')
                fp.close()
            img_list.clear()
            filename_list.clear()
        logging.info("result saved in {}".format(output_dir))
    else:
        output_dir = './results/video'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        cap = cv2.VideoCapture()
        if not cap.open(args.input):
            raise Exception("can not open the video")
        frame_list = []
        flag = True
        while flag:
            start_time = time.time()
            ret, frame = cap.read()
            decode_time += time.time() - start_time
            if not ret or frame is None:
                flag = False
            else:
                frame_list.append(frame)
                frame_num += 1
            if (frame_num % p2pnet.batch_size == 0 or flag == False) and len(frame_list):
                points = p2pnet(frame_list)
                for i, (e_img, p) in enumerate(zip(frame_list, points)):
                    logging.info("{}, point nums: {}".format(frame_num - len(frame_list) + i + 1, len(p)))
                    vis_image = draw_numpy(e_img, p)
                    save_name = os.path.join(output_dir, str(frame_num - len(frame_list) + i + 1))
                    cv2.imencode('.jpg', vis_image)[1].tofile(save_name)
                    txt_name = os.path.join(output_dir, str(frame_num - len(frame_list) + i + 1) + ".txt")
                    with open(txt_name, 'w') as fp:
                        for pt in p:
                            fp.write(str(int(pt[0])) + ' ' + str(int(pt[1])) + '\n')
                    fp.close()
                frame_list.clear()
        cap.release()
        logging.info("result saved in {}".format(output_dir))

    # calculate speed
    logging.info("------------------ P2PNet Time Info ----------------------")
    decode_time = decode_time / frame_num
    preprocess_time = p2pnet.preprocess_time / frame_num
    inference_time = p2pnet.inference_time / frame_num
    postprocess_time = p2pnet.postprocess_time / frame_num
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default="../datasets/test/images", help='input image path')
    parser.add_argument('--bmodel', type=str, default="../models/BM1684X/p2pnet_bm1684x_int8_4b.bmodel", help='bmodel path')
    parser.add_argument('--dev_id', type=int, default=0, help='device id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')