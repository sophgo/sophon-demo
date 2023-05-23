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
import sophon.sail as sail
from p2pnet_utils.preprocess_bmcv import PreProcess
from p2pnet_utils.postprocess_numpy import PostProcess
from p2pnet_utils.sophon_inference import SophonInference
from p2pnet_utils.utils import draw_bmcv, draw_numpy, is_img
import logging
logging.basicConfig(level=logging.INFO)

class P2PNet:
    def __init__(self, model_path, device_id):
        if not os.path.exists(model_path):
            raise FileNotFoundError('{} is not existed.'.format(model_path))

        self.net = SophonInference(
            model_path=model_path,
            device_id=device_id,
            input_mode=1, # use bmcv
        )
        
        self.device_id = device_id
        self.bmcv = self.net.bmcv
        self.handle = self.net.handle
        self.input_scale = list(self.net.input_scales.values())[0]
        self.img_dtype = list(self.net.img_dtypes.values())[0]

        self.batch_size = self.net.inputs_shapes[0][0]
        self.net_c = self.net.inputs_shapes[0][1]
        self.net_h = self.net.inputs_shapes[0][2]
        self.net_w = self.net.inputs_shapes[0][3]
        self.preprocess = PreProcess(
            self.net_w,
            self.net_h,
            self.batch_size,
            self.img_dtype,
            self.input_scale,
        )
        self.postprocess = PostProcess()

        print('{} is loaded.'.format(model_path))

    def predict(self, tensor):
        out_dict = self.net.infer_bmimage(tensor)

        out_keys = list(out_dict.keys())
        # print(out_keys)
        out = [out_dict[key] for key in out_keys]
        return out[0], out[1]
    
    def do_once_proc(self, file_path):
        
        batch_size = self.batch_size
        input_path = file_path

        if not os.path.exists(input_path):
            raise FileNotFoundError('{} is not existed.'.format(input_path))
        
        # imgage directory
        input_list = []
        assert is_img(input_path), "not correct img path: {}".format(input_path)
        input_list.append(input_path)
        # imgage list saved in file

        input_batch = []
        images = []
        ino = 0
        image = sail.BMImage()        
        ret = decode_image_bmcv(input_list[ino], self.handle, image, self.device_id)
        if not ret:
            # decode failed.
            print('skip: decode failed: {}'.format(input_list[ino]))
            return None
        
        images.append(image)
        input_batch.append(input_list[ino])

        if batch_size == 1:
            single_image = images[0]
            org_h, org_w = single_image.height(), single_image.width()
            # end-to-end inference
            preprocessed_img, ratio, txy = self.preprocess(
                single_image,
                self.handle,
                self.bmcv,
            )

            out_infer = self.predict([preprocessed_img])

            points, predict_cnt = self.postprocess.infer_batch(
                out_infer[0], out_infer[1], [ratio])

        else:
            print("eval just support 1 batch, actual is {}".format(batch_size))
            points = None
        return points


def decode_image_bmcv(image_path, process_handle, img, dev_id):
    # img = sail.BMImage()
    # img = sail.BMImageArray4D()
    decoder = sail.Decoder(image_path, True, dev_id)
    if isinstance(img, sail.BMImage):
        ret = decoder.read(process_handle, img)
    else:
        ret = decoder.read_(process_handle, img)
    if ret != 0:
        return False
    return True


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
    batch_size = p2pnet.batch_size
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

        support_batch_size = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
        if batch_size not in support_batch_size:
            raise NotImplementedError(
                'model batch size must be {}, but got {}.'.format(support_batch_size, batch_size))

        input_batch = []
        images = []

        for ino in range(img_num):
            image = sail.BMImage()
            start_time = time.time()
            ret = decode_image_bmcv(input_list[ino], p2pnet.handle, image, opt.dev_id)
            decode_time += time.time() - start_time
            if not ret:
                # decode failed.
                print('skip: decode failed: {}'.format(input_list[ino]))
                continue
            images.append(image)
            input_batch.append(input_list[ino])

            if len(images) != batch_size and ino != (img_num - 1):
                continue

            if batch_size == 1:
                single_image = images[0]
                org_h, org_w = single_image.height(), single_image.width()
                # end-to-end inference
                start_time = time.time()
                preprocessed_img, ratio, txy = p2pnet.preprocess(
                    single_image,
                    p2pnet.handle,
                    p2pnet.bmcv,
                )
                preprocess_time += time.time() - start_time
                start_time = time.time()
                out_infer = p2pnet.predict([preprocessed_img])
                inference_time += time.time() - start_time
                start_time = time.time()
                points, predict_cnt = p2pnet.postprocess.infer_batch(
                    out_infer[0], out_infer[1], [ratio])
                postprocess_time += time.time() - start_time
                
                image_rgb_planar = p2pnet.net.bmcv.convert_format(single_image)
                draw_bmcv(p2pnet.bmcv, image_rgb_planar, points[0])
                save_basename, _ = os.path.splitext(os.path.basename(opt.bmodel)) 
                input_name, _ = os.path.splitext(os.path.basename(input_batch[0])) 
                save_basename = save_basename + '_bmcv_python_result_{}.jpg'.format(input_name)
                save_name = os.path.join(output_dir, save_basename)
                p2pnet.bmcv.imwrite(save_name, image_rgb_planar)

                txt_name = os.path.join(output_dir, os.path.basename(input_batch[0])).replace('.jpg', '.txt')
                with open(txt_name, 'w') as fp:
                    for pt in points[0]:
                        fp.write(str(int(pt[0])) + ' ' + str(int(pt[1])) + '\n')

            else:
                # padding params
                cur_bs = len(images)
                padding_bs = batch_size - cur_bs
                # adjustment for BMImageArray
                bm_array = eval('sail.BMImageArray{}D'.format(batch_size))

                org_size_list = []
                for i in range(len(input_batch)):
                    org_h, org_w = images[i].height(), images[i].width()
                    org_size_list.append((org_w, org_h))
                start_time = time.time()
                resized_imgs = bm_array(p2pnet.handle,
                                        p2pnet.net_h,
                                        p2pnet.net_w,
                                        sail.FORMAT_RGB_PLANAR,
                                        sail.DATA_TYPE_EXT_1N_BYTE
                                        )
                # batch end-to-end inference
                resized_img_list, ratio_list, txy_list = p2pnet.preprocess.resize_batch(
                    images,
                    p2pnet.handle,
                    p2pnet.bmcv,
                )

                for i in range(len(input_batch)):
                    resized_imgs.copy_from(i, resized_img_list[i])

                # padding is not necessary for bmcv in preprcessing
                # for i in range(cur_bs, batch_size):
                #     resized_imgs.copy_from(i, resized_img_list[0])

                preprocessed_imgs = p2pnet.preprocess.norm_batch(
                    resized_imgs,
                    p2pnet.handle,
                    p2pnet.bmcv,
                )
                preprocess_time += time.time() - start_time
                start_time = time.time()
                out_infer = p2pnet.predict([preprocessed_imgs])
                inference_time += time.time() - start_time
                # # cancel padding data
                # if padding_bs != 0:
                #     out_infer = [e_data[:cur_bs] for e_data in out_infer]
                start_time = time.time()
                points, predict_cnt = p2pnet.postprocess.infer_batch(
                    out_infer[0], out_infer[1], ratio_list)
                postprocess_time += time.time() - start_time
                for i, (e_img, p) in enumerate(zip(images, points)):
                    image_rgb_planar = sail.BMImage(
                        p2pnet.handle, e_img.height(), e_img.width(),
                        sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
                    p2pnet.net.bmcv.convert_format(e_img, image_rgb_planar)
                    draw_bmcv(p2pnet.bmcv, image_rgb_planar, points[i])
                    save_basename, _ = os.path.splitext(os.path.basename(opt.bmodel)) 
                    input_name, _ = os.path.splitext(os.path.basename(input_batch[i])) 
                    save_basename = save_basename + '_bmcv_python_result_{}.jpg'.format(input_name)
                    save_name = os.path.join(output_dir, save_basename)
                    p2pnet.bmcv.imwrite(save_name, image_rgb_planar)

            images.clear()
            input_batch.clear()

        print('the results is saved: {}'.format(os.path.abspath(output_dir)))
        logging.info("decode_time(ms): {:.2f}".format(decode_time / img_num * 1000))
        logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time / img_num * 1000))
        logging.info("inference_time(ms): {:.2f}".format(inference_time / img_num * 1000))
        logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time / img_num * 1000))
    else:
        decoder = sail.Decoder(opt.input, True, opt.dev_id)
        if not decoder.is_opened():
            raise Exception("can not open the video")
        frame_list = []
        id = 0
        frame = sail.BMImage()
        while True:
            start_time = time.time()
            ret = decoder.read(p2pnet.handle, frame)
            if ret:
                break
            decode_time += time.time() - start_time
            org_h, org_w = frame.height(), frame.width()
            start_time = time.time()
            preprocessed_img, ratio, txy = p2pnet.preprocess( frame,
                    p2pnet.handle,
                    p2pnet.bmcv, )
            preprocess_time += time.time() - start_time
            frame_list.append(preprocessed_img)
            if len(frame_list) == batch_size:
                id = id + batch_size
                start_time = time.time()
                out_infer = p2pnet.predict([preprocessed_img])
                inference_time += time.time() - start_time
                start_time = time.time()
                points, predict_cnt = p2pnet.postprocess.infer_batch(
                    out_infer[0], out_infer[1], [ratio])
                postprocess_time += time.time() - start_time
                image_rgb_planar = sail.BMImage(p2pnet.handle, 
                    frame.height(), frame.width(),
                    sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
                p2pnet.net.bmcv.convert_format(frame, image_rgb_planar)
                draw_bmcv(p2pnet.bmcv, image_rgb_planar, points[0])
                
                save_basename, _ = os.path.splitext(os.path.basename(opt.bmodel))
                save_basename = save_basename + '_bmcv_python_result_{}.jpg'.format(id)
                save_name = os.path.join(output_dir, save_basename)
                p2pnet.bmcv.imwrite(save_name, image_rgb_planar)
                frame_list.clear()
        if len(frame_list):
                id = id + len(frame_list)
                start_time = time.time()
                out_infer = p2pnet.predict([frame_list])
                inference_time += time.time() - start_time
                start_time = time.time()
                points, predict_cnt = p2pnet.postprocess.infer_batch(
                    out_infer[0], out_infer[1], [ratio])
                postprocess_time += time.time() - start_time
                image_rgb_planar = sail.BMImage(p2pnet.handle, 
                    frame.height(), frame.width(),
                    sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
                p2pnet.net.bmcv.convert_format(frame, image_rgb_planar)
                draw_bmcv(p2pnet.bmcv, image_rgb_planar, points[0])
                
                save_basename, _ = os.path.splitext(os.path.basename(opt.bmodel))
                save_basename = save_basename + '_bmcv_python_result_{}.jpg'.format(id)
                save_name = os.path.join(output_dir, save_basename)
                p2pnet.bmcv.imwrite(save_name, image_rgb_planar)
                frame_list.clear()
        decoder.release()
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








