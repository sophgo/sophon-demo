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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
import shutil
import numpy as np
import cv2
import argparse
import configparser
from yolact_utils.preprocess_bmcv import PreProcess
from yolact_utils.postprocess_numpy import PostProcess
from yolact_utils.sophon_inference import SophonInference
import sophon.sail as sail
from yolact_utils.utils import draw_bmcv, draw_numpy, is_img


class Detector(object):
    def __init__(self, cfg_path, bmodel_path, device_id=0,
                 conf_thresh=0.5, nms_thresh=0.5, keep_top_k=200):
        try:
            self.get_config(cfg_path)
        except Exception as e:
            raise e

        if not os.path.exists(bmodel_path):
            raise FileNotFoundError('{} is not existed.'.format(bmodel_path))
        self.net = SophonInference(model_path=bmodel_path,
                                   device_id=device_id,
                                   input_mode=1)
        print('{} is loaded.'.format(bmodel_path))

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.keep_top_k = keep_top_k

        self.output_order_node = ['loc', 'conf', 'mask', 'proto']
        self.bmcv = self.net.bmcv
        self.handle = self.net.handle
        self.input_scale = list(self.net.input_scales.values())[0]
        self.img_dtype = list(self.net.img_dtypes.values())[0]

        self.batch_size = self.net.inputs_shapes[0][0]
        self.preprocess = PreProcess(
            self.cfg,
            self.batch_size,
            self.img_dtype,
            self.input_scale,
        )
        self.postprocess = PostProcess(
            self.cfg,
            self.conf_thresh,
            self.nms_thresh,
            self.keep_top_k,
        )

    def get_config(self, cfg_path):
        if not os.path.exists(cfg_path):
            raise FileNotFoundError('{} is not existed.'.format(cfg_path))

        config = configparser.ConfigParser()
        config.read(cfg_path)

        normalize = config.get("yolact", "normalize")
        subtract_means = config.get("yolact", "subtract_means")
        to_float = config.get("yolact", "to_float")

        width = config.get("yolact", "width")
        height = config.get("yolact", "height")
        conv_ws = config.get("yolact", "conv_ws")
        conv_hs = config.get("yolact", "conv_hs")
        aspect_ratios = config.get("yolact", "aspect_ratios")
        scales = config.get("yolact", "scales")
        variances = config.get("yolact", "variances")

        self.cfg = dict()

        self.cfg['normalize'] = int(normalize.split(',')[0])
        self.cfg['subtract_means'] = int(subtract_means.split(',')[0])
        self.cfg['to_float'] = int(to_float.split(',')[0])
        self.cfg['width'] = int(width.split(',')[0])
        self.cfg['height'] = int(height.split(',')[0])
        self.cfg['conv_ws'] = [int(i) for i in conv_ws.replace(' ', '').split(',')]
        self.cfg['conv_hs'] = [int(i) for i in conv_hs.replace(' ', '').split(',')]
        self.cfg['aspect_ratios'] = [float(i) for i in aspect_ratios.replace(' ', '').split(',')]
        self.cfg['scales'] = [int(i) for i in scales.replace(' ', '').split(',')]
        self.cfg['variances'] = [float(i) for i in variances.replace(' ', '').split(',')]

    def predict(self, tensor):
        """
        ensure output order: loc_data, conf_preds, mask_data, proto_data
        Args:
            tensor:

        Returns:

        """
        # feed: [input0]

        out_dict = self.net.infer_bmimage(tensor)
        # resort
        out_keys = list(out_dict.keys())
        ord = []
        for n in self.output_order_node:
            for i, k in enumerate(out_keys):
                if n in k:
                    ord.append(i)
                    break
        out = [out_dict[out_keys[i]] for i in ord]
        return out


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
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    else:
        shutil.rmtree(opt.output_dir)
        os.makedirs(opt.output_dir)

    yolact = Detector(
        opt.cfgfile,
        opt.model,
        device_id=opt.dev_id,
        conf_thresh=opt.conf_thresh,
        nms_thresh=opt.nms_thresh,
        keep_top_k=opt.keep,
    )

    batch_size = yolact.net.inputs_shapes[0][0]
    input_path = opt.input_path
    video_detect_frame_num = opt.video_detect_frame_num

    if not os.path.exists(input_path):
        raise FileNotFoundError('{} is not existed.'.format(input_path))

    if opt.is_video:
        if batch_size != 1:
            raise ValueError(
                'bmodel batch size must be 1 in video inference, but got {}'.format(
                    batch_size)
            )
        # decode
        decoder = sail.Decoder(input_path, True, opt.dev_id)
        if decoder.is_opened():
            print("create decoder success")
            frame = sail.BMImage()
            id = 0
            while True:
                if id >= video_detect_frame_num:
                    break

                ret = decoder.read(yolact.handle, frame)
                if ret:
                    print("stream end or decoder error")
                    break

                org_h, org_w = frame.height(), frame.width()
                preprocessed_img = yolact.preprocess(frame,
                                                     yolact.handle,
                                                     yolact.bmcv,
                                                     )

                out_infer = yolact.predict([preprocessed_img])

                classid, conf_scores, boxes, masks = \
                    yolact.postprocess(*out_infer, (org_w, org_h))

                # bmcv cannot draw with instance masks, so we convert BMImage to numpy to draw
                image_bgr_planar = sail.BMImage(yolact.handle, frame.height(), frame.width(),
                                                sail.Format.FORMAT_BGR_PLANAR, frame.dtype())
                yolact.bmcv.convert_format(frame, image_bgr_planar)
                image_tensor = yolact.bmcv.bm_image_to_tensor(image_bgr_planar)
                image_chw_numpy = image_tensor.asnumpy()[0]
                image_numpy = np.transpose(image_chw_numpy, [1, 2, 0]).copy()

                draw_numpy(image_numpy, boxes, masks=masks, classes_ids=classid, conf_scores=conf_scores)
                # draw_bmcv(yolact.bmcv, image, boxes, masks=masks, classes_ids=classid, conf_scores=conf_scores)

                save_basename = 'res_bmcv_{}'.format(id)
                save_name = os.path.join(opt.output_dir, save_basename.replace('.jpg', ''))
                cv2.imencode('.jpg', image_numpy)[1].tofile('{}.jpg'.format(save_name))
                # yolact.bmcv.imwrite('{}.jpg'.format(save_name), image)
                id += 1

        else:
            print("failed to create decoder")

    else:

        # imgage directory
        input_list = []
        if os.path.isdir(input_path):
            for img_name in os.listdir(input_path):
                if is_img(img_name):
                    input_list.append(os.path.join(input_path, img_name))
                    # imgage file
        elif is_img(input_path):
            input_list.append(input_path)
        # imgage list saved in file
        else:
            with open(input_path, 'r', encoding='utf-8') as fin:
                for line in fin.readlines():
                    line_head = line.strip("\n").split(' ')[0]
                    if is_img(line_head):
                        input_list.append(line_head)

        img_num = len(input_list)

        suppoort_batch_size = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
        if batch_size not in suppoort_batch_size:
            raise ValueError('batch_size must be {}, but got {}'.format(suppoort_batch_size, batch_size))

        inp_batch = []
        images = []
        for ino in range(img_num):
            image = sail.BMImage()
            ret = decode_image_bmcv(input_list[ino], yolact.handle, image, opt.dev_id)
            if not ret:
                # decode failed.
                print('skip: decode failed: {}'.format(input_list[ino]))
                continue
            images.append(image)
            inp_batch.append(input_list[ino])

            if len(images) != batch_size and ino != (img_num - 1):
                continue

            if batch_size == 1:
                single_image = images[0]
                org_h, org_w = single_image.height(), single_image.width()
                # end-to-end inference
                preprocessed_img = yolact.preprocess(single_image,
                                                     yolact.handle,
                                                     yolact.bmcv,
                                                     )

                out_infer = yolact.predict([preprocessed_img])

                classid, conf_scores, boxes, masks = \
                    yolact.postprocess(*out_infer, (org_w, org_h))

                # bmcv cannot draw with instance masks, so we convert BMImage to numpy to draw
                image_bgr_planar = sail.BMImage(yolact.handle, single_image.height(), single_image.width(),
                                                sail.Format.FORMAT_BGR_PLANAR, single_image.dtype())
                yolact.bmcv.convert_format(single_image, image_bgr_planar)
                image_tensor = yolact.bmcv.bm_image_to_tensor(image_bgr_planar)
                image_chw_numpy = image_tensor.asnumpy()[0]
                image_numpy = np.transpose(image_chw_numpy, [1, 2, 0]).copy()

                draw_numpy(image_numpy, boxes, masks=masks, classes_ids=classid, conf_scores=conf_scores)
                # draw_bmcv(yolact.bmcv, image, boxes, masks=masks, classes_ids=classid, conf_scores=conf_scores)

                save_basename = 'res_bmcv_{}'.format(os.path.basename(inp_batch[0]))
                save_name = os.path.join(opt.output_dir, save_basename.replace('.jpg', ''))
                cv2.imencode('.jpg', image_numpy)[1].tofile('{}.jpg'.format(save_name))
                # yolact.bmcv.imwrite('{}.jpg'.format(save_name), image)
                print('{}.jpg is saved.'.format(save_name))
            else:
                # padding params
                cur_bs = len(images)
                padding_bs = batch_size - cur_bs

                bm_array = eval('sail.BMImageArray{}D'.format(batch_size))

                org_size_list = []
                for i in range(len(inp_batch)):
                    org_h, org_w = images[i].height(), images[i].width()
                    org_size_list.append((org_w, org_h))

                resized_imgs = bm_array(yolact.handle,
                                        yolact.preprocess.height,
                                        yolact.preprocess.width,
                                        sail.FORMAT_RGB_PLANAR,
                                        sail.DATA_TYPE_EXT_1N_BYTE
                                        )

                # batch end-to-end inference
                resized_img_list = yolact.preprocess.resize_batch(
                    images,
                    yolact.handle,
                    yolact.bmcv,
                )

                for i in range(len(inp_batch)):
                    resized_imgs.copy_from(i, resized_img_list[i])

                # # padding is not necessary for bmcv in preprcessing
                # for i in range(cur_bs, batch_size):
                #     resized_imgs.copy_from(i, resized_img_list[0])

                preprocessed_imgs = yolact.preprocess.norm_batch(
                    resized_imgs,
                    yolact.handle,
                    yolact.bmcv,
                )

                out_infer = yolact.predict([preprocessed_imgs])

                # cancel padding data
                if padding_bs != 0:
                    out_infer = [e_data[:cur_bs] for e_data in out_infer]

                classid_list, conf_scores_list, boxes_list, masks_list = \
                    yolact.postprocess.infer_batch(out_infer, org_size_list)

                for i, (e_img, classid, conf_scores, boxes, masks) in enumerate(zip(images,
                                                                                    classid_list,
                                                                                    conf_scores_list,
                                                                                    boxes_list,
                                                                                    masks_list)):
                    # bmcv cannot draw with instance masks, so we convert BMImage to numpy to draw
                    image_bgr_planar = sail.BMImage(yolact.handle, e_img.height(), e_img.width(),
                                                    sail.Format.FORMAT_BGR_PLANAR, e_img.dtype())
                    yolact.bmcv.convert_format(e_img, image_bgr_planar)
                    image_tensor = yolact.bmcv.bm_image_to_tensor(image_bgr_planar)
                    image_chw_numpy = image_tensor.asnumpy()[0]
                    image_numpy = np.transpose(image_chw_numpy, [1, 2, 0]).copy()

                    draw_numpy(image_numpy, boxes, masks=masks, classes_ids=classid, conf_scores=conf_scores)
                    # draw_bmcv(yolact.bmcv, e_img, boxes, masks=masks, classes_ids=classid, conf_scores=conf_scores)

                    save_basename = 'res_bmcv_{}'.format(os.path.basename(inp_batch[i]))
                    save_name = os.path.join(opt.output_dir, save_basename.replace('.jpg', ''))
                    cv2.imencode('.jpg', image_numpy)[1].tofile('{}.jpg'.format(save_name))
                    # yolact.bmcv.imwrite('{}.jpg'.format(save_name), e_img)

            images.clear()
            inp_batch.clear()

        print('the results is saved: {}'.format(os.path.abspath(opt.output_dir)))


def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--cfgfile', type=str, help='model config file')
    parser.add_argument('--model', type=str, help='bmodel path')
    parser.add_argument('--dev_id', type=int, default=0, help='device id')
    image_path = os.path.join(os.path.dirname(__file__),"../data/images/000000162415.jpg")
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    parser.add_argument('--keep', type=int, default=100, help='keep top-k')
    parser.add_argument('--is_video', default=0, type=int, help="input is video?")
    parser.add_argument('--input_path', type=str, default=image_path, help='input path')
    DEFAULT_OUTPUT_DIR = os.path.join(__dir__, 'results', 'results_bmcv')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='output image directory')
    parser.add_argument('--video_detect_frame_num', type=int, default=10, help='detect frame number of video')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    print('all done.')