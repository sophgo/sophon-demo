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
import argparse
import sophon.sail as sail
from scipy.special import softmax
from utils import add_input_img, draw_bmcv, is_img
import logging
logging.basicConfig(level=logging.INFO)

class P2PNet:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]

        # get input
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype= self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}

        # get output
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_tensors = {}
        self.output_scales = {}
        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            output_dtype = self.net.get_output_dtype(self.graph_name, output_name)
            output_scale = self.net.get_output_scale(self.graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.output_tensors[output_name] = output
            self.output_scales[output_name] = output_scale

        # check batch size
        self.batch_size = self.input_shape[0]
        suppoort_batch_size = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
        if self.batch_size not in suppoort_batch_size:
            raise ValueError('batch_size must be {} for bmcv, but got {}'.format(suppoort_batch_size, self.batch_size))
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        # init preprocess
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        a = [1/(255.*x) for x in self.std]
        b = [-x/y for x,y in zip(self.mean, self.std)]
        self.alpha_beta = tuple([(ia * self.input_scale, ib * self.input_scale)
            for ia, ib in zip(a, b)])
        self.use_resize_padding = True
        self.use_vpp = False

        # init time
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, input_bmimg):
        rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                          sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)
        resized_img_rgb, ratio, txy = self.resize_bmcv(rgb_planar_img)
        preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, self.alpha_beta)
        return preprocessed_bmimg, ratio, txy

    def resize_bmcv(self, bmimg):
        """
        resize for single sail.BMImage
        :param bmimg:
        :return: a resize image of sail.BMImage
        """
        img_w = bmimg.width()
        img_h = bmimg.height()
        if self.use_resize_padding:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            if r_h > r_w:
                tw = self.net_w
                th = int(r_w * img_h)
                tx1 = tx2 = 0
                ty1 = int((self.net_h - th) / 2)
                ty2 = self.net_h - th - ty1
            else:
                tw = int(r_h * img_w)
                th = self.net_h
                tx1 = int((self.net_w - tw) / 2)
                tx2 = self.net_w - tw - tx1
                ty1 = ty2 = 0

            ratio = (min(r_w, r_h), min(r_w, r_h))
            txy = (tx1, ty1)
            attr = sail.PaddingAtrr()
            attr.set_stx(tx1)
            attr.set_sty(ty1)
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)

            preprocess_fn = self.bmcv.vpp_crop_and_resize_padding if self.use_vpp else self.bmcv.crop_and_resize_padding
            resized_img_rgb = preprocess_fn(bmimg, 0, 0, img_w, img_h, self.net_w, self.net_h, attr)
        else:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            ratio = (r_w, r_h)
            txy = (0, 0)
            preprocess_fn = self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
            resized_img_rgb = preprocess_fn(bmimg, self.net_w, self.net_h)
        return resized_img_rgb, ratio, txy

    def predict(self, input_tensor, img_num):
        """
        ensure output order: loc_data, conf_preds, mask_data, proto_data
        Args:
            input_tensor:
        Returns:
        """
        input_tensors = {self.input_name: input_tensor}
        self.net.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)
        outputs_dict = {}
        for name in self.output_names:
            outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num]
        # resort
        out_keys = list(outputs_dict.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n in k:
                    ord.append(i)
                    break
        out = [outputs_dict[out_keys[i]] for i in ord]
        return out[0], out[1]

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
            predict_cnt = int(scores_above_thresh.sum())
            # back to original image size
            tx1, ty1 = txy_list[i]
            points[:, 0] = (points[:, 0] - tx1) / ratios[0]
            points[:, 1] = (points[:, 1] - ty1) / ratios[1]
            points_batch.append(points)

        return points_batch

    def __call__(self, bmimg_list):
        ratio_list = []
        txy_list = []
        if self.batch_size == 1:
            start_time = time.time()
            preprocessed_img, ratio, txy = self.preprocess(bmimg_list[0])
            ratio_list.append(ratio)
            txy_list.append(txy)
            self.preprocess_time += time.time() - start_time
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_img, input_tensor)
        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(len(bmimg_list)):
                start_time = time.time()
                preprocessed_img, ratio, txy = self.preprocess(bmimg_list[i])
                ratio_list.append(ratio)
                txy_list.append(txy)
                bmimgs[i] = preprocessed_img.data()
                self.preprocess_time += time.time() - start_time

            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)

        start_time = time.time()
        out_infer = self.predict(input_tensor, len(bmimg_list))
        self.inference_time += time.time() - start_time
        start_time = time.time()
        points = self.postprocess(
            out_infer[0], out_infer[1], ratio_list, txy_list)
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
    frame_num = 0
    decode_time = 0.0
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
        bmimg_list = []
        for index in range(img_num):
            # decode
            start_time = time.time()
            decoder = sail.Decoder(input_list[index], True, args.dev_id)
            bmimg = sail.BMImage()
            ret = decoder.read(p2pnet.handle, bmimg)
            if ret:
                logging.error("{} decode failure.".format(input_list[index]))
                continue
            decode_time += time.time() - start_time
            frame_num += 1
            bmimg_list.append(bmimg)
            filename_list.append(input_list[index])

            if len(bmimg_list) != p2pnet.batch_size and index != (img_num - 1):
                continue
            points = p2pnet(bmimg_list)
            for i, (bmimg, p) in enumerate(zip(bmimg_list, points)):
                logging.info("{}, point nums: {}".format(frame_num - len(bmimg_list) + i + 1, len(p)))
                image_rgb_planar = p2pnet.bmcv.convert_format(bmimg)
                draw_bmcv(p2pnet.bmcv, image_rgb_planar, p)
                save_name = os.path.join(output_dir, os.path.basename(filename_list[i]))
                p2pnet.bmcv.imwrite(save_name, image_rgb_planar)
                txt_name = os.path.join(output_dir, os.path.basename(filename_list[i])).replace('.jpg', '.txt')
                with open(txt_name, 'w') as fp:
                    for pt in p:
                        fp.write(str(int(pt[0])) + ' ' + str(int(pt[1])) + '\n')
                fp.close()
            bmimg_list.clear()
            filename_list.clear()

        logging.info("result saved in {}".format(output_dir))
    else:
        output_dir = './results/video'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        decoder = sail.Decoder(args.input, True, args.dev_id)
        if not decoder.is_opened():
            raise Exception("can not open the video")
        frame_list = []
        flag = True
        while flag:
            frame = sail.BMImage()
            start_time = time.time()
            ret = decoder.read(p2pnet.handle, frame)
            decode_time += time.time() - start_time
            if ret:
                flag = False
            else:
                frame_list.append(frame)
                frame_num += 1
            if (frame_num % p2pnet.batch_size == 0 or flag == False) and len(frame_list):
                points = p2pnet(frame_list)
                for i, (bmimg, p) in enumerate(zip(frame_list, points)):
                    logging.info("{}, point nums: {}".format(frame_num - len(frame_list) + i + 1, len(p)))
                    image_rgb_planar = p2pnet.bmcv.convert_format(bmimg)
                    draw_bmcv(p2pnet.bmcv, image_rgb_planar, p)
                    save_name = os.path.join(output_dir,str(frame_num - len(frame_list) + i + 1) + '.jpg')
                    p2pnet.bmcv.imwrite(save_name, image_rgb_planar)
                    txt_name = os.path.join(output_dir, str(frame_num - len(frame_list) + i + 1) + ".txt")
                    with open(txt_name, 'w') as fp:
                        for pt in p:
                            fp.write(str(int(pt[0])) + ' ' + str(int(pt[1])) + '\n')
                    fp.close()
                frame_list.clear()
        logging.info("result saved in {}".format(output_dir))
        decoder.release()

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
    parser.add_argument('--bmodel', type=str, default="../models/BM1684X/p2pnet_bm1684x_int8_1b.bmodel", help='bmodel path')
    parser.add_argument('--dev_id', type=int, default=0, help='device id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')








