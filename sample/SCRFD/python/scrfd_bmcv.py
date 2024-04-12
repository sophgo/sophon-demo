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
import argparse
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
import logging

logging.basicConfig(level=logging.INFO)


class SCRFD:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))

        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]

        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}
        
        self.output_names = self.net.get_output_names(self.graph_name)
        if len(self.output_names) not in [9]:
            raise ValueError('only suport 9 outputs, but got {} outputs bmodel'.format(len(self.output_names)))
        
        self.output_tensors = {}
        self.output_scales = {}
        self.output_shapes = []
        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            output_dtype = self.net.get_output_dtype(self.graph_name, output_name)
            output_scale = self.net.get_output_scale(self.graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.output_tensors[output_name] = output
            self.output_scales[output_name] = output_scale
            self.output_shapes.append(output_shape)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        # init preprocess
        self.use_resize_padding = True
        self.use_vpp = False
        self.ab = [x * self.input_scale / 255.  for x in [1, 0, 1, 0, 1, 0]]
        
        self.keep_ratio = True
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        
        self.max_det = 1000
        
        self.agnostic = False
        self.multi_label = True
        self.postprocess = PostProcess(
                conf_thresh=self.conf_thresh,
                nms_thresh=self.nms_thresh,
                agnostic=self.agnostic,
                multi_label=self.multi_label,
                max_det=self.max_det,
            )
                       
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
    
    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
    
    
    def preprocess_bmcv(self, input_bmimg):
        rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                          sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)
        resized_img_rgb, ratio, txy = self.resize_bmcv(rgb_planar_img)
        preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, ((self.ab[0], self.ab[1]), \
                                                                 (self.ab[2], self.ab[3]), \
                                                                 (self.ab[4], self.ab[5])))
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
           
    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def distance2kps(self, points, distance, max_shape=None):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)
    
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
        return out
    
    def postprocess_scrfd(self, outs_batch, ratio, pad_list):
        bboxes_batch = []
        scores_batch = []
        for batch_size_idx in range(len(ratio)):
            (ratioh_rec, ratiow_rec) = ratio[batch_size_idx]
            (padw, padh) = pad_list[batch_size_idx]
            scores_list, bboxes_list, kpss_list = [], [], []
            for idx, stride in enumerate(self._feat_stride_fpn):
                scores = outs_batch[idx][batch_size_idx]
                bbox_preds = outs_batch[idx + self.fmc * 1][batch_size_idx] * stride
                kps_preds = outs_batch[idx + self.fmc * 2][batch_size_idx] * stride
                height = self.input_shape[2] // stride
                width = self.input_shape[3] // stride
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                pos_inds = np.where(scores >= self.conf_thresh)[0]
                bboxes = self.distance2bbox(anchor_centers, bbox_preds)
                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]
                scores_list.append(pos_scores)
                bboxes_list.append(pos_bboxes)

                kpss = self.distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
            scores = np.vstack(scores_list)
            bboxes = np.vstack(bboxes_list)
            kpss = np.vstack(kpss_list)
            bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
            ratioh, ratiow = 1 / ratioh_rec, 1 / ratiow_rec
            bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow
            bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
            bboxes[:, 2] = bboxes[:, 2] * ratiow
            bboxes[:, 3] = bboxes[:, 3] * ratioh
            kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
            kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
            bboxes_batch.append(bboxes)
            scores_batch.append(scores)
        return bboxes_batch, scores_batch
     
    
    def __call__(self, bmimg_list):
        img_num = len(bmimg_list)
        ori_size_list = []
        ori_w_list = []
        ori_h_list = []
        ratio_list = []
        txy_list = []
        if self.batch_size == 1:
            ori_h, ori_w =  bmimg_list[0].height(), bmimg_list[0].width()
            ori_size_list.append((ori_w, ori_h))
            ori_w_list.append(ori_w)
            ori_h_list.append(ori_h)
            start_time = time.time()      
            preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(bmimg_list[0])
            self.preprocess_time += time.time() - start_time
            ratio_list.append(ratio)
            txy_list.append(txy)
            
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimg, input_tensor)
                
        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                ori_h, ori_w =  bmimg_list[i].height(), bmimg_list[i].width()
                ori_size_list.append((ori_w, ori_h))
                ori_w_list.append(ori_w)
                ori_h_list.append(ori_h)
                start_time = time.time()
                preprocessed_bmimg, ratio, txy  = self.preprocess_bmcv(bmimg_list[i])
                self.preprocess_time += time.time() - start_time
                ratio_list.append(ratio)
                txy_list.append(txy)
                bmimgs[i] = preprocessed_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)
            
        start_time = time.time()
        outputs = self.predict(input_tensor, img_num)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        outputs_scrfd, scores_scrfd = self.postprocess_scrfd(outputs, ratio_list, txy_list)
        results_batch = []
        for i,(outputs_scrfd_i,scores_scrfd_i) in enumerate(zip(outputs_scrfd, scores_scrfd)):
            outputs_scrfd_i = np.expand_dims(outputs_scrfd_i, axis=0)
            scores_scrfd_i = np.expand_dims(scores_scrfd_i, axis=0)
            results = self.postprocess(outputs_scrfd_i, scores_scrfd_i, ori_size_list[i], ratio_list[i], txy_list[i])
            results_batch.append(results[0])
        self.postprocess_time += time.time() - start_time
        return results_batch
    
def draw_bmcv(bmcv, bmimg, boxes, conf_scores=None, save_path=""):
    img_bgr_planar = bmcv.convert_format(bmimg)
    thickness = 2
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        logging.debug("score={}, (x1={},y1={},w={},h={})".format(conf_scores[idx], x1, y1, x2-x1, y2-y1))
        if conf_scores[idx] < 0.25:
            continue
        color = (0, 0, 255)
        if (x2 - x1) <= thickness * 2 or (y2 - y1) <= thickness * 2:
            logging.info("width or height too small, this rect will not be drawed: (x1={},y1={},w={},h={})".format(x1, y1, x2-x1, y2-y1))
        else:
            bmcv.rectangle(img_bgr_planar, x1, y1, (x2 - x1), (y2 - y1), color, thickness)
    bmcv.imwrite(save_path, img_bgr_planar)
    
    
    
def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    
    # creat save path
    output_dir = "./results"
    results_txt_dir = "./tools/prediction_dir"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(results_txt_dir) and args.eval:
        os.mkdir(results_txt_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)
    output_txt_dir = os.path.join(output_dir, 'txt_results')
    if not os.path.exists(output_txt_dir):
        os.mkdir(output_txt_dir) 
    
    # initialize net
    scrfd = SCRFD(args)
    batch_size = scrfd.batch_size

    handle = sail.Handle(args.dev_id)
    bmcv = sail.Bmcv(handle)
    
    scrfd.init()
    
    decode_time = 0.0
    # test images
    if os.path.isdir(args.input): 
        bmimg_list = []
        filename_list = []
        results_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                img_file = os.path.join(root, filename)
                img_dir = os.path.basename(root)
                if not os.path.exists(os.path.join(results_txt_dir, img_dir)) and args.eval:
                    os.mkdir(os.path.join(results_txt_dir, img_dir))
                elif not os.path.exists(os.path.join(output_txt_dir, img_dir)):
                    os.mkdir(os.path.join(output_txt_dir, img_dir))
                cn += 1
                logging.info("{}, img_file: {}".format(cn, img_file))
                # decode
                start_time = time.time()
                decoder = sail.Decoder(img_file, True, args.dev_id)
                bmimg = sail.BMImage()
                ret = decoder.read(handle, bmimg)
                if ret != 0:
                    continue

                decode_time += time.time() - start_time
                
                bmimg_list.append(bmimg)
                filename_list.append(filename)
                if (len(bmimg_list) == batch_size or cn == len(filenames)) and len(bmimg_list):
                    # predict
                    results = scrfd(bmimg_list)
                    
                    for i, filename in enumerate(filename_list):
                        det = results[i]
                        if args.eval:
                            # save result
                            img_name = filename.rsplit('.', 1)[0]
                            txt_name = img_name + '.txt'
                            with open(os.path.join(results_txt_dir, img_dir, txt_name), 'w') as txt_f:
                                txt_f.write(filename + '\n')
                                txt_f.write(str(det.shape[0]) + '\n')
                                for idx in range(det.shape[0]):
                                    x1, y1, x2, y2, score, _ = det[idx]
                                    width = abs(x2 - x1)
                                    height = abs(y2 - y1)
                                    txt_f.write("{:} {:} {:} {:} {:}\n".format(x1, y1, width, height, score))
                        else:
                            # save image
                            save_path = os.path.join(output_img_dir, filename)
                            draw_bmcv(bmcv, bmimg_list[i], det[:,:4], conf_scores=det[:, -2], save_path=save_path)
                            
                            # save result
                            img_name = filename.rsplit('.', 1)[0]
                            txt_name = img_name + '.txt'
                            with open(os.path.join(output_txt_dir, img_dir, txt_name), 'w') as txt_f:
                                txt_f.write(filename + '\n')
                                txt_f.write(str(det.shape[0]) + '\n')
                                for idx in range(det.shape[0]):
                                    x1, y1, x2, y2, score, _ = det[idx]
                                    width = abs(x2 - x1)
                                    height = abs(y2 - y1)
                                    txt_f.write("{:} {:} {:} {:} {:}\n".format(x1, y1, width, height, score))
                        
                    bmimg_list.clear()
                    filename_list.clear()

        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        if args.eval:
            logging.info("result saved in {} ".format(results_txt_dir))
        else:            
            logging.info("result saved in {}".format(os.path.join(output_txt_dir)))
        
    # test video
    else:
        decoder = sail.Decoder(args.input, False, args.dev_id)
        if not decoder.is_opened():
            raise Exception("can not open the video")
        video_name = os.path.splitext(os.path.split(args.input)[1])[0]
        cn = 0
        frame_list = []
        end_flag = False
        while not end_flag:
            frame = sail.BMImage()
            start_time = time.time()
            ret = decoder.read(handle, frame)
            decode_time += time.time() - start_time
            if ret: #differ from cv.
                end_flag = True
            else:
                frame_list.append(frame)
            if (len(frame_list) == batch_size or end_flag) and len(frame_list):
                results = scrfd(frame_list)
                for i, frame in enumerate(frame_list):
                    det = results[i]
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                    save_path = os.path.join(output_img_dir, video_name + '_' + str(cn) + '.jpg')
                    draw_bmcv(bmcv, frame_list[i], det[:,:4], conf_scores=det[:, -2], save_path=save_path)
                frame_list.clear()
        decoder.release()
        logging.info("result saved in {}".format(output_img_dir))
        
    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = scrfd.preprocess_time / cn
    inference_time = scrfd.inference_time / cn
    postprocess_time = scrfd.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    # average_latency = decode_time + preprocess_time + inference_time + postprocess_time
    # qps = 1 / average_latency
    # logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/scrfd_fp32_1684x.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.02, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.45, help='nms threshold')
    parser.add_argument('--eval', type=bool, default=False, help='if true then gen result_txt')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')