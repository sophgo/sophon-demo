#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import argparse
import multiprocessing
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
import cv2
import logging
logging.basicConfig(level=logging.INFO)
# sail.set_print_flag(1)

COLORS = [
        [56, 0, 255],
        [226, 255, 0],
        [0, 94, 255],
        [0, 37, 255],
        [0, 255, 94],
        [255, 226, 0],
        [0, 18, 255],
        [255, 151, 0],
        [170, 0, 255],
        [0, 255, 56],
        [255, 0, 75],
        [0, 75, 255],
        [0, 255, 169],
        [255, 0, 207],
        [75, 255, 0],
        [207, 0, 255],
        [37, 0, 255],
        [0, 207, 255],
        [94, 0, 255],
        [0, 255, 113],
        [255, 18, 0],
        [255, 0, 56],
        [18, 0, 255],
        [0, 255, 226],
        [170, 255, 0],
        [255, 0, 245],
        [151, 255, 0],
        [132, 255, 0],
        [75, 0, 255],
        [151, 0, 255],
        [0, 151, 255],
        [132, 0, 255],
        [0, 255, 245],
        [255, 132, 0],
        [226, 0, 255],
        [255, 37, 0],
        [207, 255, 0],
        [0, 255, 207],
        [94, 255, 0],
        [0, 226, 255],
        [56, 255, 0],
        [255, 94, 0],
        [255, 113, 0],
        [0, 132, 255],
        [255, 0, 132],
        [255, 170, 0],
        [255, 0, 188],
        [113, 255, 0],
        [245, 0, 255],
        [113, 0, 255],
        [255, 188, 0],
        [0, 113, 255],
        [255, 0, 0],
        [0, 56, 255],
        [255, 0, 113],
        [0, 255, 188],
        [255, 0, 94],
        [255, 0, 18],
        [18, 255, 0],
        [0, 255, 132],
        [0, 188, 255],
        [0, 245, 255],
        [0, 169, 255],
        [37, 255, 0],
        [255, 0, 151],
        [188, 0, 255],
        [0, 255, 37],
        [0, 255, 0],
        [255, 0, 170],
        [255, 0, 37],
        [255, 75, 0],
        [0, 0, 255],
        [255, 207, 0],
        [255, 0, 226],
        [255, 245, 0],
        [188, 255, 0],
        [0, 255, 18],
        [0, 255, 75],
        [0, 255, 151],
        [255, 56, 0],
        [245, 255, 0],
    ]

class YOLOv5:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]

        # core_id
        self.core_id = None
        
        # get input
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype= self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}
        
        # get output
        self.output_names = self.net.get_output_names(self.graph_name)
        if len(self.output_names) not in [1, 3]:
            raise ValueError('only suport 1 or 3 outputs, but got {} outputs bmodel'.format(len(self.output_names)))
        
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
        
        # check batch size 
        self.batch_size = self.input_shape[0]
        support_batch_size = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
        if self.batch_size not in support_batch_size:
            raise ValueError('batch_size must be {} for bmcv, but got {}'.format(support_batch_size, self.batch_size))
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        # init preprocess
        self.use_resize_padding = True
        self.use_vpp = False
        self.ab = [x * self.input_scale / 255.  for x in [1, 0, 1, 0, 1, 0]]
        
        # init postprocess
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        
        self.agnostic = False
        self.multi_label = True
        self.max_det = 1000
        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )

    def set_core_id(self, core_id):
        self.core_id = core_id

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
        # if self.use_resize_padding:
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
        return resized_img_rgb, ratio, txy
    
    def predict(self, input_tensor, img_num):
        """
        ensure output order: loc_data, conf_preds, mask_data, proto_data
        Args:
            input_tensor:
        Returns:
        """
        input_tensors = {self.input_name: input_tensor} 
        self.net.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors, [self.core_id])
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
            preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(bmimg_list[0])
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
                preprocessed_bmimg, ratio, txy  = self.preprocess_bmcv(bmimg_list[i])
                ratio_list.append(ratio)
                txy_list.append(txy)
                bmimgs[i] = preprocessed_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)
            
        outputs = self.predict(input_tensor, img_num)
        
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)

        return results

def process_video_channel(args, chan_idx, class_names):
    # initialize net
    yolov5 = YOLOv5(args)

    # set launch core
    yolov5.set_core_id(chan_idx % 2)

    batch_size = yolov5.batch_size
    
    handle = sail.Handle(args.dev_id)
    bmcv = sail.Bmcv(handle)

    decoder = sail.Decoder(args.input, True, args.dev_id)
    if not decoder.is_opened():
        raise Exception("can not open the video")
    frame_count = 0
    frame_name_list = []
    frame_list = []
    end_flag = False
    while not end_flag:
        frame = sail.BMImage()
        ret = decoder.read(handle, frame)
        if ret: #differ from cv.
            end_flag = True
        else:
            frame_save_path = os.path.join(args.result_dir, f'images_chan_{chan_idx}', f'chan{chan_idx}_frame{frame_count}.jpg')
            print(f"Chan:{chan_idx}; Frame_id:{frame_count}\n")
            frame_count = frame_count + 1
            frame_list.append(frame)
            frame_name_list.append(frame_save_path)
        if (len(frame_list) == batch_size or end_flag) and len(frame_list):
            results = yolov5(frame_list)
            for i, frame in enumerate(frame_list):
                det = results[i]
                if det.shape[0] >= 1:
                    draw_image(bmcv, frame_list[i], det[:,:4], class_names, classes_ids=det[:, -1], conf_scores=det[:, -2], save_path=frame_name_list[i])
                else:
                    bmcv.imwrite(frame_name_list[i], frame_list[i])
            frame_name_list.clear()
            frame_list.clear()
    decoder.release()
    logging.info("result saved in {}".format(os.path.join(args.result_dir, f'images_chan_{chan_idx}')))

def draw_image(bmcv, bmimg, boxes, class_names, classes_ids=None, conf_scores=None, save_path=""):
    img_bgr_planar = bmcv.convert_format(bmimg)
    image = bmcv.bm_image_to_tensor(img_bgr_planar).asnumpy()[0]
    image = np.transpose(image, (1,2,0)).copy()
    thickness = 2

    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        logging.debug("class id={}, score={}, (x1={},y1={},w={},h={})".format(int(classes_ids[idx]), conf_scores[idx], x1, y1, x2-x1, y2-y1))

        if conf_scores[idx] < 0.25:
            continue

        if classes_ids is not None:
            color = np.array(COLORS[int(classes_ids[idx]) + 1]).astype(np.uint8).tolist()
        else:
            color = (0, 0, 255)

        if (x2 - x1) <= thickness * 2 or (y2 - y1) <= thickness * 2:
            logging.info("width or height too small, this rect will not be drawed: (x1={},y1={},w={},h={})".format(x1, y1, x2-x1, y2-y1))
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=thickness)
            classes_ids = classes_ids.astype(np.int8)
            cv2.putText(image, class_names[classes_ids[idx]] + ':' + str(round(conf_scores[idx], 2)),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
            cv2.imwrite(save_path, image)

def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    if not os.path.exists(args.classnames):
        raise FileNotFoundError('{} is not existed.'.format(args.classnames))
    
    # creat save path
    output_dir = args.result_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for idx in range(args.chan_num):
        output_img_dir = os.path.join(output_dir, f'images_chan_{idx}')
        if not os.path.exists(output_img_dir):
            os.mkdir(output_img_dir)

    with open(args.classnames, 'r') as file:
        lines = file.read().splitlines()
    classes_names = tuple(lines)

    processes = []

    for idx in range(args.chan_num):
        p = multiprocessing.Process(target=process_video_channel, args=(args, idx, classes_names))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str, default='../models/BM1688/yolov5s_v6.1_3output_int8_4b.bmodel', help='bmodel file path')
    parser.add_argument('--dev_id', type=int, default=0, help='TPU device id')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='confidence threshold for filter boxes')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    parser.add_argument('--input', type=str, default='../datasets/test_car_person_1080P.mp4', help='input video file path')
    parser.add_argument('--chan_num', type=int, default=2, help='copy the input video into chan_num copies')
    parser.add_argument('--classnames', type=str, default='../datasets/coco.names', help='class names file path')
    parser.add_argument('--result_dir', type=str, default='./results', help='save path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
