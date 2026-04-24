#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import logging
import os
import time
import json
import numpy as np
import sophon.sail as sail

from utils import COCO_CLASSES, COLORS

logging.basicConfig(level=logging.INFO)

class DFineModel:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]

        self.input_name = self.net.get_input_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name[0])
        self.input_dtype= self.net.get_input_dtype(self.graph_name, self.input_name[0])
        self.orig_size_shape = self.net.get_input_shape(self.graph_name, self.input_name[1])
        self.orig_size_dtype = self.net.get_input_dtype(self.graph_name, self.input_name[1])
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

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

        # init preprocess
        self.use_resize_padding = True
        self.use_vpp = False
        self.ab = [x / 255.  for x in [1, 0, 1, 0, 1, 0]]

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
        img_w = bmimg.width()
        img_h = bmimg.height()
        if self.use_resize_padding:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            r = min(r_w, r_h)
            tw = int(round(r * img_w))
            th = int(round(r * img_h))
            tx1, ty1 = self.net_w - tw, self.net_h - th  # wh padding

            tx1 /= 2  # divide padding into 2 sides
            ty1 /= 2

            ratio = (r, r)
            txy = (tx1, ty1)
            attr = sail.PaddingAtrr()
            attr.set_stx(int(round(tx1 - 0.1)))
            attr.set_sty(int(round(ty1 - 0.1)))
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(0)
            attr.set_g(0)
            attr.set_b(0)
            
            preprocess_fn = self.bmcv.vpp_crop_and_resize_padding if self.use_vpp else self.bmcv.crop_and_resize_padding
            resized_img_rgb = preprocess_fn(bmimg, 0, 0, img_w, img_h, self.net_w, self.net_h, attr, sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)
        else:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            ratio = (r_w, r_h)
            txy = (0, 0)
            preprocess_fn = self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
            resized_img_rgb = preprocess_fn(bmimg, self.net_w, self.net_h)
        return resized_img_rgb, ratio, txy

    def predict(self, input_tensor, orig_size_tensor, img_num):
        input_tensors = {self.net.get_input_names(self.graph_name)[0]: input_tensor, self.net.get_input_names(self.graph_name)[1]: orig_size_tensor} 
        self.net.process(self.graph_name, input_tensors, self.output_tensors)
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
        labels, boxes, scores = out
        return labels, boxes, scores
    
    def postprocess(self, labels, boxes, scores, ratios, paddings, thrh=0.4):
        scr_mask = scores > thrh
        filtered_labels = labels[scr_mask]
        filtered_scores = scores[scr_mask]
        filtered_boxes = boxes[scr_mask]

        ratio_val = float(ratios[0])
        pad_w, pad_h = paddings[0]

        mapped_boxes = []
        for bb in filtered_boxes:
            x1 = int((float(bb[0]) - pad_w) / ratio_val)
            y1 = int((float(bb[1]) - pad_h) / ratio_val)
            x2 = int((float(bb[2]) - pad_w) / ratio_val)
            y2 = int((float(bb[3]) - pad_h) / ratio_val)
            mapped_boxes.append([x1, y1, x2, y2])
        mapped_boxes = np.array(mapped_boxes)

        return filtered_labels, mapped_boxes, filtered_scores

    def __call__(self, bmimg_list):
        img_num = len(bmimg_list)
        ori_size_list = []
        ratio_list = []
        txy_list = []
        if self.batch_size == 1:
            ori_h, ori_w =  bmimg_list[0].height(), bmimg_list[0].width()
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()
            preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(bmimg_list[0])
            self.preprocess_time += time.time() - start_time
            orig_size = np.array([[preprocessed_bmimg.height(), preprocessed_bmimg.width()]]).astype(np.int32)
            ratio_list.append(ratio[0])
            txy_list.append(txy)
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimg, input_tensor)   
        else:
            BMImageArray = getattr(sail, 'BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                ori_h, ori_w =  bmimg_list[i].height(), bmimg_list[i].width()
                ori_size_list.append((ori_w, ori_h))
                start_time = time.time()
                preprocessed_bmimg, ratio, txy  = self.preprocess_bmcv(bmimg_list[i])
                orig_size = np.array([[preprocessed_bmimg.height(), preprocessed_bmimg.width()]]).astype(np.int32)
                self.preprocess_time += time.time() - start_time
                ratio_list.append(ratio[0])
                txy_list.append(txy)
                bmimgs[i] = preprocessed_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)
        orig_size_tensor = sail.Tensor(self.handle, self.orig_size_shape, self.orig_size_dtype, False, True)
        orig_size_tensor.update_data(orig_size)
        # orig_size_tensor = sail.Tensor(self.handle, orig_size, False, True)
        start_time = time.time()
        labels, boxes, scores = self.predict(input_tensor, orig_size_tensor, img_num)
        self.inference_time += time.time() - start_time
        
        start_time = time.time()
        filtered_labels, mapped_boxes, filtered_scores = self.postprocess(labels, boxes, scores, ratio_list, txy_list, thrh=0.4)
        self.postprocess_time += time.time() - start_time

        return filtered_labels, mapped_boxes, filtered_scores

def draw(bmimg, labels, boxes, scores, dev_id, output_img_dir=None, file_name=None, isvideo=False, cn=None):
    bmcv = sail.Bmcv(sail.Handle(dev_id))
    thickness = 2
    img_bgr_planar = bmcv.convert_format(bmimg)

    for idx, (lbl, bb) in enumerate(zip(labels, boxes)):
        x1, y1, x2, y2 = bb
        width = x2 - x1
        height = y2 - y1

        if hasattr(lbl, '__int__'):
            color = np.array(COLORS[int(lbl) + 1]).astype(np.uint8).tolist()
        else:
            color = (0, 0, 255)

        if width <= thickness * 2 or height <= thickness * 2:
            logging.info(f"width or height too small, skip drawing rect: (x1={x1}, y1={y1}, w={width}, h={height})")
            continue

        bmcv.rectangle(img_bgr_planar, x1, y1, width, height, color, thickness)
        logging.debug(f"class id={int(lbl)}, score={scores[idx]}, (x1={x1}, y1={y1}, w={width}, h={height})")

    if isvideo:
        logging.info("{}, det nums: {}".format(cn, boxes.shape[0]))
        return img_bgr_planar
    else:
        if output_img_dir is not None and file_name is not None:
            bmcv.imwrite(os.path.join(output_img_dir, file_name), img_bgr_planar)


def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 

    D_FINE = DFineModel(args)
    batch_size = D_FINE.batch_size
    D_FINE.init()
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
                cn += 1
                logging.info("{}, img_file: {}".format(cn, img_file))
                # decode
                start_time = time.time()

                decoder = sail.Decoder(img_file, True, args.dev_id)
                bmimg = sail.BMImage()
                handle = sail.Handle(args.dev_id)
                ret = decoder.read(handle, bmimg)

                if ret != 0:
                    logging.error("{} decode failure.".format(img_file))
                    continue
                decode_time += time.time() - start_time
                bmimg_list.append(bmimg)
                filename_list.append(filename)
                if (len(bmimg_list) == batch_size or cn == len(filenames)) and len(bmimg_list):
                    # predict
                    filtered_labels, mapped_boxes, filtered_scores = D_FINE(bmimg_list)
                    
                    for i, filename in enumerate(filename_list):
                        # save image
                        draw(bmimg_list[i], filtered_labels, mapped_boxes, filtered_scores, args.dev_id, output_img_dir, filename) 
                        # save result
                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['bboxes'] = []
                        for idx in range(len(filtered_labels)):
                            bbox_dict = dict()
                            x1, y1, x2, y2 = mapped_boxes[idx]
                            category_id = filtered_labels[idx]
                            score = filtered_scores[idx]
                            bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
                            bbox_dict['category_id'] = int(category_id)
                            bbox_dict['score'] = float(round(score,5))
                            res_dict['bboxes'].append(bbox_dict)
                        results_list.append(res_dict)

                    bmimg_list.clear()
                    filename_list.clear()

        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_bmcv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            # json.dump(results_list, jf)
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

    # test videos
    else:
        decoder = sail.Decoder(args.input, True, args.dev_id)
        if not decoder.is_opened():
            raise Exception("can not open the video")
        enc_params = f"width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25"
        encoder = sail.Encoder("results/output.mp4", args.dev_id, 'h264_bm', 'NV12', enc_params, 10)
        
        video_name = os.path.splitext(os.path.split(args.input)[1])[0]
        cn = 0
        frame_list = []
        yolov8_handle = sail.Handle(args.dev_id)
        end_flag = False
        while not end_flag:
            frame = sail.BMImage()
            start_time = time.time()
            ret = decoder.read(yolov8_handle, frame)
            decode_time += time.time() - start_time
            if ret: #differ from cv.
                end_flag = True
            else:
                frame_list.append(frame)
            if (len(frame_list) == batch_size or end_flag) and len(frame_list):
                filtered_labels, mapped_boxes, filtered_scores = D_FINE(frame_list)
                for i, frame in enumerate(frame_list):
                    cn += 1  
                    drawed_bmimg = draw(frame_list[i], filtered_labels, mapped_boxes, filtered_scores, args.dev_id, isvideo=True, cn=cn)
                    encoder.video_write(drawed_bmimg)
                frame_list.clear()
        decoder.release()
        encoder.release()
        logging.info("result saved in {}".format(output_img_dir))

    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = D_FINE.preprocess_time / cn
    inference_time = D_FINE.inference_time / cn
    postprocess_time = D_FINE.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bmodel", type=str, required=True, help="Path to the model file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image or video file.")
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    main(args)
