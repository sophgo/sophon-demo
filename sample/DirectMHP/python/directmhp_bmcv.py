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
import json
import argparse
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
#from utils import COCO_CLASSES, COLORS
import logging
from math import cos, sin
import math
logging.basicConfig(level=logging.INFO)

class DirectMHP:
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
        # 为不同output_names创建不同尺寸的tensor
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
        self.use_resize_padding = True
        self.use_vpp = False
        self.ab = [x * self.input_scale / 255.  for x in [1, 0, 1, 0, 1, 0]]
        
        # init postprocess
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.agnostic = False
        self.multi_label = False
        self.max_det = 300

        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )
        
        # init time
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
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)
            
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
        #for name in self.output_names:
            # outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num] * self.output_scales[name]
            #outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num]
        # resort
        outputs_dict[self.output_names[0]] = self.output_tensors[self.output_names[0]].asnumpy()[:img_num]
        out = [outputs_dict[self.output_names[0]][:img_num]]
        return out

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
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results
        
def draw_bmcv(dev_id, image, boxes, pitchs_yaws_rolls, output_img_dir, file_name, cn, conf_scores=None, isvideo=False):
    bmcv = sail.Bmcv(sail.Handle(dev_id))
    img_bgr_planar = bmcv.vpp_convert_format(image, sail.FORMAT_YUV444P)
    thickness = 2
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        if conf_scores[idx] < 0.25:
            continue
        
        color = (255,255,255)
        if (x2 - x1) <= thickness * 2 or (y2 - y1) <= thickness * 2:
            logging.info("width or height too small, this rect will not be drawed: (x1={},y1={},w={},h={})".format(x1, y1, x2-x1, y2-y1))
        else:
            bmcv.rectangle(img_bgr_planar, x1, y1, (x2 - x1), (y2 - y1), color, thickness)
        # bmcv.putText(image, COCO_CLASSES[int(classes_ids[idx] + 1)], x1, y1, tuple(color),1.0,1)
        logging.debug("score={}, (x1={},y1={},w={},h={})".format(conf_scores[idx], x1, y1, x2-x1, y2-y1))
        pitch = (pitchs_yaws_rolls[idx][0] - 0.5) * 180
        yaw = (pitchs_yaws_rolls[idx][1] - 0.5) * 360
        roll = (pitchs_yaws_rolls[idx][2] - 0.5) * 180
       
        image = plot_3axis_Zaxis(dev_id, img_bgr_planar, yaw, pitch, roll, tdx=(x1+x2)/2, tdy=(y1+y2)/2,
            size=max(y2-y1, x2-x1)*0.8, thickness=2)
    if isvideo:
        bmcv.imwrite(os.path.join(output_img_dir, file_name + '_' + str(cn) + '.jpg'), img_bgr_planar)
    else:
        bmcv.imwrite(os.path.join(output_img_dir, file_name), image)

def plot_3axis_Zaxis(dev_id, img, yaw, pitch, roll, tdx=None, tdy=None, size=50., limited=True, thickness=2):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]
    bmcv = sail.Bmcv(sail.Handle(dev_id))
  
    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
  

    if tdx != None and tdy != None:
        face_x = tdx
        face_y = tdy
    else:
        height, width = img.shape[:2]
        face_x = width / 2
        face_y = height / 2
 
    # X-Axis (pointing to right) drawn in red
    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
     
    # Y-Axis (pointing to down) drawn in green
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
     
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y
   
    # Plot head oritation line in black
    # scale_ratio = 5
    scale_ratio = 2
    base_len = math.sqrt((face_x - x3)**2 + (face_y - y3)**2)
    if face_x == x3:
        endx = tdx
        if face_y < y3:
            if limited:
                endy = tdy + (y3 - face_y) * scale_ratio
            else:
                endy = img.shape[0]
        else:
            if limited:
                endy = tdy - (face_y - y3) * scale_ratio
            else:
                endy = 0
    elif face_x > x3:
        if limited:
            endx = tdx - (face_x - x3) * scale_ratio
            endy = tdy - (face_y - y3) * scale_ratio
        else:
            endx = 0
            endy = tdy - (face_y - y3) / (face_x - x3) * tdx
    else:
        if limited:
            endx = tdx + (x3 - face_x) * scale_ratio
            endy = tdy + (y3 - face_y) * scale_ratio
        else:
            endx = img.shape[1]
            endy = tdy - (face_y - y3) / (face_x - x3) * (tdx - endx)
    
    endx = max(0, min(endx, img.width() - 1))
    endy = max(0, min(endy, img.height() - 1))
    
    a = bmcv.polylines(img, [[(int(tdx), int(tdy)), (int(endx), int(endy))]], True,[0, 255, 255])


    # X-Axis pointing to right. drawn in red
    x1 = max(0, min(x1, img.width() - 1))
    y1 = max(0, min(y1, img.height() - 1))
    bmcv.polylines(img, [[(int(face_x), int(face_y)), (int(x1),int(y1))]], False,[0,0,255],thickness)
 
    
    # Y-Axis pointing to down. drawn in green   
    x2 = max(0, min(x2, img.width() - 1))
    y2 = max(0, min(y2, img.height() - 1))
    bmcv.polylines(img, [[(int(face_x), int(face_y)), (int(x2),int(y2))]], False,[0,255,0],thickness)
  
    
    # Z-Axis (out of the screen) drawn in blue
    x3 = max(0, min(x3, img.width() - 1))
    y3 = max(0, min(y3, img.height() - 1))
    bmcv.polylines(img, [[(int(face_x), int(face_y)), (int(x3),int(y3))]], False,[255,0,0],thickness)

 
 
    return img

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

    directMHP = DirectMHP(args)
    batch_size = directMHP.batch_size
    handle = sail.Handle(args.dev_id)
    bmcv = sail.Bmcv(handle)
    

    directMHP.init()

    decode_time = 0.0
    # test images
    if os.path.isdir(args.input): 
        bmimg_list = []
        filename_list = []
        results_list = []
        cn = 0
        directMHP_handle = sail.Handle(args.dev_id)
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
                ret = decoder.read(directMHP_handle, bmimg)
                if ret != 0:
                    logging.error("{} decode failure.".format(img_file))
                    continue
                decode_time += time.time() - start_time
                bmimg_list.append(bmimg)
                filename_list.append(filename)
                if (len(bmimg_list) == batch_size or cn == len(filenames)) and len(bmimg_list):
                    # predict
                    results = directMHP(bmimg_list)
                    save_path = os.path.join(output_img_dir, filename)
                    for i, filename in enumerate(filename_list):
                        det = results[i]
                        if det.shape[0] >= 1:
                            draw_bmcv(args.dev_id,
                                  bmimg_list[i],
                                  det[:,:4],
                                  det[:,6:],
                                  output_img_dir,
                                  filename,
                                  cn, 
                                  conf_scores=det[:,-5])
                        else:
                            bmcv.imwrite(save_path, bmimg_list[i])
                        # save result
                        #res_dict = []
                        for idx in range(det.shape[0]):
                            bbox_dict = dict()
                            x1, y1, x2, y2, score, category_id, pitch, yaw, roll  = det[idx].tolist()
                            bbox_dict['image_name'] = filename
                            bbox_dict['category_id'] = int(category_id)
                            bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
                            bbox_dict['score'] = float(round(score,5))
                            bbox_dict['pitch'] = float(round((pitch - 0.5)*180, 3))
                            bbox_dict['yaw'] = float(round((yaw - 0.5)*360, 3))
                            bbox_dict['roll'] = float(round((roll - 0.5)*180, 3))
                            #res_dict.append(bbox_dict)
                            results_list.append(bbox_dict)
                        
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
        video_name = os.path.splitext(os.path.split(args.input)[1])[0]
        cn = 0
        frame_list = []
        directMHP_handle = sail.Handle(args.dev_id)
        while True:
            frame = sail.BMImage()
            start_time = time.time()
            ret = decoder.read(directMHP_handle, frame)
            if ret:
                break
            decode_time += time.time() - start_time
            frame_list.append(frame)
            if (len(frame_list) == batch_size or cn == len( frame_list)) and len(frame_list):
                results = directMHP(frame_list)
                for i, frame in enumerate(frame_list):
                    det = results[i]
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                    save_path = os.path.join(output_img_dir, video_name + '_' + str(cn) + '.jpg')
                    if det.shape[0] >= 1:
                        draw_bmcv(args.dev_id,
                              frame_list[i],
                              det[:,:4],
                              det[:,6:],
                              output_img_dir,
                              video_name,
                              cn, 
                              conf_scores=det[:,-5],
                              isvideo=True)
                    else:
                         bmcv.imwrite(save_path, frame_list[i])

                frame_list.clear()

        decoder.release()
        logging.info("result saved in {}".format(output_img_dir))


    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = directMHP.preprocess_time / cn
    inference_time = directMHP.inference_time / cn
    postprocess_time = directMHP.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/directmhp_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='nms threshold')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')


