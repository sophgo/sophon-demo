#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
import os
import time
import torch
import cv2
import argparse
import logging
import sophon.sail as sail
from sam_encoder import SamEncoder
from predictor import SamPredictor
from sam_model import Sam
from resize_func import resize

import logging
logging.basicConfig(level=logging.INFO)


def save_image_point(base_image,mask,input_point, box = False):
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not box:
        input_point = input_point[0]
        mask = mask[0][0]
        mask = mask[...,None]
        x_coord = input_point[0]
        y_coord = input_point[1]
        blue_color = np.array([255, 0, 0]) 
        green_color = (0, 255, 0)
        base_image= np.where(mask, blue_color, base_image)
        image_cv = cv2.UMat(base_image)
        base_image = cv2.drawMarker(image_cv, (x_coord, y_coord), green_color,markerType=cv2.MARKER_STAR,markerSize=50, thickness=2, line_type=cv2.LINE_AA)
        cv2.imwrite(output_dir+'/result.jpg',base_image)
    else:
        mask = mask[0][0]
        mask = mask[...,None]
        x_coord0 = input_point[0][0]
        y_coord0 = input_point[0][1]
        x_coord1 = input_point[0][2]
        y_coord1 = input_point[0][3]
        blue_color = np.array([255, 0, 0]) 
        green_color = (0, 255, 0)
        base_image= np.where(mask, blue_color, base_image)
        image_cv = cv2.UMat(base_image)
        w = x_coord1 - x_coord0
        h = y_coord1 - y_coord0
        color = (0, 255, 0) 
        cv2.rectangle(image_cv, (x_coord0, y_coord0), (x_coord0 + w, y_coord0 + h), color, 2)
        cv2.imwrite(output_dir+'/result.jpg',image_cv)       

class SAM_b(object):
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)

        self.input_shapes = [self.net.get_input_shape(self.graph_name, name) for name in self.input_names]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_shapes = [self.net.get_output_shape(self.graph_name, name) for name in self.output_names]
        logging.debug("load {} success!".format(args.bmodel))
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_names, self.input_shapes)))
        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_names, self.output_shapes)))

        self.input_shape = self.input_shapes[0]
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.orig_im_size = []
        self.image_size = 1024 #
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0


    def preprocess(self, img, sam_encoder,sam):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor = SamPredictor(sam_encoder, sam)
        predictor.set_image(img)

        # use TPU to embedding input_image
        image_embedding = predictor.get_image_embedding() 
        assert len(np.array(list(map(int, args.input_point.split(','))))) == 2 or len(np.array(list(map(int, args.input_point.split(','))))) == 4, "input coordinate length must be 2 or 4"
        # point input
        if (len(np.array(list(map(int, args.input_point.split(','))))) == 2):
            input_point = np.array([list(map(int, args.input_point.split(',')))])
            input_label = np.array([1])
            ori_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            ori_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
            ori_coord = predictor.transform.apply_coords(ori_coord, img.shape[:2]).astype(np.float32)
            ori_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            ori_has_mask_input = np.zeros(1, dtype=np.float32)
            """
            All inputs are `np.float32`.
            * `image_embeddings`: The image embedding from `predictor.get_image_embedding()`. Has a batch index of length 1.
            * `point_coords`: Coordinates of sparse input prompts, corresponding to both point inputs and box inputs. Boxes are encoded using two points, one for the top-left corner and one for the bottom-right corner. *Coordinates must already be transformed to long-side 1024.* Has a batch index of length 1.
            * `point_labels`: Labels for the sparse input prompts. 0 is a negative input point, 1 is a positive input point, 2 is a top-left box corner, and 3 is a bottom-right box corner.*
            * `mask_input`: A mask input to the model with shape 1x1x256x256. This must be supplied even if there is no mask input. In this case, it can just be zeros.
            * `has_mask_input`: An indicator for the mask input. 1 indicates a mask input, 0 indicates no mask input.
            * `orig_im_size`: The size of the input image in (H,W) format, before any transformation.
            """           
            ort_inputs = {
                "image_embeddings": image_embedding,
                "point_coords": ori_coord,
                "point_labels": ori_label,
                "mask_input": ori_mask_input,
                "has_mask_input": ori_has_mask_input,
                "orig_im_size": np.array(img.shape[:2], dtype=np.float32)
                        }
            self.orig_im_size = ort_inputs["orig_im_size"]
        # box input
        else:
            input_point = np.array(list(map(int, args.input_point.split(',')))).reshape(2, 2)
            input_label = np.array([2,3])
            ori_coord = input_point[None, :, :]
            ori_label = input_label[None, :].astype(np.float32)
            ori_coord = predictor.transform.apply_coords(ori_coord, img.shape[:2]).astype(np.float32)
            ori_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            ori_has_mask_input = np.zeros(1, dtype=np.float32)

            ort_inputs = {
                "image_embeddings": image_embedding,
                "point_coords": ori_coord,
                "point_labels": ori_label,
                "mask_input": ori_mask_input,
                "has_mask_input": ori_has_mask_input,
                "orig_im_size": np.array(img.shape[:2], dtype=np.float32)
                        }
            self.orig_im_size = ort_inputs["orig_im_size"]
        return ort_inputs

    def predict(self, input_img):
        input_data = {self.input_names[0]: input_img['image_embeddings'], self.input_names[1]: input_img['point_coords'],
                  self.input_names[2]: input_img['point_labels'], self.input_names[3]: input_img['mask_input'],
                  self.input_names[4]: input_img['has_mask_input'], self.input_names[5]: input_img['orig_im_size']}
        outputs = self.net.process(self.graph_name, input_data)
        return outputs

    def postprocess(self, outputs_0):
        '''
        4 output bmodel, resize masks on cpu
        '''
        output_name = list(outputs_0.items())[1][0]
        upscaled_masks = resize(self.image_size,torch.tensor(outputs_0[output_name]),torch.tensor(self.orig_im_size))

        return upscaled_masks > 0.0 # predictor.model.mask_threshold = 0.0
    
    def __call__(self, img, sam_encoder, sam):

        start_time = time.time()
        img = self.preprocess(img, sam_encoder, sam)
        self.preprocess_time += time.time() - start_time
        
        start_time = time.time()
        outputs_0 = self.predict(img)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        res = self.postprocess(outputs_0)
        self.postprocess_time += time.time() - start_time

        return res


def main(args):
    sam_vit_b = SAM_b(args)
    batch_size = sam_vit_b.batch_size
    sam_vit_b.init()

    # decode image
    start_time = time.time()
    src_img = cv2.imread(args.input_image)
    if src_img is None:
        logging.error("{} imread is None.".format(args.input_image))
    decode_time = time.time() - start_time

    # init sam and embedding bmodel to do preprocess 
    sam = Sam()
    sam_encoder = SamEncoder(args)

    # process images
    results = sam_vit_b(src_img, sam_encoder, sam)

    # save processed image
    input_point = np.array([list(map(int, args.input_point.split(',')))])
    if len(input_point[0]) == 2:
        save_image_point(src_img,results,input_point, box = False)
    else:
        save_image_point(src_img,results,input_point, box = True)

    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    preprocess_time = sam_vit_b.preprocess_time  
    inference_time = sam_vit_b.inference_time  
    postprocess_time = sam_vit_b.postprocess_time  
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("embedding_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("decode_mask_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

    
def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input_image', type=str, default='datasets/truck.jpg', help='path of input, must be image directory')
    parser.add_argument('--input_point', type=str, default='700,375', help='The coordinates of the input_point(point or box), point in format x,y, box in format x1,y1,x2,y2')
    parser.add_argument('--embedding_bmodel', type=str, default='models/BM1684X/embedding_bmodel/SAM-ViT-B_embedding_fp16_1b.bmodel', help='path of bmodel')
    parser.add_argument('--bmodel', type=str, default='models/BM1684X/decode_bmodel/SAM-ViT-B_decoder_fp16_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)
