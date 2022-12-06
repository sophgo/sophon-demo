#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.    All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# -*- coding: utf-8 -*- 
import time
import os
import numpy as np
import argparse
import json
import sophon.sail as sail
import math
class PreProcessor:
    """ Preprocessing class.
    """
    def __init__(self, bmcv, scale):
        """ Constructor.
        """
        self.bmcv = bmcv
        # self.ab = [x * scale for x in [1, -123, 1, -117, 1, -104]]
        self.ab = [x for x in [1, -104, 1, -117, 1, -123]]
        self.scale = scale
    def process(self, input, tmp, output):
        """ Execution function of preprocessing.
        Args:
            input: sail.BMImage, input image
            output: sail.BMImage, output data

        Returns:
            None
        """
        resized = self.bmcv.vpp_resize(input, 300, 300)
        # self.bmcv.convert_to(resized, tmp, ((self.scale, 0), (self.scale, 0), (self.scale, 0)))
        self.bmcv.convert_to(resized, output, ((self.ab[0], self.ab[1]), \
                                            (self.ab[2], self.ab[3]), \
                                            (self.ab[4], self.ab[5])))

class PostProcessor:
    """ Postprocessing class.
    """
    def __init__(self, conf_thre):
        """ Constructor.
        """
        self.conf_thre = conf_thre

    def process(self, data):
        """ Execution function of postprocessing.
        Args:
            data: Inference output
            img_w: Image width
            img_h: Image height
            ratio: aspect ratio
            txy: padding offsets
        Returns:
            Detected boxes.
        """
        data = data.reshape((data.shape[2], data.shape[3])) #[200, 7]
        ret = []
        for proposal in data:
            if proposal[2] < self.conf_thre:
                continue
            ret.append([
                    int(proposal[0]),       # image id
                    int(proposal[1]),       # class id
                    proposal[2],            # score
                    float(proposal[3]),     # x0
                    float(proposal[4]),     # x1
                    float(proposal[5]),     # y0
                    float(proposal[6])])    # y1
        return ret

def inference(bmodel_path, input_path, conf_thre, tpu_id, results_path):
    """ Load a bmodel and do inference.
    Args:
     bmodel_path: Path to bmodel
     input_path: Path to input directory
     tpu_id: ID of TPU to use
     results_path: Path of result file

    Returns:
        True for success and False for failure
    """
    # init Engine
    engine = sail.Engine(tpu_id)
    # load bmodel without built in input and output tensors
    engine.load(bmodel_path)
    # get model info
    # only one model loaded for this engine
    # only one input tensor and only one output tensor in this graph
    graph_name = engine.get_graph_names()[0]
    input_name = engine.get_input_names(graph_name)[0]
    output_name = engine.get_output_names(graph_name)[0]
    input_shape = engine.get_input_shape(graph_name, input_name)
    batch_size = input_shape[0]
    print(input_shape)
    input_shapes = {input_name: input_shape}
    output_shape = engine.get_output_shape(graph_name, output_name)
    print(output_shape)
    input_dtype= engine.get_input_dtype(graph_name, input_name)
    output_dtype = engine.get_output_dtype(graph_name, output_name)
    # get handle to create input and output tensors
    handle = engine.get_handle()
    input = sail.Tensor(handle, input_shape,    input_dtype,    False, False)
    output = sail.Tensor(handle, output_shape, output_dtype, True,    True)
    input_tensors = {input_name: input}
    output_tensors = {output_name: output}
    # set io_mode
    engine.set_io_mode(graph_name, sail.IOMode.SYSO)
    # init bmcv for preprocess
    bmcv = sail.Bmcv(handle)
    img_dtype = bmcv.get_bm_image_data_format(input_dtype)
    # init preprocessor and postprocessor
    scale = 0.00392157
    preprocessor = PreProcessor(bmcv, scale)
    postprocessor = PostProcessor(conf_thre)
    if os.path.isdir(input_path):
        input_directory = os.listdir(input_path)
        input_directory.sort(key = lambda x: int(x[:-4]))
        loop_count = len(input_directory)
    else: 
        loop_count = 1000000
        decoder = sail.Decoder(input_path)
    frame_count = 0
    if batch_size == 4:
        BMImageArray = eval('sail.BMImageArray{}D'.format(input_shape[0]))
        bmimage_array = BMImageArray()
    bmimg_list = []
    filename_list = []
    results = []
    time_infer_total = 0
    time_all_start = time.time()
    # pipeline of inference
    for i in range(0, loop_count):   
        if os.path.isdir(input_path):
            print("read image: ", input_directory[i])
            # init decoder
            decoder = sail.Decoder(os.path.join(input_path, input_directory[i]), True, tpu_id)
            filename_list.append(input_directory[i])
        else:
            print("read video frame ", i)
            filename_list.append(str(i) + ".jpg")
        frame_count += 1
        # read an image from a image file or a video file
        img0 = sail.BMImage()
        ret = decoder.read(handle, img0)
        if ret != 0:
            print("Path not valid or end of video!")
            break

        tmp = sail.BMImage(handle, input_shape[2], input_shape[3], \
                                                sail.Format.FORMAT_BGR_PLANAR, img_dtype)
        # preprocess
        img1 = sail.BMImage(handle, input_shape[2], input_shape[3], \
                                                sail.Format.FORMAT_BGR_PLANAR, img_dtype)
        preprocessor.process(img0, tmp, img1)
        if batch_size == 4:
            bmimg_list.append(img0)
            bmimage_array[i % 4] = img1.data()
            batch_id = (i + 1) % input_shape[0]
            if (batch_id == 0 ) or i + 1 == loop_count:
                bmcv.bm_image_to_tensor(bmimage_array, input)
                # inference
                t1 = time.time()
                engine.process(graph_name, input_tensors, input_shapes, output_tensors)
                t2 = time.time()
                time_infer_total += t2 - t1
                # postprocess
                real_output_shape = engine.get_output_shape(graph_name, output_name)
                out = output.asnumpy(real_output_shape)
                dets = postprocessor.process(out)
                # result
                for (image_idx, class_id, score, x0, y0, x1, y1) in dets:
                    if batch_id != 0 and image_idx >= batch_id:
                        continue
                    x0 *= bmimg_list[image_idx].width()
                    y0 *= bmimg_list[image_idx].height()
                    x1 *= bmimg_list[image_idx].width()
                    y1 *= bmimg_list[image_idx].height()
                    rIndex = filename_list[image_idx].index('.')
                    bmcv.rectangle(bmimg_list[image_idx], int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1), (255, 0, 0), 3)        
                    preds = dict()
                    preds['image_id'] = int(filename_list[image_idx][0:rIndex])
                    preds['bbox'] = [int(x0), int(y0), round(x1 - x0), round(y1 - y0)]
                    preds['score'] = float(score)
                    preds['category_id'] = class_id
                    results.append(preds)
                for id in range(0, len(bmimg_list)):
                    bmcv.imwrite(results_path + filename_list[id], bmimg_list[id])
                filename_list = []
                bmimg_list = []
        elif batch_size == 1:
            bmcv.bm_image_to_tensor(img1, input)
            t1 = time.time()
            engine.process(graph_name, input_tensors, input_shapes, output_tensors)
            t2 = time.time()
            time_infer_total += t2 - t1            
            # postprocess
            real_output_shape = engine.get_output_shape(graph_name, output_name)
            out = output.asnumpy(real_output_shape)
            dets = postprocessor.process(out)
            # result
            for (image_idx, class_id, score, x0, y0, x1, y1) in dets:
                x0 *= img0.width()
                y0 *= img0.height()
                x1 *= img0.width()
                y1 *= img0.height()
                if os.path.isdir(input_path):
                    preds = dict()
                    rIndex = input_directory[i].index('.')
                    preds['image_id'] = int(input_directory[i][0:rIndex])
                    preds['bbox'] = [int(x0), int(y0), round(x1 - x0), round(y1 - y0)]
                    preds['score'] = float(score)
                    preds['category_id'] = class_id
                    results.append(preds)
                bmcv.rectangle(img0, int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1), (255, 0, 0), 3)        
            if os.path.isdir(input_path):
                bmcv.imwrite(results_path + input_directory[i], img0)
            else:    
                bmcv.imwrite(results_path + str(i) + '.jpg', img0)

    #end of for loop
    if os.path.isdir(input_path):
        json_name = "results_bmcv.json"
        with open(json_name, 'w') as jf:
            json.dump(results, jf, indent=4, ensure_ascii=False)

    time_all_end = time.time()
    total_time = time_all_end - time_all_start
    print("total_time(ms): {:.2f}, img_num: {}".format(total_time * 1000, frame_count))
    avg_infer_time = time_infer_total / frame_count
    print("avg_infer_time(ms): {:.2f}".format(batch_size * avg_infer_time * 1000))
    
if __name__ == '__main__':
    """ A SSD example using bm-ffmpeg to decode and bmcv to preprocess.
    """
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--bmodel', default='', required=True)
    PARSER.add_argument('--input_path', default='', required=True)
    PARSER.add_argument('--conf_thre', default=0.01, type=float, required=False)
    PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
    PARSER.add_argument('--results_directory', default='results/', required=False)
    
    ARGS = PARSER.parse_args()
    if not (os.path.isdir(ARGS.input_path) or os.path.exists(ARGS.input_path)):
        raise Exception('{} is not a valid input.'.format(ARGS.input_path))
    if not os.path.exists(ARGS.results_directory):
        os.mkdir(ARGS.results_directory)

    inference(ARGS.bmodel, ARGS.input_path, ARGS.conf_thre, ARGS.tpu_id, ARGS.results_directory)
