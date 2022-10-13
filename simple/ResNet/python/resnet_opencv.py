# -*- coding: utf-8 -*- 
import os
import time
import cv2
import numpy as np
import argparse
import glob
import sophon.sail as sail
import logging
logging.basicConfig(level=logging.DEBUG)


class Resnet(object):
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.tpu_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.input_shapes = [self.net.get_input_shape(self.graph_name, name) for name in self.input_names]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_shapes = [self.net.get_output_shape(self.graph_name, name) for name in self.output_names]
        logging.debug("load {} success!".format(args.bmodel))
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_names, self.input_shapes)))
        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_names, self.output_shapes)))
        self.input_name = self.input_names[0]
        self.input_shape = self.input_shapes[0]

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

        self.dt = 0.0

        self.handle = self.net.get_handle()
        self.bmcv = sail.Bmcv(self.handle)
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_names[0])
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        print('img_dtype', self.img_dtype)
        print('input_dtype, output_dtype', self.input_dtype, self.output_dtype) 

    def preprocess(self, img):
        h, w, _ = img.shape
        if h != self.net_h or w != self.net_w:
            img = cv2.resize(img, (self.net_w, self.net_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img = (img/255-self.mean)/self.std
        img = np.transpose(img, (2, 0, 1))
        return img

    def predict(self, input_img):
        input_data = {self.input_name: input_img}
        t0 = time.time()
        outputs = self.net.process(self.graph_name, input_data)
        self.dt += time.time() - t0
        return list(outputs.values())[0]

    def postprocess(self, outputs):
        res = list()
        outputs_exp = np.exp(outputs)
        outputs = outputs_exp / np.sum(outputs_exp, axis=1)[:,None]
        predictions = np.argmax(outputs, axis = 1)
        for pred, output in zip(predictions, outputs):
            score = output[pred]
            res.append((pred,score))
        return res

    def __call__(self, img_list):
        img_num = len(img_list)
        img_input_list = []
        for img in img_list:
            img = self.preprocess(img)
            img_input_list.append(img)
        
        if img_num == self.batch_size:
            input_img = np.stack(img_input_list)
            outputs = self.predict(input_img)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(img_input_list)
            outputs = self.predict(input_img)[:img_num]

        res = self.postprocess(outputs)

        return res

    def get_time(self):
        return self.dt

def main(args):
    resnet = Resnet(args)
    batch_size = resnet.batch_size

    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    if not os.path.isdir(args.input_path):
        # logging.error("input_path must be an image directory.")
        # return 0
        raise Exception('{} is not a directory.'.format(args.input_path))
        

    img_list = []
    filename_list = []
    res_dict = {}
    t1 = time.time()
    for filename in glob.glob(args.input_path+'/*'):
        if os.path.splitext(filename)[-1] not in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            continue
        src_img = cv2.imread(filename)
        if src_img is None:
            logging.error("{} imread is None.".format(filename))
            continue
        img_list.append(src_img)
        filename_list.append(filename)
        if len(img_list) == batch_size:
            res_list = resnet(img_list)
            for i, filename in enumerate(filename_list):
                logging.info("filename: {}, res: {}".format(filename, res_list[i]))
                res_dict[filename] = res_list[i]
            img_list = []
            filename_list = []
    if len(img_list):
        res_list = resnet(img_list)
        for i, filename in enumerate(filename_list):
            logging.info("filename: {}, res: {}".format(filename, res_list[i]))
            res_dict[filename] = res_list[i]

    t2 = time.time()

    # save result
    result_file = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input_path)[-1] + "_opencv" + "_python_result.txt"
    fout = open(os.path.join(output_dir, result_file), 'w')
    for filename, (prediction, score) in res_dict.items():
        fout.write('\t'.join([filename, str(prediction), str(score)])+'\n')
    fout.close()

    logging.info("result saved in {}".format(os.path.join(output_dir, result_file)))
	    
    # calculate speed  
    cn = len(res_dict)    
    logging.info("------------------ Inference Time Info ----------------------")
    inference_time = resnet.get_time() / cn
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    total_time = t2 - t1
    logging.info("total_time(ms): {:.2f}, img_num: {}".format(total_time * 1000, cn))
    average_latency = total_time / cn
    qps = 1 / average_latency
    logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))
        
def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input_path', type=str, default='../data/images/imagenet_val_1k/img', help='path of input, must be image directory')
    parser.add_argument('--bmodel', type=str, default='../data/models/BM1684X/resnet_fp32_b1.bmodel', help='path of bmodel')
    parser.add_argument('--tpu_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)
