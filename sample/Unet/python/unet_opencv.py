import argparse
import cv2
import numpy as np
import sophon.sail as sail
import os
import time
import logging
logging.basicConfig(level=logging.INFO)


class UNet:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))
        self.handle = self.net.get_handle()
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]
        
        # get input
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype= self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}
        self.out_threshold = args.out_threshold
        self.n_classes = args.n_classes
        
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
        self.use_resize_padding = True
        self.use_vpp = False
        self.ab = [x * self.input_scale / 255.  for x in [1, 0, 1, 0, 1, 0]]

        # init time
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
    
    def preprocess_opencv(self, input_img):
        input_img = cv2.resize(input_img, (self.net_w, self.net_h))
        # cv2.imwrite("preprocessed_opencv.jpg", input_img)
        input_img = input_img.astype('float32')/255.0
        input_img = np.transpose(input_img, (2, 0, 1))[::-1]
        return input_img
    
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
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def argmax(self, pred, m, n):
        ans = -1
        max_score = -1
        for i in range(self.n_classes):
            if(pred[0, i, m, n] > max_score):
                ans = i
                max_score = pred[0, i, m, n]
        return ans

    def postprocess(self, pred_batch, ori_size_batch):
        result_list = []
        for i in range(len(pred_batch)):
            pred = pred_batch[i]
            result = np.zeros([pred.shape[2], pred.shape[3]])
            if self.n_classes == 1:
                for m in range(pred.shape[2]):
                    for n in range(pred.shape[3]):
                        result[m, n] = 0 if self.sigmoid(pred[0, 0, m, n]) < self.out_threshold else 255
            elif self.n_classes ==  2:
                for m in range(pred.shape[2]):
                    for n in range(pred.shape[3]):
                        result[m,n] = 0 if pred[0,0,m,n] > pred[0,1,m,n] else 255
            else:
                for m in range(pred.shape[2]):
                    for n in range(pred.shape[3]):
                        result[m, n] = self.argmax(pred, m, n)
            # resize
            result = cv2.resize(result, (ori_size_batch[i][0], ori_size_batch[i][1])).astype('uint8')
            result_list.append(result)
        return result_list

    
    def __call__(self, input_image_list):
        img_num = len(input_image_list)
        ori_size_list = []
        if self.batch_size == 1:
            ori_h, ori_w = input_image_list[0].shape[0], input_image_list[0].shape[1]
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()
            preprocessed_img = self.preprocess_opencv(input_image_list[0])
            self.preprocess_time += time.time() - start_time
            input_tensor = sail.Tensor(self.handle, np.stack(preprocessed_img), False)
        else:
            imgs = []
            for i in range(img_num):
                ori_h, ori_w = input_image_list[i].shape[0], input_image_list[i].shape[1]
                ori_size_list.append((ori_w, ori_h))
                start_time = time.time()
                preprocessed_img = self.preprocess_opencv(input_image_list[i])
                self.preprocess_time += time.time() - start_time
                imgs.append(preprocessed_img)
            input_tensor = sail.Tensor(self.handle, np.stack(imgs), False)
        
        start_time = time.time()
        outputs = self.predict(input_tensor, img_num)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        result = self.postprocess(outputs, ori_size_list)
        self.postprocess_time += time.time() - start_time

        return result



def main(opt):
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

    # initialize net
    unet = UNet(args)
    batch_size = unet.batch_size
    unet.init()

    decode_time = 0.0
    # test images 
    if os.path.isdir(args.input):
        input_list = []
        filename_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                img_file = os.path.join(root, filename)
                logging.info("{}, img_file: {}".format(cn, img_file))
                # decode
                start_time = time.time()
                input_image = cv2.imread(img_file) # cv2.imread, BGR default
                decode_time += time.time() - start_time
                
                input_list.append(input_image)
                filename_list.append(filename)
                if len(input_list) == batch_size:
                    # predict
                    results = unet(input_list)
                    for i, filename in enumerate(filename_list):
                        det = results[i]
                        cn += 1
                        # save image
                        cv2.imwrite(os.path.join(output_img_dir, filename), det)
                        
                    input_list.clear()
                    filename_list.clear()
        if len(input_list):
            results = unet(input_list)
            for i, filename in enumerate(filename_list):
                det = results[i]
                cn += 1
                cv2.imwrite(os.path.join(output_img_dir, filename), det)
            input_list.clear()
            filename_list.clear()
    else:
        # test video
        cap = cv2.VideoCapture()
        if not cap.open(args.input):
            raise Exception("can not open the video")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        save_video = os.path.join(output_dir, os.path.splitext(os.path.split(args.input)[1])[0] + '.avi')
        cn = 0
        frame_list = []
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            decode_time += time.time() - start_time
            if not ret or frame is None:
                break
            frame_list.append(frame)
            if len(frame_list) == batch_size:
                results = unet(frame_list)
                for i, frame in enumerate(frame_list):
                    det = cv2.resize(results[i], size)
                    cv2.imwrite(os.path.join(output_img_dir, str(cn)+".jpg"), det)
                    cn += 1
                frame_list.clear()
        if len(frame_list):
            results = unet(frame_list)
            for i, frame in enumerate(frame_list):
                det = cv2.resize(results[i], size)
                cv2.imwrite(os.path.join(output_img_dir, str(cn)+".jpg"), det)
                cn += 1
                logging.info("{}, det nums: {}".format(cn, det.shape[0]))
        cap.release()
        logging.info("result saved in {}".format(save_video))
    
    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = unet.preprocess_time / cn
    inference_time = unet.inference_time / cn
    postprocess_time = unet.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684/unet_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--out_threshold', type=float, default=0.5, help='the threshold while converting output tensor to mask, only if n_classes == 1')
    parser.add_argument('--n_classes', type=int, default=2, help='the number of segmentation classes')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')