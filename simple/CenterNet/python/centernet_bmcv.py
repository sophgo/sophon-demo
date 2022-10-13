import os
import cv2
import time
import logging
import numpy as np
import sophon.sail as sail
from datetime import datetime
import argparse
from numpy.lib.stride_tricks import as_strided
from ctdet import CtdetDetector
from debugger import Debugger

BASE_DIR = os.path.dirname(os.path.join(os.getcwd(), __file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data/')

      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for CenterNet")
    parser.add_argument('--input',  default=os.path.join(DATA_DIR, 'ctdet_test.jpg'), required=False)
    parser.add_argument('--loops',  default=1,  type=int, required=False)
    parser.add_argument('--tpu_id', default=0,  type=int, required=False)
    parser.add_argument('--bmodel', default=os.path.join(DATA_DIR, 'models/ctdet_coco_dlav0_1output_512_fp32_1batch.bmodel'), type=str, required=False)
    parser.add_argument('--class_path', default=os.path.join(DATA_DIR, 'coco_classes.txt'), type=str, required=False)

    opt = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s%(msecs)03d[%(levelname)s][%(module)s:%(lineno)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S.')
    logging.info('Start centernet detector sail demo.')

    # Initialize centernet detector instance
    cet_detector = CtdetDetector(
        arch='dlav0', model_path=opt.bmodel, tpu_id=opt.tpu_id, class_path=opt.class_path
    )
    
    batch_size = cet_detector.get_batchsize()
    logging.info("Input model batch size is {}".format(batch_size))
        
    input_path        = opt.input
    decoder           = sail.Decoder(input_path, True, opt.tpu_id)
    process_handle    = cet_detector.get_handle()
    
    for idx in range(opt.loops):
        logging.info('loop start')
        image_ost_list = []

        if batch_size == 1:
            img = sail.BMImage()
            ret = decoder.read(process_handle, img)
            if ret != 0:
                logging.warning('decoder read image failed!')
                continue
            dst_img = cet_detector.bmcv.convert_format(img)
            image_ost_list.append(dst_img)         
            logging.info('input format {}'.format(dst_img.format()))
            results = cet_detector.run(dst_img)
            
        elif batch_size == 4:
            #img = sail.BMImageArray4D()
            for i in range(4):
                rgb_img = cet_detector.bmcv.convert_format(decoder.read(process_handle))
                image_ost_list.append(rgb_img)
                #img[i] = rgb_img.data()
            results = cet_detector.run(image_ost_list)
            
        else:
            raise NotImplementedError(
                'This demo not supports inference with batch size {}'.format(cet_detector.get_batchsize()))
        
        debugger = Debugger()
        for b in range(len(results)):
            if results[b] is None:
                logging.info('batch {} detect nothing'.format(b + 1))
                continue
            cv_img    = cet_detector.bmimage_to_cvmat(image_ost_list[b])
            top_label = np.array(results[b][:, 5], dtype='int32')
            top_conf  = results[b][:, 4]
            top_boxes = results[b][:, :4]
            
            #---------------------------------------------------------#
            #   图像绘制
            #---------------------------------------------------------#
            for i, c in list(enumerate(top_label)):
                predicted_class = cet_detector.class_names[int(c)]
                box             = top_boxes[i]
                score           = top_conf[i]

                top, left, bottom, right = box

                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image_ost_list[b].height(), np.floor(bottom).astype('int32'))
                right   = min(image_ost_list[b].width(),  np.floor(right).astype('int32'))

                logging.info('[object]:{} -> label {}, top {}, left {}, bottom {}, right {}'.format(score, predicted_class, top, left, bottom, right))
                debugger.add_coco_bbox(cv_img, (left, top, right, bottom), c, score)
                #cet_detector.bmcv.rectangle(image_ost_list[b], left, top, right - left, bottom - top, (255, 0, 0), 3)

            # draw result
            det_filename = './results/ctdet_result_{}_b_{}.jpg'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), b)
            #cet_detector.bmcv.imwrite(det_filename, image_ost_list[b])
            cv2.imwrite(det_filename, cv_img)
            logging.info('Prediction result: {}'.format(det_filename))
        
    # exit
    logging.info('Demo exit..')
