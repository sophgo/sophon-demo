#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import argparse
import json
import logging
logging.basicConfig(level=logging.DEBUG)

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--gt_path', type=str, default='../data/images/imagenet_val_1k/label.txt', help='path of label')
    parser.add_argument('--pred_path', type=str, default='../results', help='path of result folder')
    parser.add_argument('--data_type', type=str, default='fp32', help='data type of the model, choose from fp32, fp16 and int8')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of the model, choose from 1 and 4')
    parser.add_argument('--img_module', type=str, default='opencv', help='image processing module, choose from bmcv and opencv')
    parser.add_argument('--lan',type=str,default='python',help='language type, choose from cpp and python')
    args = parser.parse_args()
    return args

def main(args):
    assert (args.data_type in ('fp32', 'fp16', 'int8')), "Data type must be fp32, fp16 or int8!"
    assert (args.batch_size in (1, 4)), "Batch size must be 1 or 4!"
    assert (args.img_module in ('bmcv', 'opencv')), "Please choose from bmcv and opencv."
    args.pred_path = args.pred_path + f'/resnet_{args.data_type}_b{args.batch_size}.bmodel_img_{args.img_module}_{args.lan}_result.txt'
    
    d_gt = dict([l[:-1].split('\t') for l in open(args.gt_path, 'r').readlines()])
    
    d_pred = {}
    for l in open(args.pred_path, 'r').readlines():
        name, prediction, score = l[:-1].split('\t')
        name = name.split('/')[-1]
        d_pred[name]  = prediction

    correct = 0
    for k, gt in d_gt.items():
        prediction = d_pred[k]
        if gt==prediction:
            correct += 1
    acc = correct / float(len(d_gt))

    logging.info('gt_path: {}'.format(args.gt_path))
    logging.info('pred_path: {}'.format(args.pred_path))
    logging.info('ACC: {:.5f}%'.format(acc*100))


if __name__ == '__main__':
    args = argsparser()
    main(args)
