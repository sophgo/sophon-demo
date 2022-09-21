import argparse
import json
import logging
logging.basicConfig(level=logging.DEBUG)

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--gt_path', type=str, default='../data/images/imagenet_val_1k/label.txt', help='path of label')
    parser.add_argument('--pred_path', type=str, default='../python/results/resnet_fp32_b1.bmodel_img_opencv_python_result.txt', help='path of result')
    args = parser.parse_args()
    return args

def main(args):

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
