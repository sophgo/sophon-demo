#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import motmetrics as mm
import numpy as np 
import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)
def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--gt_file', type=str, default='../datasets/mot15_trainset/ADL-Rundle-6/gt/gt.txt', help='mot ground truth')
    parser.add_argument('--ts_file', type=str, default='../python/results/mot_eval/ADL-Rundle-6_extractor_fp32_1b.bmodel.txt', help='your own results')
    parser.add_argument('--run_mot15_benchmark', type=bool, default=False)
    args = parser.parse_args()
    return args

def motMetricsEnhancedCalculator(gtSource, tSource):
    # load ground truth
    gt = np.loadtxt(gtSource, delimiter=',')

    # load tracking output
    t = np.loadtxt(tSource, delimiter=',')

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:,0].max())):
        frame += 1 # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
        t_dets = t[t[:,0]==frame,1:6] # select all detections in t

        C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                    max_iou=0.5) # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:,0].astype('int').tolist(), \
                t_dets[:,0].astype('int').tolist(), C)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                        'recall', 'precision', 'num_objects', \
                                        'mostly_tracked', 'partially_tracked', \
                                        'mostly_lost', 'num_false_positives', \
                                        'num_misses', 'num_switches', \
                                        'num_fragmentations', 'mota', 'motp' \
                                        ], \
                        name='acc')
    print("MOTA =", summary['mota']['acc'])
    strsummary = mm.io.render_summary(
        summary,
        #formatters={'mota' : '{:.2%}'.format},
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
                'precision': 'Prcn', 'num_objects': 'GT', \
                'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
                'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
                'num_misses': 'FN', 'num_switches' : 'IDsw', \
                'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
                }
    )
    print(strsummary)
if __name__ == "__main__":
    args = argsparser()
    if args.run_mot15_benchmark == False:
        motMetricsEnhancedCalculator(args.gt_file, args.ts_file)