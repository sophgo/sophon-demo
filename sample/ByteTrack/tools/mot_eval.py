import motmetrics as mm
from loguru import logger
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ground_truths',
        type=str,
        default="../data/MOT15/ADL-Rundle-6/gt/gt.txt",
    )
    parser.add_argument(
        '--detections',
        type=str,
        default="../python/results/bytetrack_opencv/img1_bytetrack_s_fp32_1b_py.txt",
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    # 评价指标
    metrics = list(mm.metrics.motchallenge_metrics)
    # 导入gt和ts文件
    gt_file = args.ground_truths
    print('gt_files', gt_file)
    ts_file = args.detections
    print('ts_file', ts_file)

    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')

    gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
    ts = mm.io.loadtxt(ts_file, fmt="mot15-2D", min_confidence=-1)
    name = os.path.splitext(os.path.basename(ts_file))[0]

    logger.info('Running metrics')

    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=metrics, name=name)
    print(mm.io.render_summary(summary, formatters=mh.formatters,
                               namemap=mm.io.motchallenge_metric_names))

    logger.info('Completed')
