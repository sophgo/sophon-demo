#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
import cv2
# import scipy.special
from .utils import softmax

class PostProcess:
    def __init__(self, conf_thresh=0.5):
        self.conf_thresh = conf_thresh

    def __call__(self, pred_scores, pred_points, ratios):
        """
        post-processing for single image
        :param pred_scores:
        :param pred_points:
        :return:
        """
        outputs_scores = pred_scores[:, 1]
        outputs_points = pred_points
        # print(outputs_points)
        # filter the predictions
        scores_above_thresh = outputs_scores > self.conf_thresh
        # print(scores_above_thresh)
        points = outputs_points[scores_above_thresh]
        predict_cnt = int(scores_above_thresh.sum())
        # back to original image size
        points[:, 0] /= ratios[0]
        points[:, 1] /= ratios[1]

        return points, predict_cnt

    def infer_batch(self, pred_logits_batch, pred_points_batch, ratios_batch):
        """
        post-processing using single post-processing for loop
        :param pred_logits_batch:
        :param pred_points_batch:
        :return:
        """
        # print("score: ", pred_logits_batch.shape, pred_logits_batch)
        # print("point: ", pred_points_batch.shape, pred_points_batch)
        # scipy.special.softmax. It is hard to install scipy in se5
        outputs_scores_batch = softmax(pred_logits_batch, axis=-1)
        # for i in outputs_scores_batch[0][0].tolist():
        #     print(i)
        outputs_points_batch = pred_points_batch
        points_batch, predict_cnt_batch = [], []
        for i in range(len(pred_logits_batch[0])):
            points, predict_cnt = self(outputs_scores_batch[0][i],
                                       outputs_points_batch[0][i],
                                       ratios_batch[i],
                                       )
            points_batch.append(points)
            predict_cnt_batch.append(predict_cnt)

        return points_batch, predict_cnt_batch

    # def infer_batch(self, out_scores, out_coordss):
    #     """
    #     post-processing
    #     :param out_scores:
    #     :param out_coords:
    #     :return:
    #     """
    #     # outputs_scores = softmax(out_scores, -1)[:, :, 1]
    #     scores_dim = out_scores.shape
    #     outputs_scores = out_scores
    #     for si in range(scores_dim[0]):
    #         for sj in range(scores_dim[1]):
    #             # print(out_scores[si,sj,:])
    #             outputs_scores[si,sj,:] = softmax_numpy(out_scores[si,sj,:])
    #     print(outputs_scores)
    #     outputs_points = out_coordss

    #     threshold = 0.5
    #     # filter the predictions
    #     points = outputs_points[outputs_scores > threshold]
    #     predict_cnt = int((outputs_scores > threshold).sum())

    #     # # draw the predictions
    #     # size = 2
    #     # img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    #     # for p in points:
    #     #     img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    #     # # save the visualized image
    #     # cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)


    #     return points, predict_cnt
