#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
import time
import sophon.sail as sail
from .deep.feature_extractor_bmcv import Extractor

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, dev_id, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100):

        self.extractor = Extractor(model_path, dev_id)
        # currently, the interface below only supports extractor bmodel with one batch
        self.postprocess = sail.deepsort_tracker_controller(max_cosine_distance=max_dist, 
                                            nn_budget=nn_budget, 
                                            k_feature_dim=self.extractor.output_shape[1], 
                                            max_iou_distance=max_iou_distance, 
                                            max_age=max_age, n_init=n_init)
        
        self.postprocess_time = 0.0
        self.crop_num = 0
        self.handle = sail.Handle(dev_id)
        self.bmcv = sail.Bmcv(self.handle)

    def update(self, detector_results, ori_img):
        # shape of single detector_result is [x1,y1,x2,y2,cls,score], only result of one pic
        # ori_img is bm_image

        # generate detections
        # before using deepsort postprocess interface
        # the shape of features should be converted to the same of detector result
        # especially when using extractor with 4 batch model
        features = self._get_features(detector_results, ori_img)[:len(detector_results)]
        start_postprocess = time.time()
        # detector_result[:, [4,5]] =detector_result[:, [5,4]]
        det_tuple = tuple(detector_results)
        track_res = self.postprocess.process(det_tuple, features)

        self.postprocess_time += time.time() - start_postprocess
        return track_res



    def _get_features(self, detector_result, ori_img):
        im_crops = []
        for det in detector_result:
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            im = self.bmcv.crop(ori_img, x1, y1, x2-x1, y2-y1)
            im_crops.append(im)
        if im_crops:
            self.crop_num += len(im_crops)
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
