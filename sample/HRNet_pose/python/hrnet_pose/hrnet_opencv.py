#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import time
import numpy as np
import sophon.sail as sail

from .preprocess_hrnet import preprocess
from .postprocess_hrnet import flip_images, flip_back, postprocess
from .utils_hrnet import FLIP_PAIRS

import logging
logging.basicConfig(level=logging.INFO)


class HRNet:
    def __init__(self, args):
        self.net = sail.Engine(args.pose_bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.pose_bmodel))

        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_name = self.net.get_output_names(self.graph_name)[0]

        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        assert self.batch_size == 1, "The batch size of human pose estimation must be 1."

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        """
            Time used to initialize preprocessing, inference, and post-processing.
        """
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def inference(self, args, preprocessed_image):
        """
        HRNet was used to estimate the pose of the input image.

        Parameters:
          args: Arguments.
          preprocessed_image (): The pre-processed image.

        Returns:
          outputs (dict): Heatmap of HRNet outputs.
        """
        input_data = {self.input_name: preprocessed_image}

        # feature is not aligned, shift flipped heatmap for higher accuracy
        # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
        if args.flip:

            start_time = time.time()
            flipped_images = flip_images(preprocessed_image)
            self.preprocess_time += time.time() - start_time

            start_time = time.time()
            flipped_outputs = self.net.process(self.graph_name, {self.input_name: flipped_images})
            outputs = self.net.process(self.graph_name, input_data)
            self.inference_time += time.time() - start_time

            start_time = time.time()
            flipped_outputs = flip_back(flipped_outputs[self.output_name], FLIP_PAIRS)
            flipped_outputs[..., 1:] = flipped_outputs[..., 0:-1]
            final_outputs = (outputs[self.output_name] + flipped_outputs) * 0.5
            outputs[self.output_name] = final_outputs
            self.postprocess_time += time.time() - start_time

        else:
            start_time = time.time()
            outputs = self.net.process(self.graph_name, input_data)
            self.inference_time += time.time() - start_time

        return outputs
    def __call__(self, args, image_pose, box):
        """
        Perform pre-processing, inference, and post-processing to return predicted keypoints and scores

        Parameters:
          args: Arguments.
          image_pose (): The original image.
          box (list): The box coordinates of the image, only one box.

        Returns:
          keypoints (ndarry): Keypoints information.
          maxvals (ndarry): The scores of keypoints.
        """

        start_time = time.time()

        preprocessed_image = preprocess(image_pose, box, [self.net_h, self.net_w])
        self.preprocess_time += time.time() - start_time

        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        outputs = self.inference(args, preprocessed_image)

        outputs = outputs[self.output_name]

        start_time = time.time()
        key_points, max_vals = postprocess(outputs, box)
        self.postprocess_time += time.time() - start_time

        return key_points, max_vals



