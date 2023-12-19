#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
import sophon.sail as sail
from collections import OrderedDict
import logging
import time

LOG_FORMAT = "%(levelname)s %(asctime)s.%(msecs)d %(thread)d %(filename)s:%(lineno)d] %(funcName)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger = logging.getLogger()


class SophonInference:
    def __init__(self, **kwargs):
        self.device_id = kwargs.get("device_id", 0)
        self.model_path = kwargs.get("model_path", None)
        self.input_mode = kwargs.get("input_mode", 0)  # numpy: 0, bm_iamge: not 0
        self.io_mode = sail.IOMode.SYSIO
        # self.data_format = kwargs.get("INIT_data_format", 'NCHW')

        # Create a Context on sophon device
        logger.info("config : {}".format(kwargs))
        tpu_count = sail.get_available_tpu_num()
        logger.debug('{} TPUs Detected, using TPU {} \n'.format(tpu_count, self.device_id))

        self.engine = sail.Engine(self.model_path, self.device_id, self.io_mode)
        self.handle = self.engine.get_handle()
        self.graph_name = self.engine.get_graph_names()[0]
        graph_count = len(self.engine.get_graph_names())
        logger.warning("{} graphs in {}, using {}".format(graph_count, self.model_path, self.graph_name))

        # create bmcv handle
        self.bmcv = sail.Bmcv(self.handle)

        # create input tensors
        input_names = self.engine.get_input_names(self.graph_name)
        input_tensors = {}
        input_shapes = {}
        input_scales = {}
        input_dtypes = {}
        img_dtypes = {}
        inputs = []
        inputs_shapes = []

        for input_name in input_names:
            input_shape = self.engine.get_input_shape(self.graph_name, input_name)
            input_dtype = self.engine.get_input_dtype(self.graph_name, input_name)
            input_scale = self.engine.get_input_scale(self.graph_name, input_name)

            logger.debug("[{}] create sail.Tensor for input: {} ".format(input_name, input_shape))
            if self.input_mode:
                input = sail.Tensor(self.handle, input_shape, input_dtype, True, True)
            else:
                input = None

            inputs.append(input)
            inputs_shapes.append(input_shape)
            input_tensors[input_name] = input
            input_shapes[input_name] = input_shape
            input_scales[input_name] = input_scale
            input_dtypes[input_name] = input_dtype
            #img_dtypes[input_name] = self.bmcv.get_bm_image_data_format(input_dtype)

        # create output tensors
        output_names = self.engine.get_output_names(self.graph_name)
        output_tensors = {}
        output_shapes = {}
        output_scales = {}
        output_dtypes = {}
        outputs = []

        for output_name in output_names:
            output_shape = self.engine.get_output_shape(self.graph_name, output_name)
            output_dtype = self.engine.get_output_dtype(self.graph_name, output_name)
            output_scale = self.engine.get_output_scale(self.graph_name, output_name)

            # create sail.Tensor for output
            logger.debug("[{}] create sail.Tensor for output: {} ".format(output_name, output_shape))
            if self.input_mode:
                output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            else:
                output = None

            outputs.append(output)
            output_tensors[output_name] = output
            output_shapes[output_name] = output_shape
            output_scales[output_name] = output_scale
            output_dtypes[output_name] = output_dtype

        # Store
        self.inputs = inputs
        self.input_names = input_names
        self.inputs_shapes = inputs_shapes
        self.input_tensors = input_tensors
        self.input_scales = input_scales
        self.input_dtypes = input_dtypes
        self.img_dtypes = img_dtypes

        self.outputs = outputs
        self.output_names = output_names
        self.output_tensors = output_tensors
        self.output_shapes = output_shapes
        self.output_scales = output_scales

        #
        logger.info("===========================================")
        logger.info("BModel: {}".format(self.model_path))
        logger.info("Input : {}, {}".format(input_shapes, input_dtypes))
        logger.info("Output: {}, {}".format(output_shapes, output_dtypes))
        logger.info("===========================================")

    def get_input_feed_numpy(self, input_names, inputs):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param inputs: image_numpy
        :return:
        """
        input_feed = {}
        for i, (name, input) in enumerate(zip(input_names, inputs)):
            input_feed[name] = np.ascontiguousarray(input)
        return input_feed

    def outputToList_numpy(self, output, real_num=None):
        results = list()
        for name in self.output_names:
            if real_num == None:
                results.append(output[name])
            else:
                results.append(output[name][:real_num][:])
        return results

    def outputToList(self, output, real_num=None):
        results = list()
        for name in self.output_names:
            if real_num == None:
                results.append(output[name].asnumpy())
            else:
                results.append(output[name][:real_num][:].asnumpy())
        return results

    def infer_numpy(self, input_data):
        """
        input_data: [input0, input1, ...]
        Args:
            input_data:

        Returns:

        """
        # logger.debug("input_data shape: {}".format(input_data.shape))
        inputs_feed = self.get_input_feed_numpy(self.input_names, input_data)
        outputs = self.engine.process(self.graph_name, inputs_feed)
        '''
        outputs_dict = OrderedDict()
        for name in self.output_names:
            outputs_dict[name] = outputs[name]
        # logger.debug(outputs.keys())
        return outputs_dict
        # return self.outputToList_numpy(outputs)
        '''
        return outputs

    def get_input_feed(self, input_names, inputs):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param inputs: image_numpy
        :return:
        """
        for i, (name, input) in enumerate(zip(input_names, inputs)):
            self.bmcv.bm_image_to_tensor(input, self.inputs[i])

    def infer_numpy_dict(self, input_dict):
        """
        Args:
            input_dict: {"input_name1": input1, "input_name2", input2 ...}
        
        Returns:
            output_dict
        """
        outputs = self.engine.process(self.graph_name, input_dict)
        return outputs
