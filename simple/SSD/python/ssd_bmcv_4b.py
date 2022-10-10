""" Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import sys
import os
import argparse
import collections
import json
import numpy as np
import sophon.sail as sail

class PreProcessor:
  """ Preprocessing class.
  """
  def __init__(self, bmcv, scale):
    """ Constructor.
    """
    self.bmcv = bmcv
    self.ab = [x * scale for x in [1, -123, 1, -117, 1, -104]]

  def process(self, input, output):
    """ Execution function of preprocessing.
    Args:
      input: sail.BMImage, input image
      output: sail.BMImage, output data

    Returns:
      None
    """
    tmp = self.bmcv.vpp_resize(input, 300, 300)
    self.bmcv.convert_to(tmp, output, ((self.ab[0], self.ab[1]), \
                                       (self.ab[2], self.ab[3]), \
                                       (self.ab[4], self.ab[5])))

class PostProcess:
  """ Postprocessing class.
  """
  def __init__(self, threshold):
    """ Constructor.
    """
    self.threshold = threshold

  def process(self, data, img_w, img_h):
    """ Execution function of postprocessing.
    Args:
      data: Inference output
      img_w: Image width
      img_h: Imgae height

    Returns:
      Detected boxes.
    """
    data = data.reshape((data.shape[2], data.shape[3]))
    ret = collections.defaultdict(list)
    for proposal in data:
      if proposal[2] < self.threshold:
        continue
      idx = proposal[0]
      ret[idx].append([
          int(proposal[1]),           # class id
          proposal[2],                # score
          int(proposal[3] * img_w),   # x0
          int(proposal[4] * img_h),   # x1
          int(proposal[5] * img_w),   # y0
          int(proposal[6] * img_h)])  # y1
    return ret

  def get_reference(self, compare_path):
    """ Get correct result from given file.
    Args:
      compare_path: Path to correct result file

    Returns:
      Correct result.
    """
    if compare_path:
      with open(compare_path, 'r') as f:
        data = json.load(f)
        reference = []
        for item in data["frames"]:
          reference.append(item["boxes"])
        return reference
    return None

  def compare(self, reference, result, loop_id):
    """ Compare result.
    Args:
      reference: Correct result
      result: Output result
      loop_id: Loop iterator number

    Returns:
      True for success and False for failure
    """
    if not reference:
      print("No verify_files file or verify_files err.")
      return True
    if loop_id > 0:
      return True
    for i, key in enumerate([0.0, 1.0, 2.0, 3.0]):
      data = []
      for line in result[key]:
        cp_line = line.copy()
        cp_line[1] = "{:.8f}".format(cp_line[1])
        data.append(cp_line)
      if len(data) != len(reference[i]):
        message = "Expected deteted number is {}, but detected {}!"
        print(message.format(len(reference), len(data)))
        return False
      ret = True
      message = "[Frame {}] Category: {}, Score: {}, Box: [{}, {}, {}, {}]"
      fail_info = "Compare failed! Expect: " + message
      ret_info = "Result Box: " + message
      for j in range(len(data)):
        box = data[j]
        ref = reference[i][j]
        if box != ref:
          print(fail_info.format(i, ref[0], float(ref[1]), ref[2],\
                                 ref[3], ref[4], ref[5]))
          print(ret_info.format(i, box[0], float(box[1]), box[2],\
                                box[3], box[4], box[5]))
          ret = False
    return ret

def inference(bmodel_path, input_path, loops, tpu_id, compare_path):
  """ Load a bmodel and do inference.
  Args:
   bmodel_path: Path to bmodel
   input_path: Path to input file
   loops: Number of loops to run
   tpu_id: ID of TPU to use
   compare_path: Path to correct result file

  Returns:
    True for success and False for failure
  """
  # init Engine
  engine = sail.Engine(tpu_id)
  # load bmodel without builtin input and output tensors
  engine.load(bmodel_path)
  # get model info
  # only one model loaded for this engine
  # only one input tensor and only one output tensor in this graph
  graph_name = engine.get_graph_names()[0]
  input_name   = engine.get_input_names(graph_name)[0]
  output_name  = engine.get_output_names(graph_name)[0]
  input_shape  = [4, 3, 300, 300]
  input_shapes = {input_name: input_shape}
  output_shape = [1, 1, 800, 7]
  input_dtype  = engine.get_input_dtype(graph_name, input_name)
  output_dtype = engine.get_output_dtype(graph_name, output_name)
  is_fp32 = (input_dtype == sail.Dtype.BM_FLOAT32)
  # get handle to create input and output tensors
  handle = engine.get_handle()
  input  = sail.Tensor(handle, input_shape,  input_dtype,  False, False)
  output = sail.Tensor(handle, output_shape, output_dtype, True,  True)
  input_tensors  = { input_name:  input  }
  output_tensors = { output_name: output }
  # set io_mode
  engine.set_io_mode(graph_name, sail.IOMode.SYSO)
  # init bmcv for preprocess
  bmcv = sail.Bmcv(handle)
  img_dtype = bmcv.get_bm_image_data_format(input_dtype)
  # init preprocessor and postprocessor
  scale = engine.get_input_scale(graph_name, input_name)
  preprocessor = PreProcessor(bmcv, scale)
  threshold = 0.59 if is_fp32 else 0.52
  postprocessor = PostProcess(threshold)
  reference = postprocessor.get_reference(compare_path)
  # init decoder
  decoder = sail.Decoder(input_path, True, tpu_id)
  status = True
  # pipeline of inference
  for i in range(loops):
    imgs_0 = sail.BMImageArray4D()
    imgs_1 = sail.BMImageArray4D(handle, input_shape[2], input_shape[3], \
                                 sail.Format.FORMAT_BGR_PLANAR, img_dtype)
    # read 4 frames from input video for batch size is 4
    flag = False
    for j in range(4):
      ret = decoder.read_(handle, imgs_0[j])
      if ret != 0:
        print("Finished to read the video!");
        flag = True
        break
    if flag:
      break
    # preprocess
    preprocessor.process(imgs_0, imgs_1)
    bmcv.bm_image_to_tensor(imgs_1, input)
    # inference
    engine.process(graph_name, input_tensors, input_shapes, output_tensors)
    # postprocess
    real_output_shape = engine.get_output_shape(graph_name, output_name)
    out = output.asnumpy(real_output_shape)
    dets = postprocessor.process(out, imgs_0[0].width(), imgs_0[0].height())
    # print result
    if postprocessor.compare(reference, dets, i):
      for j, vals in dets.items():
        frame_id = int(i * 4 + j + 1)
        img0 = sail.BMImage(imgs_0[j])
        for class_id, score, x0, y0, x1, y1 in vals:
          msg = '[Frame {} on tpu {}] Category: {}, Score: {:.3f},'
          msg += ' Box: [{}, {}, {}, {}]'
          print(msg.format(frame_id, tpu_id, class_id, score, x0, y0, x1, y1))
          bmcv.rectangle(img0, x0, y0, x1 - x0 + 1, y1 - y0 + 1, (255, 0, 0), 3)
        bmcv.imwrite('result-{}.jpg'.format(frame_id), img0)
    else:
      status = False
      break
  return status

if __name__ == '__main__':
  """ A SSD example using bm-ffmpeg to decode and bmcv to preprocess with
      batch size == 4 to speed up for int8 model.
  """
  desc='decode (ffmpeg) + preprocess (bmcv) + inference (sophon inference)'
  PARSER = argparse.ArgumentParser(description=desc)
  PARSER.add_argument('--bmodel', default='', required=True)
  PARSER.add_argument('--input', default='', required=True)
  PARSER.add_argument('--loops', default=1, type=int, required=False)
  PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
  PARSER.add_argument('--compare', default='', required=False)
  ARGS = PARSER.parse_args()
  # if not os.path.exists(ARGS.input):
  #   print("Error: {} not exists!".format(ARGS.input))
  #   sys.exit(-2)
  status = inference(ARGS.bmodel, ARGS.input, \
                     ARGS.loops, ARGS.tpu_id, ARGS.compare)
  sys.exit(0 if status else -1)

