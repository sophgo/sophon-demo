from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

import cv2
import sophon.sail as sail
import logging


class BaseDetector(object):

  def __init__(self, arch='dlav0', model_path='', tpu_id=0, class_path=''):
    # common config
    self.confidence = 0.35
    self.letterbox_image = True
    self.mean = np.array([0.408, 0.447, 0.470], dtype=np.float32)
    self.std  = np.array([0.289, 0.274, 0.278], dtype=np.float32)
    self.class_names, self.num_classes  = self.get_classes(class_path)
    
    logging.info('Creating model...')
    self.model_factory(arch)(model_path, tpu_id)

    self.max_per_image = 100
    self.num_classes = 80
    #self.scales = [1.0]
    self.nms  = False

  def model_factory(self, arch):
    factory = {
      'dlav0': self.dlav0_initialize
    }
    return factory[arch]
  
  #---------------------------------------------------#
  #   获得类
  #---------------------------------------------------#
  def get_classes(self, class_path):
      with open(class_path, encoding='utf-8') as f:
          class_names = f.readlines()
      class_names = [c.strip() for c in class_names]
      return class_names, len(class_names)
    
  def dlav0_initialize(self, model_path, tpu_id):
     # 加载bmodel
      self.engine         = sail.Engine(model_path, tpu_id, sail.IOMode.SYSO)
      self.graph_name     = self.engine.get_graph_names()[0]
      self.input_name     = self.engine.get_input_names(self.graph_name)[0]
      self.output_name    = self.engine.get_output_names(self.graph_name)[0]
      self.input_dtype    = self.engine.get_input_dtype(self.graph_name, self.input_name)
      self.output_dtype   = self.engine.get_output_dtype(self.graph_name, self.output_name)
      self.input_shape    = self.engine.get_input_shape(self.graph_name, self.input_name)
      self.input_w        = int(self.input_shape[-1])
      self.input_h        = int(self.input_shape[-2])
      self.output_shape   = self.engine.get_output_shape(self.graph_name, self.output_name)
      self.handle         = self.engine.get_handle()
      self.input          = sail.Tensor(self.handle, self.input_shape, self.input_dtype, 
                                        True, True)
      self.output         = sail.Tensor(self.handle, self.output_shape, self.output_dtype, 
                                        True, True)
      self.input_tensors  = { self.input_name  : self.input }
      self.output_tensors = { self.output_name : self.output}
      
      self.bmcv           = sail.Bmcv(self.handle)
      self.img_dtype      = self.bmcv.get_bm_image_data_format(self.input_dtype)
      self.input_scale    = self.engine.get_input_scale(self.graph_name, self.input_name)
      self.output_scale   = self.engine.get_output_scale(self.graph_name, self.output_name)

      # batch size 1
      if self.output_shape[0] == 1:
          self.input_bmimage  = sail.BMImage(self.handle, 
                                          self.input_w, self.input_h,
                                          sail.Format.FORMAT_BGR_PLANAR, 
                                          self.img_dtype)
      elif self.output_shape[0] == 4:
          self.input_bmimage  = sail.BMImageArray4D(self.handle, 
                                                    self.input_w, self.input_h,
                                                    sail.Format.FORMAT_BGR_PLANAR, 
                                                    self.img_dtype)
      else:
          raise NotImplementedError(
              'This demo not supports inference with batch size {}'.format(self.output_shape[0]))
          
      a_list  = 1 / 255 / self.std
      b_list  = -self.mean / self.std
      self.ab = []
      for i in range(3):
        self.ab.append(self.input_scale * a_list[i])
        self.ab.append(self.input_scale * b_list[i])
        
      logging.info("\n" + "*" * 50 + "\n"
                "graph_name:    {}\n"
                "input_name:    {}\n"
                "output_name:   {}\n"
                "input_dtype:   {}\n"
                "output_dtype:  {}\n"
                "input_shape:   {}\n"
                "output_shape:  {}\n"
                "img_dtype:     {}\n"
                "input_scale:   {}\n"
                "output_scale:  {}\n".format(self.graph_name, self.input_name, self.output_name,
                                              self.input_dtype, self.output_dtype, self.input_shape, 
                                              self.output_shape, self.img_dtype, self.input_scale, self.output_scale)
                                              + "*" * 50)
  def get_batchsize(self):
    return int(self.input_shape[0])
      
  def get_handle(self):
    return self.handle         
  
  def pre_process(self, input_img):
      img_list = []
      resized_list = []
      if isinstance(input_img, list):
          img_list = input_img
      else:
          img_list.append(input_img)
      for bm_image in img_list:
          # letterbox padding
          if self.letterbox_image:
              if bm_image.width() > bm_image.height():
                  resize_ratio = self.input_w / bm_image.width()
                  target_w     = int(self.input_w)
                  target_h     = int(bm_image.height() * resize_ratio)
              else:
                  resize_ratio = self.input_h / bm_image.height()
                  target_w     = int(bm_image.width() * resize_ratio)
                  target_h     = int(self.input_h)
                  
              pad = sail.PaddingAtrr()
              offset_x = 0 if target_w >= target_h  else int((self.input_w  - target_w) / 2)
              offset_y = 0 if target_w <= target_h  else int((self.input_h  - target_h) / 2)
              pad.set_stx(offset_x)
              pad.set_sty(offset_y)
              pad.set_w(target_w)
              pad.set_h(target_h)
              # padding color grey
              pad.set_r(0)
              pad.set_g(0)
              pad.set_b(0)
              # tmp = self.bmcv.vpp_resize_padding(input_img, self.size_w, self.size_h, pad)
              tmp = self.bmcv.crop_and_resize_padding(bm_image, 0, 0, 
                                                      bm_image.width(), bm_image.height(),
                                                      self.input_w, self.input_h,
                                                      pad)
          else:
              tmp = self.bmcv.vpp_resize(bm_image, self.input_w, self.input_h)
          resized_list.append(tmp)
      
      if len(resized_list) == 1:
        tmp = resized_list[0]
      else:
        tmp = sail.BMImageArray4D()
        for i in range(len(resized_list)):
          tmp[i] = resized_list[i].data()
      self.bmcv.convert_to(tmp, self.input_bmimage, ((self.ab[0], self.ab[1]), \
                                          (self.ab[2], self.ab[3]), \
                                          (self.ab[4], self.ab[5])))

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
   raise NotImplementedError
 
  def bmimage_to_cvmat(self, bmimage):
    result_tensor = self.bmcv.bm_image_to_tensor(bmimage)
    result_numpy  = result_tensor.asnumpy()
    np_array_temp = result_numpy[0]
    np_array_t    = np.transpose(np_array_temp, [1, 2, 0])
    image         = np.array(np_array_t, dtype=np.uint8)
    b,g,r = cv2.split(image)
    image = cv2.merge([b,g,r])
    return image
    
  def run(self, bmimage):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    start_time = time.time()

    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    scale_start_time = time.time()
    # resize&normalize
    self.pre_process(bmimage)
    # copy to tensor
    self.bmcv.bm_image_to_tensor(self.input_bmimage, self.input)
      
    # raw image shape
    image_shape = []
    if self.output_shape[0] == 1:
        # batch size = 1
        image_shape.append((bmimage.height(), bmimage.width()))
    else:
        for i in range(4):
            image_shape.append((bmimage[i].height(), bmimage[i].width()))
    
    logging.info('tensor shape {}, input bmimg HxW {}'.format(self.input.shape(), image_shape))
    pre_process_time = time.time()
    pre_time += pre_process_time - scale_start_time
    
    pred_hms, pred_whs, pred_off, forward_time = self.process()

    net_time += forward_time - pre_process_time
    decode_time = time.time()
    # decode
    outputs = self.decode_bbox(pred_hms, pred_whs, pred_off)
      
    dec_time += decode_time - forward_time

    
    dets = self.post_process(outputs, image_shape)

    post_process_time = time.time()
    post_time += post_process_time - decode_time

    
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    return dets
    # if self.opt.debug >= 1:
    #   self.show_results(debugger, image, results)
    
    # return {'results': detections, 'tot': tot_time, 'load': load_time,
    #         'pre': pre_time, 'net': net_time, 'dec': dec_time,
    #         'post': post_time, 'merge': merge_time}