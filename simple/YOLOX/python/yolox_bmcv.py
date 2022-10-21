import numpy as np
import sophon.sail as sail
import argparse
import os
import time

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

def process_padding_BMImage(input:sail.BMImage, bmcv:sail.Bmcv, image_w, image_h, resize_w, resize_h):
    scale_w = float(resize_w) / image_w
    scale_h = float(resize_h) / image_h

    temp_resize_w = resize_w
    temp_resize_h = resize_h

    min_radio = scale_h

    if scale_w < scale_h:
        temp_resize_h = int(image_h*scale_w)
        min_radio = scale_w
    else:
        temp_resize_w = int(image_w*scale_h)

    paddingatt = sail.PaddingAtrr()   
    paddingatt.set_stx(0)
    paddingatt.set_sty(0)
    paddingatt.set_w(temp_resize_w)
    paddingatt.set_h(temp_resize_h)
    paddingatt.set_r(114)
    paddingatt.set_g(114)
    paddingatt.set_b(114)

    output_temp = bmcv.vpp_crop_and_resize_padding(
        input,
        0,0,image_w,image_h,
        resize_w,resize_h,paddingatt)

    return output_temp,min_radio

def process_padding_BMImage_tpu(input:sail.BMImage, bmcv:sail.Bmcv, image_w, image_h, resize_w, resize_h):
    scale_w = float(resize_w) / image_w
    scale_h = float(resize_h) / image_h

    temp_resize_w = resize_w
    temp_resize_h = resize_h

    min_radio = scale_h

    if scale_w < scale_h:
        temp_resize_h = int(image_h*scale_w)
        min_radio = scale_w
    else:
        temp_resize_w = int(image_w*scale_h)

    paddingatt = sail.PaddingAtrr()   
    paddingatt.set_stx(0)
    paddingatt.set_sty(0)
    paddingatt.set_w(temp_resize_w)
    paddingatt.set_h(temp_resize_h)
    paddingatt.set_r(114)
    paddingatt.set_g(114)
    paddingatt.set_b(114)

    output_temp = bmcv.crop_and_resize_padding(
        input,
        0,0,image_w,image_h,
        resize_w,resize_h,paddingatt)

    return output_temp,min_radio


def getTensors(decoder:sail.Decoder, handle:sail.Handle,bmcv:sail.Bmcv, batch_size,video_w, video_h, resize_w, resize_h, alpha_beta, dtype):
    if batch_size == 1:
        img = decoder.read(handle)      # BMImage
        output_temp = sail.BMImage(handle, resize_h,resize_w,sail.FORMAT_BGR_PLANAR, dtype)
    elif batch_size == 4:
        img = sail.BMImageArray4D()     # BMImageArray
        output_temp = sail.BMImageArray4D(handle, resize_h,resize_w,sail.FORMAT_BGR_PLANAR, dtype)
        for idx in range(4):
            decoder.read_(handle,img[idx])
    else:
        print("Error Batch Size!")
        exit(1)
    
    output_image,min_radio=process_padding_BMImage(img,bmcv,video_w, video_h,resize_w, resize_h)
    bmcv.convert_to(output_image, output_temp, alpha_beta)
    output_tensor = bmcv.bm_image_to_tensor(output_temp)
    return img, output_image, output_tensor, min_radio




class Detector(object):
    def __init__(self, bmodel_path, tpu_id):
        self.engine = sail.Engine(bmodel_path,tpu_id,sail.IOMode.SYSO)
        self.handle = self.engine.get_handle()
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_name = self.engine.get_input_names(self.graph_name)[0]
        self.output_name = self.engine.get_output_names(self.graph_name)[0]

        self.input_dtype = self.engine.get_input_dtype(self.graph_name,self.input_name)
        self.input_shape = self.engine.get_input_shape(self.graph_name,self.input_name)
        self.input_sacle = self.engine.get_input_scale(self.graph_name,self.input_name)

        self.dtype = sail.DATA_TYPE_EXT_1N_BYTE
        if self.input_dtype == sail.BM_FLOAT32:
            self.dtype = sail.DATA_TYPE_EXT_FLOAT32

        self.output_dtype = self.engine.get_output_dtype(self.graph_name,self.output_name)
        self.output_shape = self.engine.get_output_shape(self.graph_name,self.output_name)
        self.output_scale = self.engine.get_output_scale(self.graph_name,self.output_name)

        self.batch_size, self.c, self.height, self.width = self.input_shape

        self.output_tensor = sail.Tensor(self.handle, self.output_shape, self.output_dtype, True, True)


    def inference(self,input_tensor):
        self.engine.process(self.graph_name, {self.input_name:input_tensor},{self.output_name:self.output_tensor})
        return self.output_tensor.asnumpy()

    def yolox_postprocess(self, outputs, input_w, input_h, p6=False):
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [input_h // stride for stride in strides]
        wsizes = [input_w // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs
    
    def get_detectresult(self,predictions,dete_threshold,nms_threshold):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_threshold, score_thr=dete_threshold)
        return dets


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Demo for YOLOX")
    parse.add_argument('--is_video',default=0,type=int,help="input is video?")
    parse.add_argument('--loops', default=16, type=int,help="only for video")
    parse.add_argument('--file_name',default="../data/image/val2017",type=str,help="video name or image_path")
    # parse.add_argument('--file_name',default="/workspace/test/YOLOX/datasets/ost_data",type=str,help="video name or image_path")
    # parse.add_argument('--bmodel_path',default="/workspace/test/YOLOX/models/yolox_s/int8model_bs4/compilation.bmodel",type=str)
    parse.add_argument('--bmodel_path',default="../data/models/BM1684X/yolox_s_int8_4b.bmodel",type=str)
    parse.add_argument('--device_id',default=0,type=int)
    parse.add_argument('--detect_threshold',default=0.25,type=float,required=False)
    parse.add_argument('--nms_threshold',default=0.45,type=float, required=False)
    parse.add_argument('--save_path', default='yolox_save', type=str)

    opt= parse.parse_args()

    yolox = Detector(opt.bmodel_path,opt.device_id)
    handle = yolox.handle
    bmcv = sail.Bmcv(handle)
    batch_size = yolox.batch_size
    net_w = yolox.width
    net_h = yolox.height
    alpha_beta = (yolox.input_sacle,0),(yolox.input_sacle,0),(yolox.input_sacle,0)
    save_path = opt.save_path

    mkdir(save_path)

    print("TPU: {}".format(opt.device_id))
    print("Batch Size: {}".format(batch_size))
    print("Network Input width: {}".format(net_w))
    print("Network Input height: {}".format(net_h))
    print("Save Path:{}".format(save_path))

    output_result = {}

    if opt.is_video:
        save_result_name = opt.file_name.split('/')[-1].split('.')[0]+'_'+opt.bmodel_path.split('/')[-1].split('.')[0]+'_py.txt'
        save_result_name = os.path.join(save_path,save_result_name)
        decoder = sail.Decoder(opt.file_name, True, opt.device_id)
        _,_,ost_h,ost_w = decoder.get_frame_shape()

        for i in range(opt.loops):
            ost_image,resize_image,input_tensor,min_ratio= getTensors(decoder, handle, bmcv, batch_size,
                ost_w,ost_h, net_w, net_h, alpha_beta, yolox.dtype)
            
            start_time = time.time()
            output_npy = yolox.inference(input_tensor)
            end_time = time.time()
            predictions = yolox.yolox_postprocess(output_npy, net_w, net_h)
            print("Inference time use:{:.2f} ms, Batch size:{}, avg fps:{:.1f}".format((end_time-start_time)*1000,\
                    batch_size,batch_size/(end_time-start_time)))
            if batch_size == 1:
                dete_boxs = yolox.get_detectresult(predictions[0],opt.detect_threshold, opt.nms_threshold)
                if dete_boxs is not None:
                    dete_boxs[:,0] /= min_ratio
                    dete_boxs[:,1] /= min_ratio
                    dete_boxs[:,2] /= min_ratio
                    dete_boxs[:,3] /= min_ratio
                    for dete_box in dete_boxs:
                        bmcv.rectangle(ost_image, int(dete_box[0]), int(dete_box[1]), 
                            int(dete_box[2]-dete_box[0]), int(dete_box[3]-dete_box[1]), (0, 0, 255), 4)
                # bmcv.imwrite(os.path.join(save_path,"frame_{}_device_{}.jpg".format(i,opt.device_id)),ost_image)

                image_name_temp = "frame_{}".format(i)
                output_result.update({image_name_temp:dete_boxs})
                # bmcv.imwrite("loop_{}_resize.jpg".format(i),resize_image)
            else:
                for image_idx in range(len(resize_image)):
                    dete_boxs = yolox.get_detectresult(predictions[image_idx],opt.detect_threshold, opt.nms_threshold)
                    if dete_boxs is not None:
                        dete_boxs[:,0] /= min_ratio
                        dete_boxs[:,1] /= min_ratio
                        dete_boxs[:,2] /= min_ratio
                        dete_boxs[:,3] /= min_ratio

                        for dete_box in dete_boxs:
                            bmcv.rectangle_(ost_image[image_idx], int(dete_box[0]), int(dete_box[1]), 
                                int(dete_box[2]-dete_box[0]), int(dete_box[3]-dete_box[1]), (255, 255, 0), 4)
                
                    #bmcv.imwrite_(os.path.join(save_path,"frame_{}_device_{}.jpg".format(i*batch_size+image_idx,opt.device_id)),ost_image[image_idx])
                    
                    image_name_temp = "frame_{}".format(i*batch_size+image_idx)
                    output_result.update({image_name_temp:dete_boxs})                    
                    # bmcv.imwrite_("loop_{}_{}_resize.jpg".format(i,image_idx),resize_image[image_idx])
    else:
        image_path = opt.file_name
        if image_path[-1] == '/':
            image_path = image_path[0:-1]
        save_result_name = image_path.split("/")[-1]+"_"+opt.bmodel_path.split("/")[-1].split(".")[0]+"_py.txt"
        save_result_name = os.path.join(save_path,save_result_name)

        file_list = os.listdir(image_path)
        image_list = []
        for file_name in file_list:
            ext_name = os.path.splitext(file_name)[-1]
            if ext_name in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
                image_list.append(os.path.join(image_path,file_name))
        if len(image_list) == 0:
            print("Can not find any pictures!")
            exit(1)
        if batch_size == 1:
            for image_name in image_list:
                decoder = sail.Decoder(image_name, True, opt.device_id)
                img = decoder.read(handle)
                img_bgr = sail.BMImage(handle, img.height(), img.width(),sail.FORMAT_BGR_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
                bmcv.convert_format(img,img_bgr)
                output_image,min_ratio = process_padding_BMImage_tpu(img, bmcv, img.width(), img.height(), net_w, net_h)
                #bmcv.imwrite('001.jpg',output_image)
                output_temp = sail.BMImage(handle, net_h, net_w,sail.FORMAT_BGR_PLANAR, yolox.dtype)
                bmcv.convert_to(output_image, output_temp, alpha_beta)
                input_tensor = bmcv.bm_image_to_tensor(output_temp)
                start_time = time.time()
                output_npy = yolox.inference(input_tensor)
                end_time = time.time()
                predictions = yolox.yolox_postprocess(output_npy, net_w, net_h)
                dete_boxs = yolox.get_detectresult(predictions[0],opt.detect_threshold, opt.nms_threshold)
                print("Inference time use:{:.2f} ms, Batch size:{}, avg fps:{:.1f}".format((end_time-start_time)*1000,\
                    batch_size,batch_size/(end_time-start_time)))
                if dete_boxs is not None:
                    dete_boxs[:,0] /= min_ratio
                    dete_boxs[:,1] /= min_ratio
                    dete_boxs[:,2] /= min_ratio
                    dete_boxs[:,3] /= min_ratio
                    for dete_box in dete_boxs:
                        bmcv.rectangle(img_bgr, int(dete_box[0]), int(dete_box[1]), 
                            int(dete_box[2]-dete_box[0]), int(dete_box[3]-dete_box[1]), (0, 0, 255), 2)
                #bmcv.imwrite(os.path.join(save_path,"{}".format(image_name.split('/')[-1])),img_bgr)
                
                image_name_temp = image_name.split('/')[-1]
                output_result.update({image_name_temp:dete_boxs})

        elif batch_size == 4:
            if len(image_list)%batch_size != 0:
                append_count = batch_size - (len(image_list)%batch_size)
                for idx in range(append_count):
                    image_list.append(image_list[0])
            for idx in range(0,len(image_list),batch_size):
                imgarray_bgr = sail.BMImageArray4D(handle, net_h, net_w,sail.FORMAT_BGR_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
                output_temp = sail.BMImageArray4D(handle, net_h, net_w,sail.FORMAT_BGR_PLANAR, yolox.dtype)
                ratio_list = []
                ost_image = []
                start_time = time.time()
                for i in range(4):
                    decoder = sail.Decoder(image_list[idx+i], True, opt.device_id)
                    img = decoder.read(handle)
                    img_bgr = sail.BMImage(handle, img.height(), img.width(),sail.FORMAT_BGR_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
                    bmcv.convert_format(img,img_bgr)
                    output_image,min_ratio = process_padding_BMImage_tpu(img, bmcv, img.width(), img.height(), net_w, net_h)
                    ratio_list.append(min_ratio)
                    imgarray_bgr.copy_from(i,output_image)
                    ost_image.append(img_bgr)
                
                bmcv.convert_to(imgarray_bgr, output_temp, alpha_beta)
                input_tensor = bmcv.bm_image_to_tensor(output_temp)
                start_time = time.time()
                output_npy = yolox.inference(input_tensor)
                end_time = time.time()
                print("Inference time use:{:.2f} ms, Batch size:{}, avg fps:{:.1f}".format((end_time-start_time)*1000,\
                    batch_size,batch_size/(end_time-start_time)))
                predictions = yolox.yolox_postprocess(output_npy, net_w, net_h)

                for image_idx in range(4):
                    dete_boxs = yolox.get_detectresult(predictions[image_idx],opt.detect_threshold, opt.nms_threshold)
                    if dete_boxs is not None:
                        dete_boxs[:,0] /= ratio_list[image_idx]
                        dete_boxs[:,1] /= ratio_list[image_idx]
                        dete_boxs[:,2] /= ratio_list[image_idx]
                        dete_boxs[:,3] /= ratio_list[image_idx]

                        for dete_box in dete_boxs:
                            bmcv.rectangle(ost_image[image_idx], int(dete_box[0]), int(dete_box[1]), 
                                int(dete_box[2]-dete_box[0]), int(dete_box[3]-dete_box[1]), (255, 255, 0), 4)

                    #bmcv.imwrite(os.path.join(save_path,"{}".format(image_list[idx+image_idx].split('/')[-1])),ost_image[image_idx])
                    
                    image_name_temp = image_list[idx+image_idx].split('/')[-1]
                    output_result.update({image_name_temp:dete_boxs})
        else:
            print("Error batch size: {}".format(batch_size))
            exit(1)
        
    with open(save_result_name, "w+") as fp:
        for key, values in output_result.items():
            if values is None:
                continue
            for obj in values:
                line_0 = "[{}]\n".format(key)
                line_1 = "category={:.0f}\n".format(obj[5])
                line_2 = "score={:.2f}\n".format(obj[4])
                line_3 = "left={:.2f}\n".format(obj[0])
                line_4 = "top={:.2f}\n".format(obj[1])
                line_5 = "right={:.2f}\n".format(obj[2])
                line_6 = "bottom={:.2f}\n\n".format(obj[3])
                fp.write(line_0)
                fp.write(line_1)
                fp.write(line_2)
                fp.write(line_3)
                fp.write(line_4)
                fp.write(line_5)
                fp.write(line_6)
        fp.close()
            

                
            

    

