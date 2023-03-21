# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
from byte_tracker import BYTETracker
import sophon.sail as sail
import numpy as np
import os
import time
import argparse
import logging
logging.basicConfig(level=logging.INFO)


def process_padding_BMImage(input: sail.BMImage, bmcv: sail.Bmcv, image_w, image_h, resize_w, resize_h):

    min_radio = min(resize_h / image_h, resize_w / image_w)

    temp_resize_w = int(image_w * min_radio)
    temp_resize_h = int(image_h * min_radio)

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
        0, 0, image_w, image_h,
        resize_w, resize_h, paddingatt)

    return output_temp, min_radio


def process_padding_BMImage_tpu(input: sail.BMImage, bmcv: sail.Bmcv, image_w, image_h, resize_w, resize_h):
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
        0, 0, image_w, image_h,
        resize_w, resize_h, paddingatt)

    return output_temp, min_radio


def getTensors(decoder: sail.Decoder, handle: sail.Handle, bmcv: sail.Bmcv, batch_size, video_w, video_h, resize_w, resize_h, alpha_beta, dtype):
    img = decoder.read(handle)      # BMImage
    output_temp = sail.BMImage(
        handle, resize_h, resize_w, sail.FORMAT_BGR_PLANAR, dtype)

    output_image, min_radio = process_padding_BMImage(
        img, bmcv, video_w, video_h, resize_w, resize_h)
    bmcv.convert_to(output_image, output_temp, alpha_beta)
    output_tensor = bmcv.bm_image_to_tensor(output_temp)
    # 归一化 模型对齐
    old_data = output_tensor.asnumpy()
    old_data /= 255
    output_tensor = sail.Tensor(handle, old_data)

    return img, output_image, output_tensor, min_radio


def post_process(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

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


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
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
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


class ByteTracker(object):
    def __init__(self, args):
        self.args = args

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.tracker = BYTETracker(args, frame_rate=30)

        self.engine = sail.Engine(args.bmodel, 0, sail.IOMode.SYSIO)
        self.handle = self.engine.get_handle()
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_name = self.engine.get_input_names(self.graph_name)[0]
        self.output_name = self.engine.get_output_names(self.graph_name)[0]

        self.input_dtype = self.engine.get_input_dtype(
            self.graph_name, self.input_name)
        self.input_shape = self.engine.get_input_shape(
            self.graph_name, self.input_name)
        self.input_sacle = self.engine.get_input_scale(
            self.graph_name, self.input_name)

        self.dtype = sail.DATA_TYPE_EXT_1N_BYTE
        if self.input_dtype == sail.BM_FLOAT32:
            self.dtype = sail.DATA_TYPE_EXT_FLOAT32

        self.output_dtype = self.engine.get_output_dtype(
            self.graph_name, self.output_name)
        self.output_shape = self.engine.get_output_shape(
            self.graph_name, self.output_name)
        self.output_scale = self.engine.get_output_scale(
            self.graph_name, self.output_name)

        self.batch_size, self.c, self.height, self.width = self.input_shape
        self.output_tensor = sail.Tensor(
            self.handle, self.output_shape, self.output_dtype, True, True)

        self.input_size = tuple((self.height, self.width))

        # init time
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.track_time = 0.0

    def inference(self, input_tensor):
        start_time = time.time()
        self.engine.process(self.graph_name, {self.input_name: input_tensor}, {
                            self.output_name: self.output_tensor})

        self.inference_time += time.time() - start_time
        return self.output_tensor.asnumpy()

    def _post_process(self, result, image_info):
        start_time = time.time()

        predictions = post_process(
            result,
            self.input_size,
            p6=self.args.with_p6,
        )
        predictions = predictions[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= image_info['ratio']

        dets = multiclass_nms(
            boxes_xyxy,
            scores,
            nms_thr=self.args.nms_th,
            score_thr=self.args.score_th,
        )

        self.postprocess_time += time.time() - start_time
        return dets

    def _tracker_update(self, dets, image_info):
        start_time = time.time()

        online_targets = []
        if dets is not None:
            online_targets = self.tracker.update(
                dets[:, :-1],
                [image_info['height'], image_info['width']],
                [image_info['height'], image_info['width']],
            )

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for online_target in online_targets:
            tlwh = online_target.tlwh
            track_id = online_target.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(track_id)
                online_scores.append(online_target.score)

        self.track_time += time.time() - start_time
        return online_tlwhs, online_ids, online_scores


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_video',
        type=int,
        default=0,  # if output 1 else 0
        help="whether output video file?"
    )
    parser.add_argument(
        '--is_video',
        type=int,
        default=0,  # if video 1 else 0
        help="input is video?"
    )
    parser.add_argument(
        '--bmodel',
        type=str,
        default='../models/BM1684/bytetrack_s_fp32_1b.bmodel',
    )
    parser.add_argument(
        '--file_name',
        type=str,
        # default='../data/video/sample.mp4',
        default='../datasets/MOT15/ADL-Rundle-6/img1',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='../python/results/bytetrack_bmcv',
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.7,
    )
    parser.add_argument(
        '--with_p6',
        action='store_true',
        help='Whether your model uses p6 in FPN/PAN.',
    )
    parser.add_argument(
        '--device_id',
        type=int,
        default=0,
    )

    # tracking args
    parser.add_argument(
        '--track_thresh',
        type=float,
        default=0.5,
        help='tracking confidence threshold',
    )
    parser.add_argument(
        '--track_buffer',
        type=int,
        default=30,
        help='the frames for keep lost tracks',
    )
    parser.add_argument(
        '--match_thresh',
        type=float,
        default=0.8,
        help='matching threshold for tracking',
    )
    parser.add_argument(
        '--min-box-area',
        type=float,
        default=10,
        help='filter out tiny boxes',
    )
    parser.add_argument(
        '--mot20',
        dest='mot20',
        default=False,
        action='store_true',
        help='test mot20.',
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    save_path = args.save_path

    os.makedirs(save_path, exist_ok=True)

    byte_tracker = ByteTracker(args)

    handle = byte_tracker.handle
    bmcv = sail.Bmcv(handle)
    batch_size = byte_tracker.batch_size
    net_w = byte_tracker.width
    net_h = byte_tracker.height
    alpha_beta = (byte_tracker.input_sacle,
                  0), (byte_tracker.input_sacle, 0), (byte_tracker.input_sacle, 0)

    print("TPU: {}".format(args.device_id))
    print("Batch Size: {}".format(batch_size))
    print("Network Input width: {}".format(net_w))
    print("Network Input height: {}".format(net_h))
    print("Save Path:{}".format(save_path))

    output_result = []

    # time init
    overall_time = 0.0
    preprocess_time = 0.0
    start_overall = time.time()
    frame_num = 0

    if args.is_video:
        video_path = args.file_name

        decoder = sail.Decoder(video_path, True, args.device_id)
        _, _, ost_h, ost_w = decoder.get_frame_shape()

        logging.info("Test video.........")
        save_result_name = video_path.split(
            '/')[-1].split('.')[0]+'_'+args.bmodel.split('/')[-1].split('.')[0]+'_py.txt'
        save_result_name = os.path.join(save_path, save_result_name)

        frame = sail.BMImage()
        frame_id = 1
        while True:
            logging.info("Test No.{} frame".format(
                frame_id))
            # Pre processed
            start_time_pre = time.time()

            ret = decoder.read(handle, frame)

            if ret != 0:
                logging.info("read video end.")
                break  # BMImage

            output_temp = sail.BMImage(
                handle, net_h, net_w, sail.FORMAT_BGR_PLANAR, byte_tracker.dtype)

            output_image, min_ratio = process_padding_BMImage(
                frame, bmcv, ost_w, ost_h, net_w, net_h)
            bmcv.convert_to(output_image, output_temp, alpha_beta)
            input_tensor = bmcv.bm_image_to_tensor(output_temp)
            # 归一化 模型对齐
            old_data = input_tensor.asnumpy()
            old_data /= 255
            input_tensor = sail.Tensor(handle, old_data)
            image_info = {'id': 0}
            image_info['width'] = ost_w
            image_info['height'] = ost_h
            image_info['ratio'] = min_ratio

            preprocess_time += time.time() - start_time_pre

            # Inference
            output_npy = byte_tracker.inference(input_tensor)

            # Post processed
            dets = byte_tracker._post_process(output_npy, image_info)
            # print("Detection box", len(dets))

            bboxes, ids, scores = byte_tracker._tracker_update(
                dets,
                image_info,
            )

            # save results
            output_result.append((frame_id, bboxes, ids))

            frame_id += 1
            frame_num += 1

    else:
        image_path = args.file_name
        if image_path[-1] == '/':
            image_path = image_path[0:-1]
        save_result_name = image_path.split(
            "/")[-1]+"_"+args.bmodel.split("/")[-1].split(".")[0]+"_py.txt"
        save_result_name = os.path.join(save_path, save_result_name)

        file_list = os.listdir(image_path)
        image_list = []
        for file_name in file_list:
            ext_name = os.path.splitext(file_name)[-1]
            if ext_name in ['.jpg', '.png', '.jpeg', '.bmp', '.JPEG', '.JPG', '.BMP']:
                image_list.append(os.path.join(image_path, file_name))
        if len(image_list) == 0:
            print("Can not find any pictures!")
            exit(1)

        frame_id = 1
        for index in range(len(image_list)):
            logging.info("Test {}/{} image".format(
                index+1, len(image_list)))

            decoder = sail.Decoder(image_list[index], True, args.device_id)
            img = decoder.read(handle)
            img_bgr = sail.BMImage(handle, img.height(), img.width(
            ), sail.FORMAT_BGR_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
            bmcv.convert_format(img, img_bgr)

            # Pre processed
            start_time_pre = time.time()
            output_image, min_ratio = process_padding_BMImage_tpu(
                img, bmcv, img.width(), img.height(), net_w, net_h)
            # bmcv.imwrite('001.jpg',output_image)
            output_temp = sail.BMImage(
                handle, net_h, net_w, sail.FORMAT_BGR_PLANAR, byte_tracker.dtype)
            bmcv.convert_to(output_image, output_temp, alpha_beta)
            input_tensor = bmcv.bm_image_to_tensor(output_temp)
            # align
            old_data = input_tensor.asnumpy()
            old_data /= 255.0
            input_tensor = sail.Tensor(handle, old_data)

            image_info = {'id': 0}
            image_info['width'] = img.width()
            image_info['height'] = img.height()
            image_info['ratio'] = min_ratio

            preprocess_time += time.time() - start_time_pre

            # Inference
            output_npy = byte_tracker.inference(input_tensor)

            # Post processed
            dets = byte_tracker._post_process(output_npy, image_info)

            # Track
            bboxes, ids, scores = byte_tracker._tracker_update(
                dets,
                image_info,
            )

            output_result.append((frame_id, bboxes, ids))

            frame_id += 1

        frame_num = frame_id - 1

    logging.info(
        "------------------------ByteTrack test-----------------------------")
    overall_time += time.time() - start_overall
    logging.info("frame_num:{}".format(frame_num))
    # calculate speed
    overall_time = overall_time / frame_num
    logging.info("overall_time(ms): {:.2f}".format(overall_time * 1000))
    logging.info(
        "------------------Detector Predict Time Info ----------------------")
    detector_preprocess_time = preprocess_time / frame_num
    detector_inference_time = byte_tracker.inference_time / frame_num
    detector_postprocess_time = byte_tracker.postprocess_time / frame_num
    logging.info("preprocess_time(ms): {:.2f}".format(
        detector_preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(
        detector_inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(
        detector_postprocess_time * 1000))
    logging.info(
        "-------------------------------------------------------------------")
    logging.info(
        "------------------ByteTrack Tracker Time Info ----------------------")
    bytetrack_track_time = byte_tracker.track_time / frame_num
    logging.info("track_time(ms): {:.2f}".format(
        bytetrack_track_time * 1000))
    logging.info(
        "-------------------------------------------------------------------")

    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(save_result_name, 'w+') as f:
        for frame_id, tlwhs, track_ids in output_result:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(
                    x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logging.info('Save results to {}'.format(save_result_name))
