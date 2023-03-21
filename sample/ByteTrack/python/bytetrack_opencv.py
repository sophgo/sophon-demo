# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
import numpy as np
from byte_tracker import BYTETracker
import sophon.sail as sail
import os
import copy
import time
import argparse
import cv2
import logging
logging.basicConfig(level=logging.INFO)


def pre_process(image, input_size, mean, std):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(dtype=np.float32)
    padded_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    padded_img = padded_img.transpose((2, 0, 1))

    return padded_img, r


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

        self.output_dtype = self.engine.get_output_dtype(
            self.graph_name, self.output_name)
        self.output_shape = self.engine.get_output_shape(
            self.graph_name, self.output_name)
        self.output_scale = self.engine.get_output_scale(
            self.graph_name, self.output_name)
        self.input_shape = self.engine.get_input_shape(
            self.graph_name, self.input_name)

        self.batch_size, self.c, self.height, self.width = self.input_shape
        self.input_size = tuple((self.height, self.width))

        # init time
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.track_time = 0.0

    def _pre_process(self, image):
        start_time = time.time()

        image_info = {'id': 0}

        image_info['image'] = copy.deepcopy(image)
        image_info['width'] = image.shape[1]
        image_info['height'] = image.shape[0]
        preprocessed_image, ratio = pre_process(
            image,
            self.input_size,
            self.rgb_means,
            self.std
        )
        image_info['ratio'] = ratio

        input_np = np.ones([self.batch_size, 3, self.height, self.width])*114.0
        input_np[0] = preprocessed_image

        self.preprocess_time += time.time() - start_time
        return input_np, image_info

    def inference(self, input_np):
        start_time = time.time()

        output_npy = self.engine.process(
            self.graph_name, {self.input_name: input_np})[self.output_name]

        self.inference_time += time.time() - start_time
        return output_npy

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
        default='../python/results/bytetrack_opencv',
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


def get_id_color(index):
    temp_index = abs(int(index)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def draw_tracking_info(
    image,
    tlwhs,
    ids,
    scores,
    frame_id=0,
    elapsed_time=0.,
):
    text_scale = 1.5
    text_thickness = 2
    line_thickness = 2

    text = 'frame: %d ' % (frame_id)
    text += 'elapsed time: %.0fms ' % (elapsed_time * 1000)
    text += 'num: %d' % (len(tlwhs))
    cv2.putText(
        image,
        text,
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 255, 0),
        thickness=text_thickness,
    )

    for index, tlwh in enumerate(tlwhs):
        x1, y1 = int(tlwh[0]), int(tlwh[1])
        x2, y2 = x1 + int(tlwh[2]), y1 + int(tlwh[3])

        color = get_id_color(ids[index])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # text = str(ids[index]) + ':%.2f' % (scores[index])
        text = str(ids[index])
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN,
                    text_scale, (0, 0, 0), text_thickness + 3)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN,
                    text_scale, (255, 255, 255), text_thickness)
    return image


if __name__ == '__main__':
    args = get_args()

    byte_tracker = ByteTracker(args)
    batch_size = byte_tracker.batch_size
    net_w = byte_tracker.width
    net_h = byte_tracker.height

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    print("TPU: {}".format(args.device_id))
    print("Batch Size: {}".format(batch_size))
    print("Network Input width: {}".format(net_w))
    print("Network Input height: {}".format(net_h))
    print("Save Path:{}".format(save_path))

    output_result = []

    # time init
    overall_time = 0.0
    start_overall = time.time()
    frame_num = 0

    if args.is_video:
        video_path = args.file_name
        output_video = args.output_video

        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        logging.info("Test video.........")
        save_result_name = video_path.split(
            '/')[-1].split('.')[0]+'_'+args.bmodel.split('/')[-1].split('.')[0]+'_py.txt'
        save_result_name = os.path.join(save_path, save_result_name)

        if output_video:
            save_video_path = os.path.join(
                save_path, video_path.split("/")[-1])
            logging.info(f"video save path is {save_path}")

            video_writer = cv2.VideoWriter(
                save_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (int(width), int(height)),
            )

        frame_id = 1
        while True:
            logging.info("Test No.{} frame".format(
                frame_id))
            start_time_elapsed = time.time()

            ret, frame = cap.read()
            if not ret:
                break
            debug_image = copy.deepcopy(frame)

            # Pre processed
            input_np, image_info = byte_tracker._pre_process(frame)

            # Inference
            output_npy = byte_tracker.inference(input_np)

            # Post processed
            dets = byte_tracker._post_process(output_npy, image_info)
            # print("Detection box", len(dets))

            # track
            bboxes, ids, scores = byte_tracker._tracker_update(
                dets,
                image_info,
            )

            elapsed_time = time.time() - start_time_elapsed

            # output video
            if output_video:
                debug_image = draw_tracking_info(
                    debug_image,
                    bboxes,
                    ids,
                    scores,
                    frame_id,
                    elapsed_time,
                )
                video_writer.write(debug_image)
            logging.info(
                'frame {}/{} ({:.2f} ms)'.format(frame_id, int(frame_count),
                                                 elapsed_time * 1000), )

            # save results
            output_result.append((frame_id, bboxes, ids))

            frame_id += 1

        frame_num = frame_id - 1

    else:
        image_path = args.file_name
        logging.info("Test image.........")

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

        logging.info("Image nums:{}".format(len(image_list)))
        frame_id = 1

        for index in range(len(image_list)):
            logging.info("Test {}/{} image".format(
                index+1, len(image_list)))

            start_time_elapsed = time.time()

            frame = cv2.imread(image_list[index])
            debug_image = copy.deepcopy(frame)
            # Pre processed
            input_np, image_info = byte_tracker._pre_process(frame)

            # Inference
            output_npy = byte_tracker.inference(input_np)

            # Post processed
            dets = byte_tracker._post_process(output_npy, image_info)

            # track
            bboxes, ids, scores = byte_tracker._tracker_update(
                dets,
                image_info,
            )

            elapsed_time = time.time() - start_time_elapsed

            # save results
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
    detector_preprocess_time = byte_tracker.preprocess_time / frame_num
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
