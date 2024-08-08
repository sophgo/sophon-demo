#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import cv2
import time
import json
import argparse
import numpy as np

from hrnet_pose.utils_hrnet import COCO_INSTANCE_CATEGORY_NAMES
from hrnet_pose.postprocess_hrnet import draw_pose

import logging
logging.basicConfig(level=logging.INFO)


def get_person_detection_boxes(detection_model, img_list, args, threshold):
    preds = detection_model(img_list)

    person_boxes_list = []
    person_scores_list = []

    start_time = time.time()

    for pred in preds:
        if args.use_cpu_opt:
            pred_column = pred[:, 4]
            pred_classes = np.array([COCO_INSTANCE_CATEGORY_NAMES[int(i)] for i in pred_column])
            pred_scores = pred[:, 5]

        else:
            pred_column = pred[:, 5]
            pred_classes = np.array([COCO_INSTANCE_CATEGORY_NAMES[int(i)] for i in pred_column])

            pred_scores = pred[:, 4]

        pred_boxes = pred[:, :4]

        if not pred_scores.size or np.max(pred_scores) < threshold:
            person_boxes_list.append([])
            person_scores_list.append([])
            continue

        pred_t = np.where(pred_scores > threshold)[0][-1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_classes = pred_classes[:pred_t + 1]
        pred_scores = pred_scores[:pred_t + 1]

        person_boxes = pred_boxes[pred_classes == 'person']
        person_scores = pred_scores[pred_classes == 'person']

        person_boxes_list.append(person_boxes.tolist())
        person_scores_list.append(person_scores.tolist())

    detection_model.postprocess_time += (time.time() - start_time)
    return person_boxes_list, person_scores_list


def save_coco_keypoints(keypoints, maxvals, image_name, box_score):
    image_id = int(image_name.split('.')[0].lstrip('0'))

    keypoints = np.squeeze(keypoints)
    maxvals = np.squeeze(maxvals)
    maxvals = np.expand_dims(maxvals, axis=1)

    mask = np.greater(maxvals, 0.2)
    if mask.sum() == 0:
        k_score = 0
    else:
        k_score = np.mean(maxvals[mask])

    keypoints = np.concatenate([keypoints, maxvals], axis=1)
    keypoints = np.reshape(keypoints, -1)

    # keypoints = [round(k, 2) for k in keypoints.tolist()]
    keypoints = keypoints.tolist()

    res = {"image_id": image_id,
           "category_id": 1,  # person
           "keypoints": keypoints,
           "score": box_score * k_score}
    return res


class yolov5_arg:
    def __init__(self, bmodel, dev_id, conf_thresh, nms_thresh, use_cpu_opt):
        self.bmodel = bmodel
        self.dev_id = dev_id
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.use_cpu_opt = use_cpu_opt

def main(args):
    """
    Use the object detection model to detect a single person in the image,
    then feed the single person into the human pose estimation model and record the pose estimation results.

    Parameters:
          args: Arguments.

    Returns:
    """
    if args.backend == 'opencv':
        from hrnet_pose.hrnet_opencv import HRNet
        from detector.yolov5.yolov5_opencv import YOLOv5
    # elif args.backend == 'bmcv':
    #     from hrnet_pose.hrnet_bmcv import HRNet
    #     from detector.yolov5.yolov5_bmcv import YOLOv5

    yolov5_args = yolov5_arg(args.detection_bmodel, args.dev_id, args.conf_thresh, args.nms_thresh, args.use_cpu_opt)

    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.pose_bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.pose_bmodel))
    if not os.path.exists(args.detection_bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.detection_bmodel))

    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_image_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_image_dir):
        os.mkdir(output_image_dir)

    key_points_results_dir = os.path.join(output_dir, 'keypoints_results_python.json')

    yolov5 = YOLOv5(yolov5_args)
    yolov5_batch_size = yolov5.batch_size
    yolov5.init()

    hrnet = HRNet(args)
    hrnet.init()

    decode_time = 0.0

    if os.path.isdir(args.input):

        image_list = []
        filename_list = []
        result_list = []

        count_det = 0

        count_pose = 0

        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:

                if os.path.splitext(filename)[-1].lower() not in ['.jpg', '.png', '.jpeg', '.bmp', '.webp']:
                    continue

                img_file = os.path.join(root, filename)
                count_det += 1
                logging.info("{}, img_file: {}".format(count_det, img_file))

                start_time = time.time()

                src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)

                if src_img is None:
                    logging.error("{} imdecode is None.".format(img_file))
                    continue

                if len(src_img.shape) != 3:
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                else:
                    if src_img.shape[2] != 3:
                        print("The model assumes input to be grayscale or RGB images and is designed to process RGB images. Please modify the conversion according to your actual situation.")
                        exit()

                decode_time += time.time() - start_time

                image_list.append(src_img)
                filename_list.append(filename)

                if (len(image_list) == yolov5_batch_size or count_det == len(filenames)) and len(image_list):

                    pred_boxes_list, pred_scores_list = get_person_detection_boxes(yolov5, image_list, args, args.person_thresh)

                    for i, (pred_boxes, pred_scores) in enumerate(zip(pred_boxes_list, pred_scores_list)):
                        
                        save_image_path = os.path.join(output_image_dir, filename_list[i])
                        for box, score in zip(pred_boxes, pred_scores):

                            count_pose += 1
                            pose_preds, maxvals = hrnet(args, image_list[i], box)

                            if len(pose_preds) >= 1:
                                res = save_coco_keypoints(pose_preds, maxvals, filename_list[i], score)
                                result_list.append(res)

                                for kpt in pose_preds:
                                    draw_pose(kpt, image_list[i])

                                cv2.imwrite(save_image_path, image_list[i])
                        # print('the result image has been saved as {}'.format(save_image_path))

                    image_list.clear()
                    filename_list.clear()

        with open(key_points_results_dir, 'w') as f:
            json.dump(result_list, f, indent=4)

    else:
        vidcap = cv2.VideoCapture()

        if not vidcap.open(args.input):
            raise Exception("can not open the video")

        save_video_path = os.path.join(output_dir, os.path.splitext(os.path.split(args.input)[1])[0] + '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        count_det = 0
        count_pose = 0

        frame_list = []

        end_flag = False

        while not end_flag:

            start_time = time.time()
            ret, image_bgr = vidcap.read()
            decode_time += time.time() - start_time

            if not ret or image_bgr is None:
                end_flag = True
            else:
                frame_list.append(image_bgr)

            if (len(frame_list) == yolov5_batch_size or end_flag) and len(frame_list):

                pred_boxes_list, _ = get_person_detection_boxes(yolov5, frame_list, args, threshold=args.person_thresh)

                for i, pred_boxes in enumerate(pred_boxes_list):

                    count_det += 1
                    print(f"{count_det}, det_persons_num: {len(pred_boxes)}")

                    for box in pred_boxes:

                        count_pose += 1

                        pose_preds, maxvals = hrnet(args, frame_list[i], box)

                        if len(pose_preds) >= 1:
                            for kpt in pose_preds:
                                draw_pose(kpt, frame_list[i])

                    out.write(frame_list[i])

                frame_list.clear()

        vidcap.release()
        out.release()
        logging.info("video has been saved as {}".format(save_video_path))

    logging.info("------------------ Predict Time Info ----------------------")

    decode_time = decode_time / count_det

    hrnet_preprocess_time = hrnet.preprocess_time / count_pose
    hrnet_inference_time = hrnet.inference_time / count_pose
    hrnet_postprocess_time = hrnet.postprocess_time / count_pose

    yolov5_preprocess_time = yolov5.preprocess_time / count_det
    yolov5_inference_time = yolov5.inference_time / count_det
    yolov5_postprocess_time = yolov5.postprocess_time / count_det

    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))

    logging.info("yolov5_preprocess_time(ms): {:.2f}".format(yolov5_preprocess_time * 1000))
    logging.info("yolov5_inference_time(ms): {:.2f}".format(yolov5_inference_time * 1000))
    logging.info("yolov5_postprocess_time(ms): {:.2f}".format(yolov5_postprocess_time * 1000))

    logging.info("hrnet_preprocess_time(ms): {:.2f}".format(hrnet_preprocess_time * 1000))
    logging.info("hrnet_inference_time(ms): {:.2f}".format(hrnet_inference_time * 1000))
    logging.info("hrnet_postprocess_time(ms): {:.2f}".format(hrnet_postprocess_time * 1000))


def argsparser():

    parser = argparse.ArgumentParser(prog=__file__)

    parser.add_argument('--backend', type=str, choices=['opencv', 'bmcv'], default='opencv', help='the backend to use for HRNet and YOLOv5')

    parser.add_argument('--input', type=str, default='./datasets/test_images', help='data root')

    parser.add_argument('--pose_bmodel', type=str, default='./models/BM1684X/hrnet_w32_256x192_int8.bmodel', help='path of pose estimation bmodel')

    parser.add_argument('--dev_id', type=int, default=0, help='device id for inference')

    parser.add_argument('--flip', type=bool, default=True, help='whether using flipped images')

    parser.add_argument('--person_thresh', type=float, default=0.8, help='person threshold')

    parser.add_argument('--detection_bmodel', type=str, default='./models/BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel', help='path of detection bmodel')

    parser.add_argument('--conf_thresh', type=float, default=0.001, help='confidence threshold')

    parser.add_argument('--nms_thresh', type=float, default=0.6, help='nms threshold')

    parser.add_argument('--use_cpu_opt', action="store_true", default=False, help='accelerate yolov5 cpu postprocess')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')