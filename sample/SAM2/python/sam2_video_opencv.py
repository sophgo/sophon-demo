import argparse
import os
import sys
import time
import re

import cv2
from pympler import asizeof
import numpy as np
from sam2_video_base import SAM2VideoBase
from video_utils import show_box, show_mask, show_points

np.random.seed(3)

def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return 0  

class VideoReader:

    def __init__(self, video_path, skip_num=1):
        self.video_type = 0
        self.skip_num = skip_num
        if video_path.split(".")[-1] in ["mp4", "avi"]:
            self.video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                print("Error: Could not open video.")
        else:
            self.video_type = 1
            self.video_path = video_path
            self.video = [
                p
                for p in os.listdir(video_path)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            self.video.sort(key=extract_number)
            tmp_img = cv2.imread(os.path.join(self.video_path, self.video[0]))
            self.video_height, self.video_width = tmp_img.shape[:2]

        self.frame_idx = 0

    def read(self):
        if self.video_type == 0:
            ret, frame = self.video.read()
            if ret:
                self.frame_idx += self.skip_num
                return True, frame
            else:
                return False, None
        else:
            if self.frame_idx < len(self.video):
                frame = cv2.imread(
                    os.path.join(self.video_path, self.video[self.frame_idx])
                )
                self.frame_idx += self.skip_num
                return True, frame
            else:
                return False, None

    def release(self):
        if self.video_type == 0:
            self.video.release()

    def shape(self):
        if self.video_type == 0:
            return self.video.get(cv2.CAP_PROP_FRAME_HEIGHT), self.video.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        return self.video_height, self.video_width


class SAM2Video(SAM2VideoBase):

    def __init__(
        self,
        dev_id,
        output_dir,
        image_encoder_path,
        image_decoder_path,
        memory_attention_path,
        memory_encoder_path,
        constant_path,
        save_results=True,
        image_size=1024,
        num_mask_mem=7,
        max_obj_ptrs_in_encoder=16,
    ):
        self.version = "1.0.0"
        super().__init__(
            dev_id=dev_id,
            image_encoder_path=image_encoder_path,
            image_decoder_path=image_decoder_path,
            memory_attention_path=memory_attention_path,
            memory_encoder_path=memory_encoder_path,
            constant_path=constant_path,
            num_mask_mem=num_mask_mem,
            max_obj_ptrs_in_encoder=max_obj_ptrs_in_encoder,
        )

        self.image_size = image_size
        self.inference_state = self.init_state()
        self.reset_state(self.inference_state)

        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

        self.save = save_results
        self.output_dir = output_dir
        self.run = True

        self.preprocess_time = 0
        self.inference_time = 0
        self.postporcess_time = 0

    def preprocess_frame(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img / 255.0
        img = img - self.img_mean
        img = img / self.img_std
        img = np.transpose(img, (2, 0, 1))
        return img

    def load_video(self, video_path):
        self.video = VideoReader(video_path)
        self.frame_idx = self.video.frame_idx
        self.video_height, self.video_width = self.video.shape()

    def video_preflight(self, points, labels, box=None):
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        _, out_obj_ids, out_mask_logits = self.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
            box=box,
        )

        self.propagate_in_video_preflight(self.inference_state)

    def inference(self, points, labels, box=None):
        
        while self.run:
            ret, frame = self.video.read()
            if not ret:
                break
            print(f"正在分割第 {self.frame_idx} 帧图像")
            preprocess_start_time = time.time()
            image = self.preprocess_frame(frame)
            self.preprocess_time += time.time() - preprocess_start_time

            self.append_image(self.inference_state, image)

            if self.frame_idx == 0:
                self.video_preflight(points, labels)

            inference_start_time = time.time()

            _, out_obj_ids, out_mask_logits = self.propagate_in_video(
                self.inference_state, frame_idx=self.frame_idx
            )
            self.inference_time += time.time() - inference_start_time

            postprocess_start_time = time.time()
            frame = show_mask(
                (out_mask_logits[0] > 0.0),
                frame,
                color=np.array([30, 144, 255]),
                obj_id=out_obj_ids[0],
            ).astype(np.uint8)
            self.postporcess_time = time.time() - postprocess_start_time

            if self.frame_idx == 0:
                frame = show_points(
                    points.astype(np.int64), labels.astype(np.int64), frame
                )
                frame = show_box(box, frame)

            if self.save:
                cv2.imwrite(
                    f"{self.output_dir}/video/frame_{self.frame_idx}.jpg", frame
                )

            self.frame_idx += 1

    def stop(self):
        self.run = False

    def get_times(self):
        return self.preprocess_time / self.frame_idx, self.inference_time / self.frame_idx, self.postporcess_time / self.frame_idx

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument(
        "--video_path",
        type=str,
        default="datasets/video",
        help="Path of input video or dateset",
    )
    parser.add_argument(
        "--points",
        type=str,
        default="[[210, 350], [250, 220]]",
        help='The coordinates of the input_point, format "[[x,y]]"',
    )
    parser.add_argument(
        "--label", type=str, default="[1,1]", help="Label of input points"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Used to save the output results of the model",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="models/BM1688/video/sam2_image_encoder_no_pos.bmodel",
        help="Path of sam2 image encoder bmodel",
    )
    parser.add_argument(
        "--image_decoder_path",
        type=str,
        default="models/BM1688/video/sam2_image_decoder.bmodel",
        help="Path of sam2 image decoder bmodel",
    )
    parser.add_argument(
        "--memory_attention_path",
        type=str,
        default="models/BM1688/video/sam2_memory_attention_nomatmul.bmodel",
        help="Path of sam2 memory attention bmodel",
    )
    parser.add_argument(
        "--memory_encoder_path",
        type=str,
        default="models/BM1688/video/sam2_memory_encoder.bmodel",
        help="Path of sam2 memory encoder bmodel",
    )
    parser.add_argument(
        "--skip_num",
        type=int,
        default=1,
        help="Number of frames to draw, one every few frames",
    )
    parser.add_argument(
        "--constant_path",
        type=str,
        default="models/BM1688/video/",
        help="Path of constant npz file",
    )
    parser.add_argument(
        "--dev_id",
        type=int,
        default=0,
        help="TPU device id",
    )
    return parser.parse_args()


def main(args):

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"{args.video_path} is not existed.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, "video")):
        os.makedirs(os.path.join(args.output_dir, "video"))

    import ast

    input_point = np.array([ast.literal_eval(args.points)], dtype=np.float32)
    input_label = np.array([ast.literal_eval(args.label)], np.int32)

    sam2_video = SAM2Video(
        dev_id=args.dev_id,
        image_encoder_path=args.image_encoder_path,
        image_decoder_path=args.image_decoder_path,
        memory_attention_path=args.memory_attention_path,
        memory_encoder_path=args.memory_encoder_path,
        constant_path=args.constant_path,
        output_dir=args.output_dir,
    )

    sam2_video.load_video(args.video_path)
    sam2_video.inference(input_point, input_label)


if __name__ == "__main__":
    args = argsparser()
    main(args)
