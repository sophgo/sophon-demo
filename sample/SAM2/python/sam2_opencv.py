# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
import argparse
import ast
import copy
import json
import logging
import os
import time

import cv2
import datasets
import numpy as np
import sophon.sail as sail
from pympler import asizeof
from tqdm import tqdm
from utils import (bytes_to_megabytes, draw_masks, logger,
                   mask_to_coco_segmentation)

logger.setLevel(level=logging.DEBUG)


class SAM2ImageEncoder:

    def __init__(self, encoder_model_path: str, dev_id: int = 0) -> None:
        self.encoder = sail.Engine(encoder_model_path, dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.encoder.get_graph_names()[0]
        self.input_names = self.encoder.get_input_names(self.graph_name)[0]
        self.input_shape = self.encoder.get_input_shape(
            self.graph_name, self.input_names
        )
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.encoder_time = 0
        self.preprocess_time = 0

    def __call__(self, image: np.ndarray):
        return self.encode_image(image)

    def encode_image(self, image: np.ndarray):

        input_tensor = self.prepare_input(image)
        start = time.perf_counter()
        outputs = self.infer(input_tensor)
        self.encoder_time = time.perf_counter() - start
        return [output for output in outputs.values()]

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        start = time.perf_counter()

        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(
            input_img,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        input_img = input_img / 255.0
        # 使用 OpenCV 的加权求和函数加速归一化
        normalized_img = cv2.subtract(input_img, self.mean)
        normalized_img = cv2.divide(normalized_img, self.std)
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        self.preprocess_time = time.perf_counter() - start
        return input_tensor

    def infer(self, input_tensor: np.ndarray):
        return self.encoder.process(self.graph_name, {self.input_names: input_tensor})


class SAM2ImageDecoder:

    def __init__(
        self,
        decoder_model_path: str,
        encoder_input_size,
        orig_im_size=None,
        mask_threshold: float = 0.0,
        dev_id: int = 0,
        select_best: bool = True,
    ) -> None:
        self.decoder = sail.Engine(decoder_model_path, dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.decoder.get_graph_names()[0]
        self.input_names = self.decoder.get_input_names(self.graph_name)
        self.orig_im_size = (
            orig_im_size if orig_im_size is not None else encoder_input_size
        )
        self.encoder_input_size = encoder_input_size
        self.mask_threshold = mask_threshold
        self.scale_factor = 4
        self.select_best = select_best
        self.decoder_time = 0
        self.preprocess_time = 0
        self.postprocess_time = 0

    def __call__(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords,
        point_labels,
    ):

        return self.predict(
            image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels
        )

    def predict(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords,
        point_labels,
    ):
        inputs = self.prepare_inputs(
            image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels
        )
        start = time.perf_counter()
        outputs = self.infer(inputs)
        self.decoder_time = time.perf_counter() - start
        return self.process_output(outputs)

    def prepare_inputs(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords,
        point_labels,
    ):
        start = time.perf_counter()

        input_point_coords, input_point_labels = self.prepare_points(
            point_coords, point_labels
        )
        num_labels = input_point_labels.shape[0]
        mask_input = np.zeros(
            (
                num_labels,
                1,
                self.encoder_input_size[0] // self.scale_factor,
                self.encoder_input_size[1] // self.scale_factor,
            ),
            dtype=np.float32,
        )
        has_mask_input = np.array([0], dtype=np.float32)

        self.preprocess_time = time.perf_counter() - start
        return (
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            input_point_coords,
            input_point_labels,
            mask_input,
            has_mask_input,
        )

    def prepare_points(self, point_coords, point_labels):
        if isinstance(point_coords, np.ndarray):
            input_point_coords = point_coords[np.newaxis, ...]
            input_point_labels = point_labels[np.newaxis, ...]
        else:
            max_num_points = max([coords.shape[0] for coords in point_coords])
            input_point_coords = np.zeros(
                (len(point_coords), max_num_points, 2), dtype=np.float32
            )
            input_point_labels = (
                np.ones((len(point_coords), max_num_points), dtype=np.float32) * -1
            )

            for i, (coords, labels) in enumerate(zip(point_coords, point_labels)):
                input_point_coords[i, : coords.shape[0], :] = coords
                input_point_labels[i, : labels.shape[0]] = labels

        input_point_coords[..., 0] = (
            input_point_coords[..., 0]
            / self.orig_im_size[1]
            * self.encoder_input_size[1]
        )  # Normalize x
        input_point_coords[..., 1] = (
            input_point_coords[..., 1]
            / self.orig_im_size[0]
            * self.encoder_input_size[0]
        )  # Normalize y

        return input_point_coords.astype(np.float32), input_point_labels.astype(
            np.float32
        )

    def infer(self, inputs):
        return self.decoder.process(
            self.graph_name,
            {self.input_names[i]: inputs[i] for i in range(len(self.input_names))},
        )

    def process_output(self, outputs):
        # 每次转bmodel之后，bmodel中的outputname末尾最添加一个f32之类的字符串
        start = time.perf_counter()
        for key, value in outputs.items():
            if "iou" in key:
                scores = value.squeeze()
            if "mask" in key:
                masks = value
        if self.select_best:
            # 基于scores选中分数最高的mask
            masks = masks.squeeze()
            max_score = np.argmax(scores)
            best_mask = masks[max_score]
            best_mask = cv2.resize(
                best_mask, (self.orig_im_size[1], self.orig_im_size[0])
            )
            self.postprocess_time = time.perf_counter() - start
            return np.array([[best_mask]]), format(np.max(scores), ".4f")
        else:
            return masks, scores.tolist()


class SAM2Image:
    # 在转bmodel的时候将模型定义为静态，每次给image decoder输入一个点的坐标
    # 该类主要对所要分割的图像的点和框的预测做管理，部分达到多点输入效果（多点之间相互隔离）
    def __init__(self, encoder_path, decoder_path, select_best):
        self.sam2_encoder = SAM2ImageEncoder(encoder_path)
        self.decoder_path = decoder_path
        self.select_best = select_best
        self.image_info = {}
        self.res = {}
        self.point_nums = 0
        self.init_times()

    def init_times(self):
        self.preprocess_time = 0
        self.encoder_time = 0
        self.decoder_time = 0
        self.postprocess_time = 0

    def set_image(self, img):
        self.h, self.w, _ = img.shape
        # print(img.shape)
        # self.img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        self.img = copy.deepcopy(img)
        self.image_embeddings = self.sam2_encoder(self.img)
        self.sam2_decoder = SAM2ImageDecoder(
            self.decoder_path,
            self.sam2_encoder.input_shape[2:]
        )
        self.encoder_time += self.sam2_encoder.encoder_time
        self.preprocess_time += self.sam2_encoder.preprocess_time
        self.reset_points()

    def add_point(self, point_coords, label):
        self.image_info[str(self.point_nums)] = {
            "label": label,
            "coords": point_coords,
        }
        self.point_nums += 1

    def add_box(self, box_coords, label):
        # 将坐标框的中心点作为参考点，box的表示参照coco
        point_coords = (
            box_coords[0] + box_coords[2] / 2,
            box_coords[1] + box_coords[3] / 2,
        )
        self.image_info[str(self.point_nums)] = {
            "label": label,
            "coords": point_coords,
            "origin": box_coords,
        }
        self.point_nums += 1

    def predict(self):
        # select_best为True时输出score最高的mask，否则输出多个mask
        high_res_feats_0, high_res_feats_1, image_embed = self.image_embeddings

        for point_id in self.image_info.keys():
            input_point = np.array([self.image_info[str(point_id)]["coords"]])
            input_label = np.array([self.image_info[str(point_id)]["label"]])
            masks, scores = self.sam2_decoder(
                image_embed,
                high_res_feats_0,
                high_res_feats_1,
                input_point,
                input_label,
            )
            self.decoder_time += self.sam2_decoder.decoder_time
            self.preprocess_time += self.sam2_decoder.preprocess_time
            self.postprocess_time += self.sam2_decoder.postprocess_time
            self.res[point_id] = {"masks": masks, "scores": scores}

        return self.res

    def save_img(self, image_id):
        for _res in self.res.values():
            masks = _res["masks"]
            self.img = (
                draw_masks(self.img, [masks])
                if self.select_best
                else draw_masks(self.img, masks)
            )
        # print("Image Shape in save_img", self.h, self.w)
        res_img = cv2.resize(self.img, (self.w, self.h))
        cv2.imwrite("{}.jpg".format(image_id), res_img)

    def reset_points(self):
        self.image_info = {}
        self.res = {}
        self.point_nums = 0

    def get_times_info(self):

        return (
            self.preprocess_time,
            self.encoder_time,
            self.decoder_time,
            self.postprocess_time,
        )


def pred_dataset(args):

    logger.setLevel(level=logging.CRITICAL)

    dataset_type = args.dataset_type
    dataset_names = getattr(datasets, "__all__", None)
    if dataset_type in dataset_names:
        Dataset = getattr(datasets, dataset_type)
    else:
        raise RuntimeError('Invalid Dataset type: "{}"'.format(dataset_type))

    dataset = Dataset(args.img_path, args.gt_path, args.detect_num)
    sam2 = SAM2Image(args.encoder_bmodel, args.decoder_bmodel, args.select_best)
    seg_res = []

    memory_used = 0
    pbar = tqdm(dataset, total=args.detect_num)
    for data_info in pbar:
        pbar.set_description(f"Segmentation Info Memeroy Used: {memory_used} MB")
        img = data_info["img"]
        centers_info = data_info["centers_info"]
        image_id = data_info["image_id"]
        h, w, _ = img.shape
        scale_x = 1024 / w
        scale_y = 1024 / h
        sam2.set_image(img)

        bbox_list = []
        for center_info in centers_info:
            point = center_info["center"]
            label = center_info["label"]
            bbox_list.append(center_info["bbox"])
            # 对中心点坐标和segmentation的坐标进行相应的缩放
            point = [point[0] * scale_x, point[1] * scale_y]
            sam2.add_point(point, label)

        res = sam2.predict()
        sam2.save_img(os.path.join(args.output_dir, "images", str(image_id)))
        segmentations = []
        for key in res.keys():
            _res = res[key]
            mask = np.squeeze(_res["masks"])
            # 默认对边界框进行简化，防止数据太多内存不足
            segmentation = mask_to_coco_segmentation(mask, img.shape, simplify=True)
            segmentations.append(segmentation)
            score = _res["scores"]

        seg_res.append(
            {
                "image_id": image_id,
                "category_id": label,
                "bbox": bbox_list,
                "segmentation": segmentations,
                "score": float(score),
            }
        )

        memory_used = round(bytes_to_megabytes(asizeof.asizeof(seg_res)), 4)
        sam2.reset_points()
        if memory_used > 800:
            break

    base_filename = os.path.splitext(os.path.basename(args.encoder_bmodel))[0]
    json_name = "{}_{}_opencv_python_result.json".format(
        base_filename, args.dataset_type
    )
    with open(os.path.join(args.output_dir, json_name), "w") as jf:
        json.dump(seg_res, jf, indent=4, ensure_ascii=False)

    def avg_time(x):
        return x / args.detect_num

    preprocess_time, encoder_time, decoder_time, postprocess_time = list(
        map(avg_time, sam2.get_times_info())
    )

    print(
        f"Preprocess time(ms): {preprocess_time * 1000:.2f}\n",
        f"Encoder time(ms): {encoder_time * 1000:.2f}\n ",
        f"Decoder time(ms): {decoder_time * 1000:.2f}\n ",
        f"Postprocess time(ms): {postprocess_time * 1000:.2f}\n",
    )
    print("\nResult saved in {}".format(os.path.join(args.output_dir, json_name)))


def pred(args):
    img = cv2.imread(args.img_path)
    sam2 = SAM2Image(args.encoder_bmodel, args.decoder_bmodel, args.select_best)
    sam2.set_image(img)

    input_points = ast.literal_eval(args.points)
    for point in input_points:
        if len(point) == 2:
            sam2.add_point(point, args.label)
        elif len(point) == 4:
            sam2.add_box(point, args.label)

    sam2.predict()
    sam2.save_img(os.path.join(args.output_dir, "images", "res"))

    preprocess_time, encoder_time, decoder_time, postprocess_time = (
        sam2.get_times_info()
    )

    print(
        f"Preprocess time(ms): {preprocess_time * 1000:.2f}\n",
        f"Encoder time(ms): {encoder_time * 1000:.2f}\n ",
        f"Decoder time(ms): {decoder_time * 1000:.2f}\n ",
        f"Postprocess time(ms): {postprocess_time * 1000:.2f}\n",
    )


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument(
        "--mode",
        type=str,
        default="img",
        help="Segmentation mode in [img, dataset], in img mode, need img_path, points and label, in dataset mode, need img_path, dateset_type and gt_path",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="datasets/truck.jpg",
        help="Path of input image or dateset",
    )
    parser.add_argument(
        "--points",
        type=str,
        default="[[500, 375], [345, 300]]",
        help='The coordinates of the input_point(point or box), point format "[[x,y]]", box format "[[x1,y1,w,h]]", like coco bbox',
    )
    parser.add_argument("--label", type=int, default=1, help="Label of input points")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="COCODataset",
        help="Dataset type in [COCODataset]",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="datasets/instances_val2017.json",
        help="Ground truth file path",
    )
    parser.add_argument(
        "--detect_num",
        type=int,
        default=1000,
        help="Number of images in dataset for sam2 to segmentation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Used to save the output results of the model",
    )
    parser.add_argument(
        "--encoder_bmodel",
        type=str,
        default="models/BM1688/image_encoder/sam2_encoder_f16_1b_2core.bmodel",
        help="Path of encoder bmodel",
    )
    parser.add_argument(
        "--decoder_bmodel",
        type=str,
        default="models/BM1688/image_decoder/sam2_decoder_f16_1b_2core.bmodel",
        help="Path of decoder bmodel",
    )
    parser.add_argument(
        "--select_best",
        type=bool,
        default=True,
        help="Select best mask for segmentation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argsparser()
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"{args.input_image} is not existed.")
    if not os.path.exists(args.encoder_bmodel):
        raise FileNotFoundError(f"{args.encoder_bmodel} is not existed.")
    if not os.path.exists(args.decoder_bmodel):
        raise FileNotFoundError("{} is not existed.".format(args.decoder_bmodel))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, "images"))

    if args.mode == "img":
        pred(args)
    elif args.mode == "dataset":
        pred_dataset(args)
    else:
        raise RuntimeError('Invalid mode "{}"'.format(args.mode))
