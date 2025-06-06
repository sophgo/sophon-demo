from __future__ import annotations

import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
import cv2
import psutil
import SILK2.Tools.logger as Logger
from sophon import sail
import numpy as np


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


BMCV_RESIZE_ALGORITHM_MAP = {
    "INTER_NEAREST": sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST,
    "INTER_LINEAR": sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR,
    "INTER_BICUBIC": sail.bmcv_resize_algorithm.BMCV_INTER_BICUBIC
}

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels or max_pixels == min_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def bmimage2image(bmcv: sail.Bmcv, image: sail.BMImage) -> Image.Image:
    if image.format() != sail.Format.FORMAT_RGB_PACKED:
        image = bmcv.convert_format(image, sail.Format.FORMAT_RGB_PACKED)
    image = image.asnumpy()
    image = Image.fromarray(image)
    return image

def fetch_image(handle: sail.Handle, bmcv: sail.Bmcv, ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> sail.BMImage:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif not isinstance(image, str):
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    elif image.startswith("http://") or image.startswith("https://"):
        # image_obj = Image.open(requests.get(image, stream=True).raw)
        with open(requests.get(image, stream=True).raw, "rb") as fr:
            image_bytes = fr.read()
        image_obj = bmcv.imdecode(image_bytes)
    elif image.startswith("file://"):
        # image_obj = Image.open(image[7:])
        with open(image[7:], "rb") as fr:
            image_bytes = fr.read()
        image_obj = bmcv.imdecode(image_bytes)
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # image_obj = Image.open(BytesIO(data))
            image_obj = bmcv.imdecode(BytesIO(data).getvalue())
        else:
            raise ValueError(f"Unrecognized image input type, support base64")

    else:
        # image_obj = Image.open(image)
        with open(image, "rb") as fr:
            image_bytes = fr.read()
        image_obj = bmcv.imdecode(image_bytes)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    if isinstance(image_obj, Image.Image):
        image = image_obj.convert("RGB")
        image = np.array(image)
        image = sail.BMImage(handle=handle, buffer=image, h=image.shape[0], \
                        w=image.shape[1], format=sail.Format.FORMAT_RGB_PACKED)
    else:
        image = image_obj
    return image

def compute_image_size(image: sail.BMImage, ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR):
    ## resize
    if "max_side" in ele and ele["max_side"] is not None:
        width, height = image.width(), image.height()
        if width > height:
            height = int(ele["max_side"] / width * height)
            width = int(ele["max_side"])
        else:
            width = int(ele["max_side"] / height * width)
            height = int(ele["max_side"])
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
        )
    elif "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.width(), image.height()
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    return resized_width, resized_height # wh

def image_resize(bmcv: sail.Bmcv, image: Image.Image, resized_width: int, resized_height: int, resize_algorithm: str):
    # image = image.resize((resized_width, resized_height))
    image = bmcv.resize(image, resized_width, resized_height, BMCV_RESIZE_ALGORITHM_MAP[resize_algorithm])
    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _cpu_memory_check(total_bytes, rate=2.2):
    mem = psutil.virtual_memory()
    assert total_bytes * rate < mem.available, "available memory not suffice to load data"

def _read_video_torchvision(
    ele: dict,
) -> torch.Tensor:
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]

    # memory check
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if ele.get("video_end", None) is None and ele.get("video_start", 0.0) > 0:
        num_frames = 1 + num_frames - ele.get("video_start", 0.0) * fps # torchvision.io.read_video has a other frame 
    elif ele.get("video_end", None) is not None and ele.get("video_start", 0.0) >= 0:
        assert ele.get("video_end", None) - ele.get("video_start", 0.0) > 0
        num_frames = 1 + (ele.get("video_end", None) - ele.get("video_start", 0.0)) * fps # torchvision.io.read_video has a other frame 
    elif ele.get("video_end", None) is None and ele.get("video_start", 0.0) == 0:
        pass
    else:
        raise ValueError(f"error params video_end {video_end} or video_start {video_start}")
    frame_size = width * height * 3 # bytes
    _cpu_memory_check(num_frames * frame_size)

    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    video = video[idx]
    return video


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
    ele: dict,
) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    return video

def _read_video_pyav(
    ele: dict,
) -> torch.Tensor:
    """读取视频并转换为张量

    Args:
        ele (dict): 视频配置字典
            - video: 视频路径

    Returns:
        torch.Tensor: 视频张量 (T, C, H, W)
    """
    import av
    video_path = ele["video"]
    st = time.time()
    
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == 'video')
    
    total_frames = video_stream.frames
    video_fps = float(video_stream.average_rate)
    
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    _cpu_memory_check(nframes * video_stream.width * video_stream.height * 3)
    
    frame_indices = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    
    frames = []
    for idx in frame_indices:
        target_pts = int(idx / video_fps / video_stream.time_base)
        
        container.seek(target_pts, stream=video_stream)
        
        for frame in container.decode(video_stream):
            frame_idx = int(frame.pts * video_stream.time_base * video_fps)
            if frame_idx >= idx:
                img = frame.to_image()
                frames.append(np.array(img))
                break
    
    video = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)  # TCHW
    
    logger.info(f"PyAV: {video_path=}, {total_frames=}, {video_fps=:.2f}, "
                f"selected={nframes}frames, time={time.time()-st:.3f}s")
    
    return video


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "pyav": _read_video_pyav,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(handle: sail.Handle, bmcv: sail.Bmcv, ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video = None
        if ele["video"].startswith("http://") or ele["video"].startswith("https://"):
            ele["video"] = requests.get(image, stream=True).raw
            video_reader_backend = get_video_reader_backend()
            video = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        elif ele["video"].startswith("data:video"):
            if "base64," in ele["video"]:
                _, base64_data = ele["video"].split("base64,", 1)
                data = base64.b64decode(base64_data)
                file_name = "input_video_temp"
                with open(file_name, "wb") as f:
                    f.write(data)
                ele["video"] = file_name
                video_reader_backend = get_video_reader_backend()
                video = VIDEO_READER_BACKENDS[video_reader_backend](ele)
            else:
                raise ValueError(f"Unrecognized video input type, support base64")
        else:
            video_reader_backend = get_video_reader_backend()
            video = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        if video is None:
            raise ValueError(f"Unrecognized video input, support local path, http url, frame path list, base64, got {ele['video']}")
        if "video_sample_num" in ele:
            video = video[::ele["video_sample_num"]]
        video = video[::]
        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        for video_element in ele["video"]:
            if not isinstance(video_element, str):
                raise ValueError(f"Unrecognized video input, support local path, http url, frame path list, base64, got {ele['video']}")
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            bmimage2image(bmcv, fetch_image(handle, bmcv, {"image": video_element, **process_info}, size_factor=image_factor))
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    handle: sail.Handle, conversations: list[dict] | list[list[dict]], max_vision_shape: list[int], merge_size: int=2, logger: Logger=None
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    bmcv = sail.Bmcv(handle)
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []

    # compute adaptive size with keeping ratio
    max_pixels_num = 1
    for dim in max_vision_shape:
        max_pixels_num *= dim
    remain_max_pixels_num = max_pixels_num
    has_video = False
    imgs_and_size = []
    auto_indxs = []
    img_indx = 0
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            if "resize_type" not in vision_info:
                vision_info["resize_type"] = "INTER_LINEAR"
            if vision_info["resize_type"] != "Origin":
                if "max_side" not in vision_info:
                    vision_info["max_side"] = 0
            if "max_side" not in vision_info or vision_info["max_side"] > 0:
                image = fetch_image(handle, bmcv, vision_info)
                # if vision_info["max_side"] == 0:
                #     del vision_info["max_side"]
                #     vision_info["max_pixels"] = image.width() * image.height()
                sizes = compute_image_size(image, vision_info)
                imgs_and_size.append([image, *sizes, vision_info["resize_type"]])
                # compute remain space
                remain_max_pixels_num = remain_max_pixels_num - 3 * sizes[0] * sizes[1] * merge_size
                if remain_max_pixels_num < 0:
                    raise ValueError(
                            f"The vision input_length must be shorter than model's vision seq_length (got `input_length`: {max_pixels_num-remain_max_pixels_num}"
                            f" and `seq_length`: {max_pixels_num})."
                    )
            else:
                auto_indxs.append(img_indx)
                image = fetch_image(handle, bmcv, vision_info)
                imgs_and_size.append([image, vision_info, vision_info["resize_type"]])
            img_indx += 1
        elif "video" in vision_info:
            has_video = True
        else:
            raise ValueError("image, image_url or video should in content.")
    # generate auto size
    if len(auto_indxs) > 0:
        if has_video:
            raise ValueError("auto param only support full images, but get video")
        auto_mean_max_pixels_num = remain_max_pixels_num // 3 // len(auto_indxs) // merge_size
        if auto_mean_max_pixels_num < MIN_PIXELS:
            raise ValueError(
                    f"The vision input_length must be shorter than model's vision seq_length (got `input_length`: auto"
                    f" and `seq_length`: {max_pixels_num})."
            )
        for img_indx in auto_indxs:
            if imgs_and_size[img_indx][1]["max_side"] == 0:
                auto_mean_max_pixels_num = min(auto_mean_max_pixels_num, imgs_and_size[img_indx][0].width() * imgs_and_size[img_indx][0].height())
            resize_type = imgs_and_size[img_indx][1]["resize_type"]
            imgs_and_size[img_indx] = [imgs_and_size[img_indx][0]]
            auto_size = compute_image_size(imgs_and_size[img_indx][0], {"max_pixels": auto_mean_max_pixels_num, "min_pixels": auto_mean_max_pixels_num})
            imgs_and_size[img_indx].extend(auto_size)
            imgs_and_size[img_indx].append(resize_type)
            if logger is not None:
                logger.info(f"{Logger.file_lineno()} generate image auto size {auto_size}")

    img_indx = 0
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            if imgs_and_size[img_indx][-1] == "Origin":
                imgs_and_size[img_indx][-1] = "INTER_LINEAR"
                print("Chanege to INTER_LINEAR")
            image = image_resize(bmcv, *(imgs_and_size[img_indx]))
            image_inputs.append(bmimage2image(bmcv, image))
            img_indx += 1
        elif "video" in vision_info:
            video_inputs.append(fetch_video(handle, bmcv, vision_info))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    
    return image_inputs, video_inputs
