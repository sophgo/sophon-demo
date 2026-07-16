"""Unlimited-OCR image preprocessing (numpy / cv2 / Pillow, no torch).

Ported from the original model (modeling_unlimitedocr.py infer + deepencoder.py
BasicImageTransform) so the SAIL runtime can reproduce the exact image pipeline.

Pipeline summary (see sample README §3 for the full architecture):
  PIL image
    -> dynamic_preprocess (gundam: tile into 640x640 + 1024x1024 global; base: single 1024)
    -> BasicImageTransform (ToTensor + Normalize mean=std=0.5, RGB) -> [-1,1] float32
    -> [vision tower bmodel] -> image embeddings [num_img_tokens, 1280]   (TODO: deeplip bmodel)
    -> replace <image> placeholder positions in the LLM token embedding.

Image token id = 128815. patch_size=16, downsample_ratio=4.
  num_queries      = ceil((image_size // 16) / 4)   # 640 -> 10
  num_queries_base = ceil((base_size  // 16) / 4)   # 1024 -> 16
"""

import math
from PIL import Image, ImageOps
import numpy as np

IMAGE_TOKEN_ID = 128815
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4

# BasicImageTransform: ToTensor + Normalize(mean=std=0.5)
IMG_MEAN = (0.5, 0.5, 0.5)  # RGB
IMG_STD = (0.5, 0.5, 0.5)
PAD_COLOR = tuple(int(x * 255) for x in IMG_MEAN)  # (127,127,127)


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    avg_ar = sum(r[0] / r[1] for r in target_ratios) / len(target_ratios)
    best = None
    best_diff = float("inf")
    for r in target_ratios:
        ar = r[0] / r[1]
        d = abs(aspect_ratio - ar)
        if d < best_diff or (d == best_diff and abs(avg_ar - ar) < abs(avg_ar - best[0] / best[1])):
            best = r
            best_diff = d
    return best


def dynamic_preprocess(image, min_num=2, max_num=32, image_size=640, use_thumbnail=False):
    """Tile `image` into non-overlapping image_size x image_size patches.

    Returns (list_of_PIL_tiles, (width_tiles, height_tiles))."""
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num},
        key=lambda x: x[0] * x[1],
    )
    target = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    tw = image_size * target[0]
    th = image_size * target[1]
    img = image.resize((tw, th))
    tiles = []
    for h in range(target[1]):
        for w in range(target[0]):
            tiles.append(img.crop((w * image_size, h * image_size,
                                   (w + 1) * image_size, (h + 1) * image_size)))
    if use_thumbnail:
        tiles.append(image.resize((image_size, image_size)))
    return tiles, (target[0], target[1])


def basic_transform(pil_img):
    """ToTensor + Normalize(mean=std=0.5), RGB order. Returns [3, H, W] float32 in [-1,1]."""
    arr = np.asarray(pil_img.convert("RGB").astype(np.float32) if hasattr(pil_img, "astype") else
                     np.asarray(pil_img.convert("RGB"), dtype=np.float32))
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = arr / 255.0
    arr = (arr - np.array(IMG_MEAN, dtype=np.float32).reshape(3, 1, 1)) / \
          np.array(IMG_STD, dtype=np.float32).reshape(3, 1, 1)
    return arr  # [3, H, W] float32


def preprocess_image(image, image_mode="gundam", base_size=1024, image_size=640, crop_mode=True):
    """Produce (images_crop, images_ori, crop_ratio) as numpy float32 arrays.

    gundam (crop_mode=True): global view padded to base_size x base_size, plus
        image_size x image_size tiles (if the image is larger than image_size).
    base   (crop_mode=False): single padded image_size x image_size view.
    """
    image = image.convert("RGB")
    if image_mode == "gundam" and crop_mode:
        if image.size[0] <= image_size and image.size[1] <= image_size:
            crop_ratio = [1, 1]
            tiles = []
        else:
            tiles, cr = dynamic_preprocess(image, min_num=2, max_num=32, image_size=image_size)
            crop_ratio = list(cr)
        global_view = ImageOps.pad(image, (base_size, base_size), color=PAD_COLOR)
        images_ori = basic_transform(global_view)[None]  # [1, 3, base, base]
        if tiles and (crop_ratio[0] > 1 or crop_ratio[1] > 1):
            images_crop = np.stack([basic_transform(t) for t in tiles], axis=0)  # [N, 3, is, is]
        else:
            images_crop = np.zeros((1, 3, base_size, base_size), dtype=np.float32)
        return images_crop, images_ori, tuple(crop_ratio)
    else:  # base
        # pad 640x640 view to 1024x1024 so the fixed-size vit bmodel can
        # process it; the vit output [16,16,1280] is then cropped to
        # [nq,nq,1280] (nq=10) in forward_vision.
        view = ImageOps.pad(image, (image_size, image_size), color=PAD_COLOR)
        view = ImageOps.pad(view, (base_size, base_size), color=PAD_COLOR)
        images_ori = basic_transform(view)[None]  # [1, 3, 1024, 1024]
        images_crop = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
        return images_crop, images_ori, (1, 1)


def num_queries_for(size):
    return math.ceil((size // PATCH_SIZE) / DOWNSAMPLE_RATIO)


def build_image_tokens(image_mode="gundam", base_size=1024, image_size=640, crop_ratio=(1, 1)):
    """Build the image-placeholder token sequence (list of 128815)."""
    nq = num_queries_for(image_size)
    nqb = num_queries_for(base_size)
    if image_mode == "gundam":
        toks = ([IMAGE_TOKEN_ID] * nqb + [IMAGE_TOKEN_ID]) * nqb  # 272
        toks += [IMAGE_TOKEN_ID]                                   # +1 sep -> 273
        w, h = crop_ratio
        if w > 1 or h > 1:
            toks += ([IMAGE_TOKEN_ID] * (nq * w) + [IMAGE_TOKEN_ID]) * (nq * h)
    else:  # base
        toks = ([IMAGE_TOKEN_ID] * nq + [IMAGE_TOKEN_ID]) * nq  # 110
        toks += [IMAGE_TOKEN_ID]                                  # +1 -> 111
    return toks


def build_input_ids(tokenizer, prompt, image_token_seq, bos_id=0):
    """Build the full input_ids: split prompt on '<image>', splice image tokens.

    prompt is expected to contain a literal '<image>' placeholder (e.g.
    '<image>document parsing.'). Returns (input_ids list, images_seq_mask list)."""
    before, after = (prompt.split("<image>", 1) + [""])[:2] if "<image>" in prompt else (prompt, "")
    ids_before = tokenizer.encode(before, add_special_tokens=False)
    ids_after = tokenizer.encode(after, add_special_tokens=False)
    input_ids = [bos_id] + ids_before + list(image_token_seq) + ids_after
    mask = [False] + [False] * len(ids_before) + [True] * len(image_token_seq) + [False] * len(ids_after)
    return input_ids, mask
