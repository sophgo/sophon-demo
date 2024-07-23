#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def get_box(image_pil, x):
    # (cx cy w h) to (x y x y)
    x_c, y_c, w, h = x
    boxes = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    
    img_w, img_h = image_pil.size
    scale_fct = np.array([img_w, img_h, img_w, img_h])

    boxes = boxes * scale_fct
    return boxes

def plot_boxes_to_image(image_pil, tgt):
    img_w, img_h  = image_pil.size
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    scale_fct = np.array([img_w, img_h, img_w, img_h])
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        # cx, cy, w, h to x0 y0 x1 y1
        x_c, y_c, w, h = box
        box = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)] * scale_fct

        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()

        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)

        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)
    
    return image_pil, mask

def get_phrases_from_posmap_np(
    posmap, tokenized, tokenizer, left_idx: int = 0, right_idx: int = 255
):
    if len(posmap.shape) == 1:
        posmap[0: left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = np.nonzero(posmap)[0].tolist()
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        result = tokenizer.decode(token_ids)
        return result
    else:
        raise NotImplementedError("posmap must be 1-dim")

def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list=[101, 102, 1012, 1029]):
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = np.zeros((bs, num_token), dtype=bool)
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = np.column_stack(np.where(special_tokens_mask))

    # generate attention mask and positional ids
    attention_mask = np.eye(num_token, dtype=bool).reshape((1, num_token, num_token)).repeat(bs, axis=0)
    position_ids = np.zeros((bs, num_token), dtype=int)

    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = np.arange(0, col - previous_col)

        previous_col = col

    return attention_mask, position_ids

def gen_encoder_output_proposals():
    N, S, C = 1, 13294, 256
    proposals = []
    _cur = 0
    memory_padding_mask = np.zeros((1, 13294), dtype=bool)
    spatial_shapes = np.array([[100, 100], [50, 50], [25, 25], [13, 13]])
    
    for lvl, (H, W) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H * W)].reshape(N, H, W, 1)
        valid_H = np.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = np.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = np.meshgrid(
            np.linspace(0, H - 1, H, dtype=np.float32),
            np.linspace(0, W - 1, W, dtype=np.float32),
        )
        
        # there is a transpose between torch and np meshgrid
        grid_y = grid_y.T
        grid_x = grid_x.T
        
        grid = np.concatenate([grid_x[..., np.newaxis], grid_y[..., np.newaxis]], axis=-1)

        scale = np.concatenate([valid_W[..., np.newaxis], valid_H[..., np.newaxis]], axis=1).reshape(N, 1, 1, 2)
        grid = (grid[np.newaxis, ...] + 0.5) / scale

        wh = np.ones_like(grid) * 0.05 * (2.0**lvl)

        proposal = np.concatenate((grid, wh), axis=-1).reshape(N, -1, 4)
        proposals.append(proposal)
        _cur += H * W

    output_proposals = np.concatenate(proposals, axis=1)
    return output_proposals

def create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    num_boxes = len(token_span)
    positive_map = np.zeros((num_boxes, max_text_len), dtype=np.float32)

    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None

            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos : end_pos + 1] = 1

    row_sum = positive_map.sum(-1)
    positive_map /= (row_sum[:, None] + 1e-6)

    return positive_map

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
