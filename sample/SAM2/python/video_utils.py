# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
import numpy as np
import cv2

def show_mask(mask, img, color = np.array([255, 144, 30]), obj_id=None):
    color = color.reshape(1, 1, -1)

    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1)

    mask_image = mask * color
    img = (img * ~mask) + (img * mask) * 0.6 + mask_image * 0.4

    return img


def show_points(coords, labels, img):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    for p in pos_points:
        cv2.drawMarker(
            img, p, (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
            markerSize=30, thickness=5)
    for p in neg_points:
        cv2.drawMarker(
            img, p, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
            markerSize=30, thickness=5)

    return img


def show_box(box, img):
    if box is None:
        return img

    cv2.rectangle(
        img, (box[0], box[1]), (box[2], box[3]), color=(2, 118, 2),
        thickness=3,
        lineType=cv2.LINE_4,
        shift=0)

    return img


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = np.arange(pe_dim, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = np.concatenate([pos_embed.sin(), pos_embed.cos()], axis=-1)
    return pos_embed


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = np.concatenate([old_point_inputs["point_coords"], new_points], axis=1)
        labels = np.concatenate([old_point_inputs["point_labels"], new_labels], axis=1)

    return {"point_coords": points, "point_labels": labels}


def trunc_normal(size, std=0.02, a=-2, b=2):
    values = np.random.normal(loc=0., scale=std, size=size)
    values = np.clip(values, a*std, b*std)
    return values.astype(np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def reflection_padding(target_shape, original_data):
    rows_to_add = target_shape[0] - original_data.shape[0]
    pad_before = rows_to_add // 2
    pad_after = rows_to_add - pad_before
    padded_tensor = np.pad(original_data, ((pad_before, pad_after), (0, 0), (0, 0)), mode='reflect')

    return padded_tensor


def interpolate(low_res_multimasks, image_size):
    high_res_multimasks = np.zeros((low_res_multimasks.shape[0], low_res_multimasks.shape[1], image_size[0], image_size[1]), dtype=np.float32)
    for b in range(low_res_multimasks.shape[0]):
        for c in range(low_res_multimasks.shape[1]):
            high_res_multimasks[b][c] = cv2.resize(low_res_multimasks[b][c], (image_size[1], image_size[0]), high_res_multimasks, interpolation=cv2.INTER_LINEAR)

    return high_res_multimasks