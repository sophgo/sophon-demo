#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import matplotlib.pyplot as plt
import base64
import cv2 
from io import BytesIO
import numpy as np
import torch
import os
import gc

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def draw_output_image(image, masks, boxes_filt, pred_phrases, save_image=False):
    plt.figure(figsize=(10, 10))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    
    for mask in masks:
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        show_mask(mask_np, plt.gca(), random_color=True)
    
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box, plt.gca(), label)

    plt.axis('off')

    if save_image:
        plt.savefig(
            os.path.join("../results", "grounded_sam_output.jpg"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        plt.close()
        return None
    else:
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read())
        img_base64_str = img_base64.decode('utf-8')
        plt.close()
        
        del buffer, img_base64
        gc.collect()
        return img_base64_str

def get_grounding_output(image, boxes, labels):
    # image shape H, W, C
    img_h, img_w, _ = image.shape
    scale_fct = np.array([img_w, img_h, img_w, img_h])
    assert len(boxes) == len(labels), "boxes and labels must have same length"
    abs_boxes = []
    for box, label in zip(boxes, labels):
        x_c, y_c, w, h = box
        box = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)] * scale_fct
        abs_boxes.append([int(x) for x in box])
    return abs_boxes