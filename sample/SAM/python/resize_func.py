import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def resize_longest_image_size(
    input_image_size: torch.Tensor, longest_side: int
) -> torch.Tensor:
    input_image_size = input_image_size.to(torch.float32)
    scale = longest_side / torch.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
    return transformed_size

def mask_postprocessing(img_size, masks: torch.Tensor, orig_im_size: torch.Tensor) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    prepadded_size = resize_longest_image_size(orig_im_size, img_size).to(torch.int64)
    masks = masks[..., : prepadded_size[0], : prepadded_size[1]]  # type: ignore

    orig_im_size = orig_im_size.to(torch.int64)
    h, w = orig_im_size[0], orig_im_size[1]
    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
    return masks


def resize(img_size,masks,orig_im_size):
    upscaled_masks = mask_postprocessing(img_size,masks, orig_im_size)
    return upscaled_masks


'''
test resize (1,1,1024,1024) to (1,1,1200,1800)
'''
# masks =  torch.tensor(np.random.rand(1,1,1024,1024))

# orig_im_size = torch.tensor([1200,1800])
# # img_size = model.image_encoder.img_size
# img_size = 1024 # img_size (int): Input image size.
# resize(img_size,masks,orig_im_size)