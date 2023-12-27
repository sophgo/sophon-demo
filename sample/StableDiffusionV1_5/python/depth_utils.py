import numpy as np
from scipy.ndimage import zoom
from PIL import Image

def _prepare_depth_image(controlnet_img, depth_processor):
    controlnet_img = controlnet_img.resize((384, 384),resample = 2, reducing_gap = None)
    controlnet_img = np.array(controlnet_img)
    controlnet_img = controlnet_img*0.00392156862745098
    controlnet_img = controlnet_img.astype(np.float32)
    mean = np.array([0.5,0.5,0.5], dtype = controlnet_img.dtype)
    std = np.array([0.5,0.5,0.5], dtype = controlnet_img.dtype)
    controlnet_img = (controlnet_img - mean) / std
    controlnet_img = controlnet_img.transpose((2,0,1))
    controlnet_img = controlnet_img.reshape(1, 3, 384, 384)

    controlnet_img = [controlnet_img]
    depth_map = depth_processor(controlnet_img)
    depth_map = depth_map[0]
    upscale_factor = 512 / 384
    controlnet_img = zoom(depth_map, zoom=(1, upscale_factor, upscale_factor), order=3)
    formatted = (controlnet_img * 255 / np.max(controlnet_img)).astype(np.uint8)
    controlnet_img = np.clip(formatted, 0, 255)
    controlnet_img = np.squeeze(controlnet_img)
    controlnet_img = Image.fromarray(controlnet_img)

    controlnet_img = controlnet_img.convert("RGB")
    # pil to numpy
    controlnet_img = np.array(controlnet_img).astype(np.float32) / 255.0
    controlnet_img = [controlnet_img]
    controlnet_img = np.stack(controlnet_img, axis = 0)

    # (batch, channel, height, width)
    controlnet_img = controlnet_img.transpose(0, 3, 1, 2)

    controlnet_img_copy = np.copy(controlnet_img)
    controlnet_img = np.concatenate((controlnet_img,controlnet_img_copy), axis = 0)
    return controlnet_img