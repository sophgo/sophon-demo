import torch
from controlnet_aux import OpenposeDetector
import numpy as np
import cv2
import os

save_dir = "processors"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = "cpu"

img_size = [(128, 384), (128, 448), (128, 512), (192, 384), (192, 448), (192, 512), (256, 384), 
            (256, 448), (256, 512), (320, 384), (320, 448), (320, 512), (384, 384), (384, 448), 
            (384, 512), (448, 448), (448, 512), (512, 512), (512, 576), (512, 640), (512, 704),
            (512, 768), (512, 832), (512, 896), (768, 768), (384, 128), (448, 128), (512, 128),
            (384, 192), (448, 192), (512, 192), (384, 256), (448, 256), (512, 256), (384, 320),
            (448, 320), (512, 320), (448, 384), (512, 384), (512, 448), (576, 512), (640, 512), 
            (704, 512), (768, 512), (832, 512), (896, 512)]

resolution = 512

processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

def export_body_processor():
    openpose_body_processor = processor.body_estimation.model
    openpose_body_processor.eval()

    for parameter in openpose_body_processor.parameters():
        parameter.requires_grad=False

    for img_height, img_width in img_size:

        fake_img = np.random.random((img_height, img_width, 3)).astype(np.float32)

        fake_img = resize_image(fake_img, resolution)

        fake_img = fake_img[:, :, ::-1].copy()

        scale = 0.5 * 368 / fake_img.shape[0]

        imageToTest = cv2.resize(fake_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        imageToTest_padded, pad = padRightDownCorner(imageToTest, 8, 0)
        fake_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1))

        B, C, H, W = fake_img.shape[0], fake_img.shape[1], fake_img.shape[2], fake_img.shape[3]
        fake_input = torch.randn(B, C, H, W).to(torch.float32).to(device)

        def build_body_flow(input):
            with torch.no_grad():
                paf, heat =openpose_body_processor(input)
            return paf, heat

        traced_model=torch.jit.trace(build_body_flow, (fake_input))
        traced_model.save(f"./{save_dir}/openpose_body_processor_{H}_{W}.pt")

def export_hand_processor():
    openpose_hand_processor = processor.hand_estimation.model
    openpose_hand_processor.eval()

    for parameter in openpose_hand_processor.parameters():
        parameter.requires_grad=False

    fake_input = torch.randn(1, 3, 184, 184).to(torch.float32).to(device)

    def build_hand_flow(input):
        with torch.no_grad():
            out = openpose_hand_processor(input)
        return out 

    traced_model=torch.jit.trace(build_hand_flow, (fake_input))
    traced_model.save(f"./{save_dir}/openpose_hand_processor.pt")

def export_face_processor():
    openpose_face_processor = processor.face_estimation.model
    openpose_face_processor.eval()
    for parameter in openpose_face_processor.parameters():
        parameter.requires_grad=False

    input = torch.randn(1,3,384,384).to(torch.float32).to(device)
    def build_face_flow(image):
        with torch.no_grad():
            heatmaps = openpose_face_processor(image)
        return heatmaps[5]
    traced_model=torch.jit.trace(build_face_flow, (input))
    traced_model.save(f"./{save_dir}/openpose_face_processor.pt")

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

export_body_processor()
export_hand_processor()
export_face_processor()