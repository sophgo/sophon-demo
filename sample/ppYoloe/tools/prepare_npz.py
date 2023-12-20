import numpy as np
import cv2
import glob
import os

# ----------------------------------------------config---------------------------------------------------
# image format
img_extension = ['.jpg','.jpeg','.png']

# shape of the input image
img_target_shape = (640, 640)

# dataset used to generate npz files
dataset_path = "../datasets/coco128"

# mean and std
channel_means = np.array([0.485, 0.456, 0.406])
channel_stds = np.array([0.229, 0.224, 0.225])

# npz path
npz_path = dataset_path+"_npz"
os.makedirs(npz_path)
# ----------------------------------------------config end------------------------------------------------

imgs_path = []

for ext in img_extension:
    imgs_path.extend(glob.glob(os.path.join(dataset_path,"*"+ext)))

for img_path in imgs_path:

#   image file name
    img_name = os.path.basename(img_path)
    img_name_withoutEXT = os.path.splitext(img_name)[0]

#   image input
    img_bgr_uint8 = cv2.imread(img_path)
    img_rgb_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
    img_resized_uint8 = cv2.resize(img_rgb_uint8, img_target_shape)
    img_resized_fp32 = img_resized_uint8.astype("float32")  # change data format
    img = img_resized_fp32.transpose((2, 0 , 1))    # change (height, width, channel) to (channel, height, width)
    img = img/255.  # (0, 255) to (0, 1.)
    normalized_img = (img - channel_means[:, np.newaxis, np.newaxis])/channel_stds[:, np.newaxis, np.newaxis]   # normalize
    normalized_img = normalized_img[np.newaxis, ...]  # add new axis, (batch, channel, height, width)

#   ratio input
    img_ratio_h = img_target_shape[0]/float(img_rgb_uint8.shape[0])
    img_ratio_w = img_target_shape[1]/float(img_rgb_uint8.shape[1])

#   generate npz files
    img_npz = {}
    img_npz['image'] = normalized_img
    img_npz['scale_factor'] = np.array(([img_ratio_h, img_ratio_w],)).astype('float32')
    npz_save_path = os.path.join(npz_path, img_name_withoutEXT+".npz")
    np.savez(npz_save_path, **img_npz)