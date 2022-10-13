import os
import sys
import cv2
import time
import json
import logging
import numpy as np
from pprint import pprint
from PIL import Image, ImageDraw, ImageFont

def checkfolder(path):
    if not os.path.isdir(path): os.mkdir(path)

def timetest(input_func):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = input_func(*args, **kwargs)
        end_time = time.time()
        #print "Method Name - {0}, Args - {1}, Kwargs - {2}, Execution Time - {3}".format(input_func.__name__, args, kwargs, end_time - start_time)
        print("Method Name - {0}, Execution Time - {1}".format(input_func.__name__, end_time - start_time))
        return result
    return timed

def draw_bboxes(frame, bboxes_xyxy):
    if bboxes_xyxy is not None:
        for i in range(bboxes_xyxy.shape[0]):
            x1, y1, x2, y2 = bboxes_xyxy[i]
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            # score = round(scores[i], 2)
            # draw_zh_cn(frame, str(score), (0,255,0), (x1, y1-5))
    return frame

def draw_zh_cn(frame, string, color, position):
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)

    draw = ImageDraw.Draw(pil_im)
    size_font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc", 20)
    draw.text(position, string, color, font=size_font)

    img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    return img

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0: return (None, None)      # lines are parallel
    return (x/z, y/z)

def use_heatmap(image, box_centers):
    """http://jjguy.com/heatmap/
    https://github.com/jjguy/heatmap/issues/23
    """
    import heatmap
    hm = heatmap.Heatmap()
    box_centers = [(i, image.shape[0] - j) for i, j in box_centers]
    img = hm.heatmap(
            points=box_centers, 
            dotsize=100, 
            size=(image.shape[1], image.shape[0]),
            scheme='classic',
            opacity=128, 
            area=((0, 0), (image.shape[1], image.shape[0]))
            )
    return img


def frame2video():
    import glob
    import os
    from tqdm import tqdm
    from os.path import splitext, basename, isdir
    from ipdb import set_trace as pause

    img_paths = sorted(glob.glob("/mnt/ny/video/2019-0809-1757-54-1-fpn-kalman-auto-100/frame/**.png"))

    video_path = "output/video/"
    frameRate = 5
    bname = splitext(basename(video_path))[0]
    outputFile = os.path.join(video_path, "demo.avi".format(frameRate))
    fourcc =  cv2.VideoWriter_fourcc(*'MJPG')  ## MJPG MP4V 
    output = cv2.VideoWriter(outputFile, fourcc, frameRate, 
                (1280,720),
                # (2048,2048),
                )

    for idx, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        frame = cv2.imread(img_path)
        output.write(frame)


if __name__ == "__main__":
    pass
