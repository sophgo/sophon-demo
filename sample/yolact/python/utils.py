import os
import numpy as np
import cv2

COCO_CLASSES = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')
 
COLORS = [
        [56, 0, 255],
        [226, 255, 0],
        [0, 94, 255],
        [0, 37, 255],
        [0, 255, 94],
        [255, 226, 0],
        [0, 18, 255],
        [255, 151, 0],
        [170, 0, 255],
        [0, 255, 56],
        [255, 0, 75],
        [0, 75, 255],
        [0, 255, 169],
        [255, 0, 207],
        [75, 255, 0],
        [207, 0, 255],
        [37, 0, 255],
        [0, 207, 255],
        [94, 0, 255],
        [0, 255, 113],
        [255, 18, 0],
        [255, 0, 56],
        [18, 0, 255],
        [0, 255, 226],
        [170, 255, 0],
        [255, 0, 245],
        [151, 255, 0],
        [132, 255, 0],
        [75, 0, 255],
        [151, 0, 255],
        [0, 151, 255],
        [132, 0, 255],
        [0, 255, 245],
        [255, 132, 0],
        [226, 0, 255],
        [255, 37, 0],
        [207, 255, 0],
        [0, 255, 207],
        [94, 255, 0],
        [0, 226, 255],
        [56, 255, 0],
        [255, 94, 0],
        [255, 113, 0],
        [0, 132, 255],
        [255, 0, 132],
        [255, 170, 0],
        [255, 0, 188],
        [113, 255, 0],
        [245, 0, 255],
        [113, 0, 255],
        [255, 188, 0],
        [0, 113, 255],
        [255, 0, 0],
        [0, 56, 255],
        [255, 0, 113],
        [0, 255, 188],
        [255, 0, 94],
        [255, 0, 18],
        [18, 255, 0],
        [0, 255, 132],
        [0, 188, 255],
        [0, 245, 255],
        [0, 169, 255],
        [37, 255, 0],
        [255, 0, 151],
        [188, 0, 255],
        [0, 255, 37],
        [0, 255, 0],
        [255, 0, 170],
        [255, 0, 37],
        [255, 75, 0],
        [0, 0, 255],
        [255, 207, 0],
        [255, 0, 226],
        [255, 245, 0],
        [188, 255, 0],
        [0, 255, 18],
        [0, 255, 75],
        [0, 255, 151],
        [255, 56, 0],
        [245, 255, 0],
    ]        

def draw_numpy(image, boxes, masks=None, classes_ids=None, conf_scores=None):
    used_colors = set()  # Used to track used colors
    for idx in range(len(boxes)):
        left, top, width, height = boxes[idx, :].astype(np.int32).tolist()
        
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) % len(COLORS)]
        else:
            color = (0, 0, 255)
        
        thickness = 2  # Bounding box line thickness
        
        cv2.rectangle(image, (left, top), (left + width, top + height), color, thickness=thickness)
        
        if masks is not None:
            mask = masks[:, :, idx]
            
            # Select different color to differentiate masks
            class_id = int(classes_ids[idx]) % len(COLORS)
            while class_id in used_colors:
                class_id = (class_id + 8) % len(COLORS)
            used_colors.add(class_id)
            color = COLORS[class_id]
            
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
            
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            
            text = COCO_CLASSES[classes_ids[idx] + 1] + ':' + str(round(conf_scores[idx], 2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8  # Text size
            text_thickness = 2  # Text line thickness
            
            # Calculate text position
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
            text_position = (left, top + height - 10)
            
            # Ensure text does not go beyond image boundary
            if text_position[0] + text_width > image.shape[1]:
                text_position = (left, top - 5)
            
            cv2.putText(image, text, text_position, font, font_scale, (0, 255, 0), thickness=text_thickness)

def draw_bmcv(bmcv, image, boxes, masks=None, classes_ids=None, conf_scores=None):
    for idx in range(len(boxes)):
        left, top, width, height = boxes[idx, :].astype(np.int32).tolist()
        bmcv.rectangle(image, left, top, width, height, (0, 0, 255), 2)

def is_img(file_name):
    """judge the file is available image or not
    Args:
        file_name (str): input file name
    Returns:
        (bool) : whether the file is available image
    """
    if os.path.splitext(file_name)[-1] in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
        return True
    else:
        return False