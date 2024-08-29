import os

import cv2
from pycocotools.coco import COCO

__all__ = ["COCODataset"]


class COCODataset:

    def __init__(self, img_path, annos_path, num=10):

        if not os.path.exists(annos_path):
            raise FileNotFoundError("找不到该annotation文件!")
        if not os.path.isdir(img_path):
            raise FileNotFoundError("数据集路径有误!")

        self.root = img_path
        self.coco = COCO(annos_path)
        self.image_ids = self.coco.getImgIds()
        self.det_num = num

    def __len__(self):
        return self.det_num

    def __getitem__(self, index):
        if index == self.det_num:
            raise StopIteration
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)
        img_name = image_info[0]["file_name"]
        img_path = os.path.join(self.root, img_name)
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        centers_info = []
        for annotation_id in annotation_ids:
            annotations = self.coco.loadAnns(annotation_id)
            for annotation in annotations:
                bbox = annotation["bbox"]
                label = annotation["category_id"]
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                centers_info.append(
                    {
                        "bbox": bbox,
                        "center": [center_x, center_y],
                        "label": label,
                    }
                )
        img = cv2.imread(img_path)
        # print("Image shape in dataset: ", img.shape)
        return {
            "img": img,
            "image_id": image_id,
            "centers_info": centers_info,
        }
