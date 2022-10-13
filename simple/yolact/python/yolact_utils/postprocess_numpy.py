import numpy as np
import cv2

class PostProcess:
    def __init__(self, cfg, conf_thresh=0.5, nms_thresh=0.5, keep_top_k=200):
        self.cfg = cfg

        self.conf_thresh = conf_thresh
        self.nms_thres = nms_thresh
        self.keep_top_k = keep_top_k
        self.width = cfg['width']
        self.height = cfg['height']
        self.conv_ws = cfg['conv_ws']
        self.conv_hs = cfg['conv_hs']
        self.aspect_ratios = cfg['aspect_ratios']
        self.scales = cfg['scales']
        self.variances = cfg['variances']
        self.priors = self.make_priors()

    def make_priors(self):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        prior_data = []

        for conv_w, conv_h, scale in zip(self.conv_ws, self.conv_hs, self.scales):
            for i in range(conv_h):
                for j in range(conv_w):
                    # +0.5 because priors are in center-size notation
                    cx = (j + 0.5) / conv_w
                    cy = (i + 0.5) / conv_h

                    for ar in self.aspect_ratios:
                        ar = np.sqrt(ar)

                        w = scale * ar / self.width
                        h = scale / ar / self.height

                        # This is for backward compatability with a bug where I made everything square by accident
                        h = w

                        prior_data += [cx, cy, w, h]

        self.priors = np.array(prior_data).reshape(-1, 4)
        return self.priors

    def decode(self, loc, priors, img_w, img_h):
        boxes = np.concatenate(
            (
                priors[:, :2] + loc[:, :2] * self.variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * self.variances[1]),
            ),
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        # boxes[:, 2:] += boxes[:, :2]

        # crop
        np.where(boxes[:, 0] < 0, 0, boxes[:, 0])
        np.where(boxes[:, 1] < 0, 0, boxes[:, 1])
        np.where(boxes[:, 2] > 1, 1, boxes[:, 2])
        np.where(boxes[:, 3] > 1, 1, boxes[:, 3])

        # decode to img size
        boxes[:, 0] *= img_w
        boxes[:, 1] *= img_h
        boxes[:, 2] = boxes[:, 2] * img_w + 1
        boxes[:, 3] = boxes[:, 3] * img_h + 1
        return boxes

    def sanitize_coordinates_numpy(self, _x1, _x2, img_size, padding=0):
        # _x1 = _x1 * img_size
        # _x2 = _x2 * img_size
        x1 = np.minimum(_x1, _x2)
        x2 = np.maximum(_x1, _x2)
        x1 = np.clip(x1 - padding, a_min=0, a_max=1000000)
        x2 = np.clip(x2 + padding, a_min=0, a_max=img_size)
        return x1, x2

    def crop_numpy(self, masks, boxes, padding=1):
        h, w, n = masks.shape
        x1, x2 = self.sanitize_coordinates_numpy(boxes[:, 0], boxes[:, 2], w, padding)
        y1, y2 = self.sanitize_coordinates_numpy(boxes[:, 1], boxes[:, 3], h, padding)

        rows = np.tile(np.arange(w)[None, :, None], (h, 1, n))
        cols = np.tile(np.arange(h)[:, None, None], (1, w, n))

        masks_left = rows >= (x1.reshape(1, 1, -1))
        masks_right = rows < (x2.reshape(1, 1, -1))
        masks_up = cols >= (y1.reshape(1, 1, -1))
        masks_down = cols < (y2.reshape(1, 1, -1))

        crop_mask = masks_left * masks_right * masks_up * masks_down

        return masks * crop_mask


    def after_nms_numpy(self, box_p, coef_p, proto_p, img_h, img_w, cfg=None):
        def np_sigmoid(x):
            return 1 / (1 + np.exp(-x))

        masks = np_sigmoid(np.matmul(proto_p, coef_p.T))

        if True:
            masks = self.crop_numpy(masks, box_p)

        ori_size = img_w, img_h
        masks = cv2.resize(masks, ori_size, interpolation=cv2.INTER_LINEAR)
        if masks.ndim == 2:
            masks = masks[:, :, None]

        masks = np.transpose(masks, (2, 0, 1))
        masks = masks > 0.5  # Binarize the masks because of interpolation.
        masks = masks[:, 0: img_h, :] if img_h < img_w else masks[:, :, 0: img_w]

        return box_p, masks


    def __call__(self, loc_data, conf_preds, mask_data, proto_data, org_size):
        """
        post-processing
        Args:
            loc_data: (19248, 4) or (1, 19248, 4) when yolact_base 550
            conf_preds: (19248, 81) or (1, 19248, 81) when yolact_base 550
            mask_data: (19248, 32) or (1, 19248, 32) when yolact_base 550
            proto_data: (138, 138, 32) or (1, 138, 138, 32) when yolact_base 550
            org_size: (org_w, org_h), original image size

        Returns: classid, conf_scores, boxes, masks for a image

        """
        img_w, img_h = org_size
        if loc_data.ndim == 3:
            loc_data = loc_data.squeeze(0)
            conf_preds = conf_preds.squeeze(0)
            mask_data = mask_data.squeeze(0)
            proto_data = proto_data.squeeze(0)

        cur_scores = conf_preds[:, 1:]
        classid = np.argmax(cur_scores, axis=1)
        conf_scores = cur_scores[range(cur_scores.shape[0]), classid]

        # filte by conf_thresh
        keep = conf_scores > self.conf_thresh
        conf_scores = conf_scores[keep]
        classid = classid[keep]
        loc_data = loc_data[keep, :]
        prior_data = self.priors[keep, :]
        masks = mask_data[keep, :]

        boxes = self.decode(loc_data, prior_data, img_w, img_h)

        if len(boxes) == 0:
            return [],[],[],[]

        # nms per category
        unique_id = np.unique(classid)

        new_classid = []
        new_conf_scores = []
        new_boxes = []
        new_masks = []
        for i,cls in enumerate(unique_id):
            cls_loc = (classid == cls)
            classid_cls = classid[cls_loc]
            conf_scores_cls = conf_scores[cls_loc]
            boxes_cls = boxes[cls_loc]
            masks_cls = masks[cls_loc]

            ind_nms = cv2.dnn.NMSBoxes(boxes_cls.tolist(), conf_scores_cls.tolist(),
                                   self.conf_thresh, self.nms_thres, top_k=self.keep_top_k)
            # opencv return (5,1) or (5, ) in different version
            if len(ind_nms.shape) == 2:
                ind_nms = ind_nms.squeeze(1)

            classid_cls, conf_scores_cls, boxes_cls, masks_cls = classid_cls[ind_nms], conf_scores_cls[ind_nms], boxes_cls[ind_nms], masks_cls[ind_nms]
            new_classid += [classid_cls]
            new_conf_scores += [conf_scores_cls]
            new_boxes += [boxes_cls]
            new_masks += [masks_cls]

        classid = np.concatenate(new_classid)
        conf_scores = np.concatenate(new_conf_scores)
        boxes = np.concatenate(new_boxes)
        masks = np.concatenate(new_masks)

        masks = np.matmul(proto_data, masks.T)
        masks = 1 / (1 + np.exp(-masks))
        # Scale masks up to the full image
        masks = cv2.resize(masks, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        # [h,w,n], but get [h,w] when masks shape is [h,w,1]
        if len(masks.shape) != 3:
            masks = masks[:, :, np.newaxis]
        xyxy = boxes.copy()
        xyxy[:, 2] += xyxy[:, 0]
        xyxy[:, 3] += xyxy[:, 1]
        masks = self.crop_numpy(masks, xyxy)
        masks = masks > 0.5  # Binarize the masks because of interpolation.

        return classid, conf_scores, boxes, masks

    def infer_batch(self, preds_batch, org_size_list):
        """
        batch post-processing
        Args:
            preds_batch: (loc_data_batch, conf_preds_batch, mask_data_batch, proto_data_batch)
                            loc_data_batch: (n, 19248, 4) when yolact_base 550
                            conf_preds_batch: (n, 19248, 81) when yolact_base 550
                            mask_data_batch: (n, 19248, 32) when yolact_base 550
                            proto_data_batch: (n, 138, 138, 32) when yolact_base 550
            org_size_list: [(org_w, org_h), ...], a list of original images size, length:n

        Returns: classid_list, conf_scores_list, boxes_list, masks_list for every image

        """
        classid_list, conf_scores_list, boxes_list, masks_list = [], [], [], []
        loc_data_batch, conf_preds_batch, mask_data_batch, proto_data_batch = preds_batch
        for i in range(len(org_size_list)):
            loc_data = loc_data_batch[i]
            conf_preds = conf_preds_batch[i]
            mask_data = mask_data_batch[i]
            proto_data = proto_data_batch[i]
            org_size = org_size_list[i]
            classid, conf_scores, boxes, masks = self(loc_data,
                                                      conf_preds,
                                                      mask_data,
                                                      proto_data,
                                                      org_size)
            classid_list.append(classid)
            conf_scores_list.append(conf_scores)
            boxes_list.append(boxes)
            masks_list.append(masks)
        return classid_list, conf_scores_list, boxes_list, masks_list