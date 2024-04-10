#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
import os
import time
import torch
import cv2
import argparse
import logging
import sophon.sail as sail
from sam_encoder import SamEncoder
from predictor import SamPredictor
from sam_model import Sam
from automatic_mask_generator import SamAutomaticMaskGenerator
from torchvision.ops.boxes import batched_nms, box_area
import matplotlib.pyplot as plt
import logging
from amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
logging.basicConfig(level=logging.INFO)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, ax, marker_size=375):
    ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(coords[:, 0], coords[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def save_image_point(base_image,results,input_point, box = False): # results = (mask,score) or [(mask,score),(mask,score),(mask,score)]
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
    plt.savefig(output_dir+'/base_img.jpg', bbox_inches='tight', pad_inches=0)
    for i in range(len(results)):
        if i == 3:
            break
        res = results[i]
        show_mask(res[0], plt.gca())
        if not box:
            show_points(input_point, plt.gca())
        else:
            show_box(input_point[0], plt.gca())
        plt.title(f"Mask {i}, Score: {res[1]:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(output_dir+f'/result_box_{i}.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()
    
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

class SAM_b(object):
    def __init__(self, args):
        self.args = args
        # load bmodel
        self.net = sail.Engine(self.args.decode_bmodel, self.args.dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)

        self.input_shapes = [self.net.get_input_shape(self.graph_name, name) for name in self.input_names]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_shapes = [self.net.get_output_shape(self.graph_name, name) for name in self.output_names]

        
        logging.debug("load {} success!".format(self.args.decode_bmodel))
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_names, self.input_shapes)))
        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_names, self.output_shapes)))

        self.input_shape = self.input_shapes[0]
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.orig_im_size = []
        self.image_size = 1024 #
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0


    def preprocess(self, img, sam_encoder,sam):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor = SamPredictor(sam_encoder, sam)
        predictor.set_image(img)

        # use TPU to embedding input_image
        image_embedding = predictor.get_image_embedding() 
        assert len(np.array(list(map(int, self.args.input_point.split(','))))) == 2 or len(np.array(list(map(int, self.args.input_point.split(','))))) == 4, "input coordinate length must be 2 or 4"
        # point input
        if (len(np.array(list(map(int, self.args.input_point.split(','))))) == 2):
            input_point = np.array([list(map(int, self.args.input_point.split(',')))])
            input_label = np.array([1])
            ori_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            ori_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
            ori_coord = predictor.transform.apply_coords(ori_coord, img.shape[:2]).astype(np.float32)
            ori_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            ori_has_mask_input = np.zeros(1, dtype=np.float32)
            """
            All inputs are `np.float32`.
            * `image_embeddings`: The image embedding from `predictor.get_image_embedding()`. Has a batch index of length 1.
            * `point_coords`: Coordinates of sparse input prompts, corresponding to both point inputs and box inputs. Boxes are encoded using two points, one for the top-left corner and one for the bottom-right corner. *Coordinates must already be transformed to long-side 1024.* Has a batch index of length 1.
            * `point_labels`: Labels for the sparse input prompts. 0 is a negative input point, 1 is a positive input point, 2 is a top-left box corner, and 3 is a bottom-right box corner.*
            * `mask_input`: A mask input to the model with shape 1x1x256x256. This must be supplied even if there is no mask input. In this case, it can just be zeros.
            * `has_mask_input`: An indicator for the mask input. 1 indicates a mask input, 0 indicates no mask input.
            * `orig_im_size`: The size of the input image in (H,W) format, before any transformation.
            """           
            ort_inputs = {
                "image_embeddings": image_embedding,
                "point_coords": ori_coord,
                "point_labels": ori_label,
                "mask_input": ori_mask_input,
                "has_mask_input": ori_has_mask_input,
                "orig_im_size": np.array(img.shape[:2], dtype=np.float32)
                        }
            self.orig_im_size = ort_inputs["orig_im_size"]
        # box input
        else:
            input_point = np.array(list(map(int, self.args.input_point.split(',')))).reshape(2, 2)
            input_label = np.array([2,3])
            ori_coord = input_point[None, :, :]
            ori_label = input_label[None, :].astype(np.float32)
            ori_coord = predictor.transform.apply_coords(ori_coord, img.shape[:2]).astype(np.float32)
            ori_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            ori_has_mask_input = np.zeros(1, dtype=np.float32)

            ort_inputs = {
                "image_embeddings": image_embedding,
                "point_coords": ori_coord,
                "point_labels": ori_label,
                "mask_input": ori_mask_input,
                "has_mask_input": ori_has_mask_input,
                "orig_im_size": np.array(img.shape[:2], dtype=np.float32)
                        }
            self.orig_im_size = ort_inputs["orig_im_size"]
        return ort_inputs

    def predict(self, input_img):
        input_data = {self.input_names[0]: input_img['image_embeddings'], self.input_names[1]: input_img['point_coords'],
                      self.input_names[2]: input_img['point_labels'], self.input_names[3]: input_img['mask_input'],
                      self.input_names[4]: input_img['has_mask_input'], self.input_names[5]: input_img['orig_im_size']}
        outputs = self.net.process(self.graph_name, input_data)
        return outputs

    def postprocess(self, outputs_0):
        '''
        4 output bmodel, resize masks on cpu
        '''
        mask_outputs = list(outputs_0.items())[1][1] # '/Slice_9_output_0_Slice_f32'
        mask_scores = list(outputs_0.items())[2][1] # 'low_res_masks_Reshape_f32'

        if mask_outputs.shape[1] == 1:
            print("your bmodel is one mask output")
            mask = np.squeeze(mask_outputs, axis=(0, 1))
            upscaled_masks = cv2.resize(mask, (self.orig_im_size[1].astype(int),self.orig_im_size[0].astype(int)))
            return [(upscaled_masks > 0.0 , mask_scores[0][0])]  
        elif mask_outputs.shape[1] == 4:
            print("your bmodel is three mask output")
            res = []
            for i, (mask_output, score) in enumerate(zip(mask_outputs[0], mask_scores[0])):
                upscaled_masks = cv2.resize(mask_output, (self.orig_im_size[1].astype(int),self.orig_im_size[0].astype(int)))
                res.append((upscaled_masks > 0.0 , score))  
            return res
        
    def auto_mask(self, image, sam_encoder,sam):
        start_time = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor = SamPredictor(sam_encoder, sam)
        """
        There are several tunable parameters in automatic mask generation that control how densely points are sampled and what the thresholds are for removing low quality or duplicate masks.
        * `points_per_side`(int): The number of points to sample along one side of the image. The total number of points is points_per_side2^2. Default is 32.
        * `points_per_batch`(int): Set the number of points that the model detects simultaneously. Higher numbers may be faster but use more GPU memory. Default is 64.
        * `pred_iou_thresh`(float): Filtering threshold in [0,1], model’s prediction mask quality. The default value is 0.88.
        * `stability_score_thresh`(float): The filtering threshold in [0,1], the stability of the mask under changes in cutoff values, is used to binarize the model's mask predictions. The default value is 0.95.
        * `stability_score_offset`(float): The amount by which the cutoff value is offset when calculating stability scores. The default value is 1.0.
        * `box_nms_thresh`(float): Non-maximum suppression box IoU cutoff for filtering duplicate masks. The default value is 0.7.
        * `crop_nms_thresh`(float): Box IoU cutoff for non-maximum suppression to filter duplicate masks between different objects. The default value is 0.7.
        * `crop_overlap_ratio`(float): Sets how much objects overlap. In the first crop layer, the crop will overlap this portion of the image's length. Later layers with more objects will reduce this overlap. Default value is 512/1500.
        * `crop_n_points_downscale_factor`(int): The number of points on each side sampled in layer n is scaled down by "crop_n_points_downscale_factorn"^n. The default value is 1.
        * `min_mask_region_area`(int): If >0, post-processing will be applied to remove masks with areas smaller than "min_mask_region_area" to break disconnected regions and holes. Opencv is required. Default is 0.
        * `output_mode`(str): Mask output mode. Can be 'binary_mask', 'uncompressed_rle' or 'coco_rle', 'coco_rle' requires pycocotools. For large resolutions, 'binary_mask' may consume a lot of memory. Default is 'binary_mask'.
        """  
        mask_generator = SamAutomaticMaskGenerator(sam_encoder, sam, points_per_side=self.args.points_per_side,
                                                                     points_per_batch=self.args.points_per_batch,
                                                                     pred_iou_thresh=self.args.pred_iou_thresh,
                                                                     stability_score_thresh=self.args.stability_score_thresh,
                                                                     stability_score_offset=self.args.stability_score_offset,
                                                                     box_nms_thresh=self.args.box_nms_thresh,
                                                                     crop_nms_thresh=self.args.crop_nms_thresh,
                                                                     crop_overlap_ratio=self.args.crop_overlap_ratio,
                                                                     crop_n_points_downscale_factor=self.args.crop_n_points_downscale_factor,
                                                                     min_mask_region_area=self.args.min_mask_region_area,
                                                                     output_mode=self.args.output_mode)
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, mask_generator.crop_n_layers, mask_generator.crop_overlap_ratio
        )
        # Iterate over image crops
        data_mask = MaskData()
        self.preprocess_time += time.time() - start_time

        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            start_time = time.time()
            x0, y0, x1, y1 = crop_box
            cropped_im = image[y0:y1, x0:x1, :]
            cropped_im_size = cropped_im.shape[:2]  
            predictor.set_image(cropped_im)
            image_embedding = predictor.get_image_embedding()

        # Get points for this crop
            points_scale = np.array(cropped_im_size)[None, ::-1]
            points_for_image = mask_generator.point_grids[layer_idx] * points_scale
            data_crop = MaskData()
            self.preprocess_time += time.time() - start_time
        # Generate masks for this crop in batches
            for i, (points,) in enumerate(batch_iterator(mask_generator.points_per_batch, points_for_image)):

                # Run model on this batch
                start_time = time.time()
                transformed_points = mask_generator.predictor.transform.apply_coords(points, cropped_im_size)
                in_points = torch.as_tensor(transformed_points, device=mask_generator.predictor.device)
                in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
                onnx_coord = np.array(in_points)[:, None, :]
                onnx_label = np.array(in_labels)[ :, None].astype(np.float32)
                onnx_coord = predictor.transform.apply_coords(onnx_coord, cropped_im.shape[:2]).astype(np.float32)
                onnx_mask_input = np.zeros((64, 1, 256, 256), dtype=np.float32)
                onnx_has_mask_input = np.zeros(1, dtype=np.float32)

                ort_inputs = {
                    "image_embeddings": image_embedding,
                    "point_coords": onnx_coord,
                    "point_labels": onnx_label,
                    "mask_input": onnx_mask_input,
                    "has_mask_input": onnx_has_mask_input,
                    "orig_im_size": np.array(cropped_im.shape[:2], dtype=np.float32)
                }
                
                self.preprocess_time += time.time() - start_time

                start_time = time.time()
                output_mask = self.predict(ort_inputs)
                self.inference_time += time.time() - start_time
                logging.debug("{} masks finish!".format(i))

                start_time = time.time()
                low_res_logits = output_mask[list(output_mask.keys())[3]]
                iou_preds = output_mask[list(output_mask.keys())[1]]
                low_res_logits = torch.from_numpy(low_res_logits)

                input_image = predictor.transform.apply_image(image)
                input_image_torch = torch.as_tensor(input_image, device=predictor.device)
                input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
                low_res_logits = sam.postprocess_masks(low_res_logits,tuple(input_image_torch.shape[-2:]),image.shape[:2] )

                masks = low_res_logits

                iou_preds = torch.from_numpy(iou_preds)

                # Serialize predictions and store in MaskData
                data_batch = MaskData(
                    masks=masks.flatten(0, 1),
                    iou_preds=iou_preds.flatten(0, 1),
                    points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
                )
                del masks

                # Filter by predicted IoU
                if mask_generator.pred_iou_thresh > 0.0:
                    keep_mask = data_batch["iou_preds"] > mask_generator.pred_iou_thresh
                    data_batch.filter(keep_mask)

                # Calculate stability score
                data_batch["stability_score"] = calculate_stability_score(
                    data_batch["masks"], mask_generator.predictor.model.mask_threshold, mask_generator.stability_score_offset
                )
                if mask_generator.stability_score_thresh > 0.0:
                    keep_mask = data_batch["stability_score"] >= mask_generator.stability_score_thresh
                    data_batch.filter(keep_mask)

                # Threshold masks and calculate boxes
                data_batch["masks"] = data_batch["masks"] > mask_generator.predictor.model.mask_threshold
                data_batch["boxes"] = batched_mask_to_box(data_batch["masks"])

                # Filter boxes that touch crop boundaries
                orig_h, orig_w = orig_size
                keep_mask = ~is_box_near_crop_edge(data_batch["boxes"], crop_box, [0, 0, orig_w, orig_h])
                if not torch.all(keep_mask):
                    data_batch.filter(keep_mask)

                # Compress to RLE
                data_batch["masks"] = uncrop_masks(data_batch["masks"], crop_box, orig_h, orig_w)
                data_batch["rles"] = mask_to_rle_pytorch(data_batch["masks"])
                del data_batch["masks"]

                data_crop.cat(data_batch)

                del data_batch
                self.postprocess_time += time.time() - start_time
            
            start_time = time.time()
            mask_generator.predictor.reset_image()
            keep_by_nms = batched_nms(
                data_crop["boxes"].float(),
                data_crop["iou_preds"],
                torch.zeros_like(data_crop["boxes"][:, 0]),  # categories
                iou_threshold=mask_generator.box_nms_thresh,
            )
            data_crop.filter(keep_by_nms)

            # Return to the original image frame
            data_crop["boxes"] = uncrop_boxes_xyxy(data_crop["boxes"], crop_box)
            data_crop["points"] = uncrop_points(data_crop["points"], crop_box)
            data_crop["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data_crop["rles"]))])
            
            data_mask.cat(data_crop)   
            self.postprocess_time += time.time() - start_time
        start_time = time.time()
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data_mask["crop_boxes"])
            scores = scores.to(data_mask["boxes"].device)
            keep_by_nms_2 = batched_nms(
                data_mask["boxes"].float(),
                scores,
                torch.zeros_like(data_mask["boxes"][:, 0]),  # categories
                iou_threshold=mask_generator.crop_nms_thresh,
            )
            data_mask.filter(keep_by_nms_2)

        data_mask.to_numpy()

        # Filter small disconnected regions and holes in masks
        if mask_generator.min_mask_region_area > 0:
            data_mask = mask_generator.postprocess_small_regions(
                data_mask,
                mask_generator.min_mask_region_area,
                max(mask_generator.box_nms_thresh, mask_generator.crop_nms_thresh),
            )

        # Encode masks
        if mask_generator.output_mode == "coco_rle":
            data_mask["segmentations"] = [coco_encode_rle(rle) for rle in data_mask["rles"]]
        elif mask_generator.output_mode == "binary_mask":
            data_mask["segmentations"] = [rle_to_mask(rle) for rle in data_mask["rles"]]
        else:
            data_mask["segmentations"] = data_mask["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(data_mask["segmentations"])):
            ann = {
                "segmentation": data_mask["segmentations"][idx],
                "area": area_from_rle(data_mask["rles"][idx]),
                "bbox": box_xyxy_to_xywh(data_mask["boxes"][idx]).tolist(),
                "predicted_iou": data_mask["iou_preds"][idx].item(),
                "point_coords": [data_mask["points"][idx].tolist()],
                "stability_score": data_mask["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(data_mask["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)
        self.postprocess_time += time.time() - start_time
        
        # Save result
        output_dir = "./results"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(curr_anns)
        plt.axis('off')
        plt.savefig(output_dir+'/result_auto.jpg', bbox_inches='tight', pad_inches=0)

        return 0
    
    def __call__(self, img, sam_encoder, sam):
        if (self.args.auto == 0):
            start_time = time.time()
            img = self.preprocess(img, sam_encoder, sam)
            self.preprocess_time += time.time() - start_time
            
            start_time = time.time()
            outputs_0 = self.predict(img)
            self.inference_time += time.time() - start_time

            start_time = time.time()
            res = self.postprocess(outputs_0)
            self.postprocess_time += time.time() - start_time
            return res
        else:
            res =self.auto_mask(img, sam_encoder, sam)

        


def main(args):
    sam_vit_b = SAM_b(args)
    batch_size = sam_vit_b.batch_size
    sam_vit_b.init()

    # decode image
    start_time = time.time()
    src_img = cv2.imread(args.input_image)
    if src_img is None:
        logging.error("{} imread is None.".format(args.input_image))
    decode_time = time.time() - start_time

    # init sam and embedding bmodel to do preprocess 
    sam = Sam()
    sam_encoder = SamEncoder(args)

    # process images
    if args.auto == 0:
        results = sam_vit_b(src_img, sam_encoder, sam) # (mask,score) or [(mask,score),(mask,score),(mask,score)]

        # save processed image
        input_point = np.array([list(map(int, args.input_point.split(',')))])
        if len(input_point[0]) == 2:
            save_image_point(src_img,results,input_point, box = False)
        else:
            save_image_point(src_img,results,input_point, box = True)
    else:
        results = sam_vit_b(src_img, sam_encoder, sam)

    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    preprocess_time = sam_vit_b.preprocess_time  
    inference_time = sam_vit_b.inference_time  
    postprocess_time = sam_vit_b.postprocess_time  
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("embedding_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("decode_mask_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

    
def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input_image', type=str, default='datasets/truck.jpg', help='path of input, must be image directory')
    parser.add_argument('--input_point', type=str, default='700,375', help='The coordinates of the input_point(point or box), point in format x,y, box in format x1,y1,x2,y2')
    parser.add_argument('--embedding_bmodel', type=str, default='models/BM1684X/embedding_bmodel/SAM-ViT-B_embedding_fp16_1b.bmodel', help='path of bmodel')
    parser.add_argument('--decode_bmodel', type=str, default='models/BM1684X/decode_bmodel/SAM-ViT-B_decoder_multi_mask_fp16_1b.bmodel', help='path of bmodel')
    parser.add_argument('--auto', type=bool, default=0, help='Whether to use an automatic mask generator: 0 for no, 1 for yes')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    #auto parsers
    parser.add_argument('--points_per_side', type=int, default=32, help='see the description in the auto_mask method for details')
    parser.add_argument('--points_per_batch', type=int, default=64, help='see the description in the auto_mask method for details')
    parser.add_argument('--pred_iou_thresh', type=float, default=0.88, help='see the description in the auto_mask method for details')
    parser.add_argument('--stability_score_thresh', type=float, default=0.95, help='see the description in the auto_mask method for details')
    parser.add_argument('--stability_score_offset', type=float, default=1.0, help='see the description in the auto_mask method for details')
    parser.add_argument('--box_nms_thresh', type=float, default=0.7, help='see the description in the auto_mask method for details')
    parser.add_argument('--crop_nms_thresh', type=float, default=0.7, help='see the description in the auto_mask method for details')
    parser.add_argument('--crop_overlap_ratio', type=float, default=512/1500, help='see the description in the auto_mask method for details')
    parser.add_argument('--crop_n_points_downscale_factor', type=int, default=1, help='see the description in the auto_mask method for details')
    parser.add_argument('--min_mask_region_area', type=int, default=0, help='see the description in the auto_mask method for details')
    parser.add_argument('--output_mode', type=str, default='binary_mask', help='see the description in the auto_mask method for details')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)
