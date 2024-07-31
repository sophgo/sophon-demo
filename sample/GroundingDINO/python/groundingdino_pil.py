import argparse
import os
import numpy as np
from transformers import BertTokenizerFast
from PostProcess import PostProcess
from utils import plot_boxes_to_image, generate_masks_with_special_tokens_and_transfer_map, gen_encoder_output_proposals

import time
from PIL import Image
import sophon.sail as sail

import logging
logging.basicConfig(level=logging.INFO)

def load_image(image):
     # Load image using PIL
    image_pil = image.convert("RGB")

    # Resize the image
    image_pil_resize = image_pil.resize((800, 800))

    # Convert PIL image to NumPy array
    image_np = np.array(image_pil_resize)

    # Normalize the image manually
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np / 255.0 - mean) / std

    # Permute dimensions (transpose) to match the order (C, H, W)
    image_np = np.transpose(image_np, (2, 0, 1))
    return image_pil, image_np

class GroundingDINO():
    def __init__(self, args):
        # input info
        self.bmodel_path = args.bmodel
        self.text_prompt = args.text_prompt
        self.tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)
        self.token_spans = args.token_spans 
        self.output_dir = args.output_dir
        self.text_threshold = args.text_threshold
        self.box_threshold = args.box_threshold
        self.token_spans = args.token_spans
        self.dev_id = args.dev_id

        # set the text_threshold to None if token_spans is set.
        if self.token_spans is not None:
            self.text_threshold = None
            print("Using token_spans. Set the text_threshold to None.")

        # load bmodel
        self.net = sail.Engine(self.bmodel_path, self.dev_id, sail.IOMode.SYSIO)
        logging.debug("load {} success!".format(self.bmodel_path))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)
        self.img_input_shape = self.net.get_input_shape(self.graph_name, self.input_name[0])
        
        # make dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            print("Output directory is not specified, please ignore")

        # check batch size
        self.batch_size = self.img_input_shape[0]
        if self.batch_size != 1:
            raise ValueError("GroundingDINO only support one batch_size")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # init postprocess
        self.postprocess = PostProcess(
            caption = self.text_prompt,
            token_spans = eval(f"{self.token_spans}"),
            tokenizer = self.tokenizer, 
            box_threshold = self.box_threshold, 
            text_threshold = self.text_threshold
        )

        # init time
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        # warm up decoder
        self.tokenizer.decode([0])
    def check(self):
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"The specified path '{self.image_path}' does not exist.")
        if self.net is None:
            raise FileNotFoundError(f"The specified path '{self.bmodel_path}' does not exist.")
        if self.text_prompt is None:
            raise ValueError("text prompt should not be None!")
        
        assert self.text_threshold is not None or self.token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    
    def decode(self, img_path):
        self.img = Image.open(img_path)
    
    def preprocess(self, tokenizer, captions, np_image=None):
        if np_image is None:
            self.image_pil, samples = load_image(self.img)
            samples = samples[None, :, :, :]
        else:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            samples = (np_image / 255.0 - mean) / std
            samples = np.transpose(samples, (2, 0, 1))
            samples = samples[None, :, :, :]

        captions = captions.lower().strip()
        if not captions.endswith("."):
            captions += "."

        captions = [captions]

        max_text_len = 256

        # Tokenize and generate masks
        tokenized = tokenizer(captions, padding="max_length", return_tensors="np")
        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(tokenized)

        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
            position_ids = position_ids[:, :max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :max_text_len]

        # Extract relevant information
        text_token_mask = tokenized["attention_mask"].astype(bool)
        input_ids, token_type_ids, attention_mask = tokenized["input_ids"], tokenized["token_type_ids"], text_self_attention_masks
        proposals = gen_encoder_output_proposals()

        # Convert tensors to numpy arrays
        data = [samples, position_ids,
            text_self_attention_masks,
            input_ids,
            token_type_ids,
            attention_mask,
            text_token_mask,
            proposals
        ]

        return data

    def __call__(self, data):
        if isinstance(data, list):
            values = data
        elif isinstance(data, dict):
            values = list(data.values())
        else:
            raise TypeError("data is not list or dict")
        data = {}
        for i in range(len(values)):
            data[self.input_name[i]] = values[i]
        
        output = self.net.process(self.graph_name, data)
        res = []

        for name in self.output_names:
            res.append(output[name])
        return res

def main(args):

    # Initial
    groundingdino = GroundingDINO(args)
    groundingdino.init()

    # decode
    decode_start = time.time()
    groundingdino.decode(args.image_path)
    decode_time = time.time() - decode_start

    # preprocess
    preprocess_start = time.time()
    data = groundingdino.preprocess(groundingdino.tokenizer, groundingdino.text_prompt)
    preprocess_time = time.time() - preprocess_start

    # visualize raw image
    groundingdino.image_pil.save(os.path.join(args.output_dir, "raw_image.jpg"))
    
    # Inference
    inference_start = time.time()

    output = groundingdino(data)
    inference_time = time.time() - inference_start

    # PostProcess
    postprocess_start = time.time()
    
    boxes_filt, pred_phrases = groundingdino.postprocess(output)
    postprocess_time = time.time() - postprocess_start

    # visualize pred
    pred_dict = {
        "boxes": boxes_filt,
        "labels": pred_phrases,
    }
    
    # PIL format
    image_with_box = plot_boxes_to_image(groundingdino.image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(groundingdino.output_dir, "pred_bmodel_new.jpg"))
    print("Image was save in {}".format(groundingdino.output_dir))
    
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

def argsparser():
    parser = argparse.ArgumentParser("Grounding DINO", add_help=True)
    parser.add_argument(
        "--bmodel", "-p", type=str, required=False, default="../models/BM1684X/groundingdino_bm1684x_fp16.bmodel",help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, default="../datasets/test/zidane.jpg", help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, default="person", help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="./results", required=False, help="output directory"
    )
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--dev_id", type=int, default=0, help="TPU id")
    parser.add_argument("--tokenizer_path", type=str, default="../models/bert-base-uncased", help="tokenizer path")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
