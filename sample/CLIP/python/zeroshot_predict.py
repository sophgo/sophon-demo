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
import cv2
import clip
import argparse
import logging
import numpy as np
import sys

# 创建一个处理器，将日志输出到 stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

# 创建一个格式化器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 获取根日志记录器，并添加处理器
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# 确保没有其他处理器
if logger.hasHandlers():
    logger.handlers.clear()
    logger.addHandler(handler)


def main(args):
    # Load bmodel
    text = args.text
    model, preprocess = clip.load(args.image_model, args.text_model, args.dev_id)
    
    # Tokenize 
    text_inputs = clip.tokenize(text)

    if os.path.isfile(args.image_path):
        image_paths = [args.image_path]
    else:
        image_paths = [os.path.join(args.image_path, fname) for fname in os.listdir(args.image_path)]
    
    for filename in image_paths:
        # Preprocess
        logging.info("Filename: {}".format(filename))
        image = cv2.imread(filename)
        image_input = np.expand_dims(preprocess(image), axis=0)
        # predict
        values, indices = model.predict(image_input, text_inputs)
        for i in range(len(text)):
            logging.info(f"Text: {text[indices[i]]}, Similarity: {values[i].item()}")


    image_num = len(image_paths)
    logging.info(("-------------------Image num {}, Preprocess average time ------------------------").format(image_num))
    logging.info("preprocess(ms): {:.2f}".format(model.preprocess_time / image_num * 1000))

    logging.info(("------------------ Image num {}, Image Encoding average time ----------------------").format(image_num))
    logging.info("image_encode(ms): {:.2f}".format(model.encode_image_time / image_num * 1000))

    logging.info(("------------------ Image num {}, Text Encoding average time ----------------------").format(image_num))
    logging.info("text_encode(ms): {:.2f}".format(model.encode_text_time / image_num * 1000))



def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--image_path', type=str, default='./datasets/CLIP.png', help='path of input')
    parser.add_argument('--text', nargs='+', default=['a diagram', 'a dog', 'a cat'], help='text of input')
    parser.add_argument('--image_model', type=str, default='./models/BM1684X/clip_image_vitb32_bm1684x_f16_1b.bmodel', help='path of image bmodel')
    parser.add_argument('--text_model', type=str, default='./models/BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel', help='path of text bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')