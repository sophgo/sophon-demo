#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
#!/bin/python3
import cv2
import logging
import sys
import os
import time
import argparse

from blip import blip_cap, init_tokenizer, preprocess

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
    # Initial
    model = blip_cap(args)
    image_size = 384

    # Tokenize
    tokenizer = init_tokenizer(args.tokenizer_path)

    if os.path.isfile(args.image_path):
        image_paths = [args.image_path]
    else:
        image_paths = [os.path.join(args.image_path, fname) for fname in os.listdir(args.image_path)]

    preprocess_time = 0.0
    for filename in image_paths:
        # Preprocess
        logging.info("Filename: {}".format(filename))
        image = cv2.imread(filename)

        start_time = time.time()
        image_input = preprocess(image, image_size)
        preprocess_time += time.time() - start_time
        # predict
        outputs = model.predict(image_input)
        captions = []
        for output in outputs.astype('int64'):
            caption = tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[13:])
        logging.info(f"Caption : {captions[0]}")


    image_num = len(image_paths)
    logging.info(("-------------------Image num {}, Preprocess average time ------------------------").format(image_num))
    logging.info("preprocess(ms): {:.2f}".format(preprocess_time / image_num * 1000))

    logging.info(("------------------ Image num {}, Predict average time ----------------------").format(image_num))
    logging.info("predict(ms)   : {:.2f}".format(model.predict_time / image_num * 1000))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--image_path', type=str, default='./datasets/test', help='path of input')
    parser.add_argument('--tokenizer_path', type=str, default="./models/bert-base-uncased", help="path of tokenizer")
    parser.add_argument('--bmodel_path', type=str, default='./models/BM1684X/blip_cap_bm1684x_f32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
