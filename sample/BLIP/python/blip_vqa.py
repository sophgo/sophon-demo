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
import sys
import os
import time
import argparse

from blip import blip_vqa, init_tokenizer, preprocess

def main(args):
    # Initial
    text = args.text
    model = blip_vqa(args)
    image_size = 480

    if not os.path.isfile(args.image_path):
        print("Please select one image file")
        return -1

    # Preprocess
    image = cv2.imread(args.image_path)
    start_time = time.time()
    image_input = preprocess(image, image_size)
    preprocess_time = time.time() - start_time
    # image encoding
    image_embeds = model.image_process(image_input)
    # Tokenize
    tokenizer = init_tokenizer(args.tokenizer_path)
    predict_time = 0
    num = 0
    text = input("Enter your question: ")
    while text != "exit":
        text_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=35,return_tensors="np")
        # predict
        start_time = time.time()
        outputs = model.predict(image_embeds, text_inputs)
        predict_time += time.time() - start_time
        num += 1
        answer = tokenizer.decode(outputs[0].astype('int64'), skip_special_tokens=True)
        print("Answer : "+answer)
        text = input("Enter your question: ")
    print("---------------------------------- Preprocess average time ------------------------")
    print("preprocess(ms): "+format(preprocess_time/num*1000,'.2f'))
    print("INFO - ------------------ Question num "+str(num)+", Predict average time ----------------------")
    print("predict(ms)   : "+format(predict_time/num*1000,'.2f'))



def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--image_path', type=str, default='./datasets/test/demo.jpg', help='file path of image')
    parser.add_argument('--text', nargs='+', default=['where is the women sitting'], help='text of input')
    parser.add_argument('--tokenizer_path', type=str, default="./models/bert-base-uncased", help="path of tokenizer")
    parser.add_argument('--venc_bmodel_path', type=str, default='./models/BM1684X/blip_vqa_venc_bm1684x_f32_1b.bmodel', help='path of image encoding bmodel')
    parser.add_argument('--tenc_bmodel_path', type=str, default='./models/BM1684X/blip_vqa_tenc_bm1684x_f32_1b.bmodel', help='path of text encoding bmodel')
    parser.add_argument('--tdec_bmodel_path', type=str, default='./models/BM1684X/blip_vqa_tdec_bm1684x_f32_1b.bmodel', help='path of text decoding bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
