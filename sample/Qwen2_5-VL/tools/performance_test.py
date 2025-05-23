import sys
import os 
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../python")
import json
import torch
from qwen2_5_vl import Qwen2_5_VL
import copy
import time
import argparse


def main(args):
    model = Qwen2_5_VL(dev_id=args.dev_id, bmodel_path=args.bmodel_path, log_level=args.log_level, processor_path=args.processor_path, tokenizer_path=args.tokenizer_path, config=args.config)
    vision_inputs = [{"type":"image_url", "image_url":{}, \
                                    "max_side":-1}]
    text = "请描述图片中的内容"
    file_paths = [os.path.join(args.vision_inputs, file_name) for file_name in os.listdir(args.vision_inputs)]
    message_duration_mean = 0
    preprocess_duration_mean = 0
    vision_duration_mean = 0
    first_duration_mean = 0
    tps_mean = 0
    
    for file_path in file_paths:
        message_start_time = time.time()
        first_start = time.time()
        cur_data = copy.deepcopy(vision_inputs)
        cur_data.append({"type": "text", "text": text})
        cur_data[0]["image_url"]["url"] = file_path
        messages, _ = model.generate_message([], cur_data, "user")
        messages = [messages]
        message_duration = time.time() - message_start_time
        message_duration_mean += message_duration

        # preprocess text and images/video, get model inputs
        preprocess_start_time = time.time()
        inputs = model.preprocess(messages=messages)
        token_len = inputs.input_ids.numel()
        preprocess_duration = time.time() - preprocess_start_time
        preprocess_duration_mean += preprocess_duration

        # vision
        vision_duration = -1
        vision_start = time.time()
        model.forward_embed(inputs.input_ids.numpy())
        vit_token_list = torch.where(inputs.input_ids == model.ID_IMAGE_PAD)[1].tolist()
        vit_offset = vit_token_list[0]
        model.vit_process_image(inputs, vit_offset)
        position_ids = model.get_rope_index(inputs.input_ids, inputs.image_grid_thw,
                                            model.ID_IMAGE_PAD)
        max_posid = int(position_ids.max())
        position_ids = position_ids.numpy()
        vision_duration = time.time() - vision_start
        vision_duration_mean += vision_duration
        print(f"vision inference cost time(s): {time.time() - vision_start}")

        # Chat
        print("\nAnswer: ", end = '')
        token = model.forward_first_new(position_ids)
        first_end = time.time()
        tok_num = 0
        # Following tokens
        full_word_tokens = []
        text = ""
        while not (model.is_end_with_reason(token)[0] or model.is_end_with_reason(token)[1]):
            full_word_tokens.append(token)
            word = model.tokenizer.decode(full_word_tokens,
                                        skip_special_tokens=True)
            if "�" not in word:
                if len(full_word_tokens) == 1:
                    pre_word = word
                    word = model.tokenizer.decode(
                        [token, token],
                        skip_special_tokens=True)[len(pre_word):]
                text += word
                print(word, flush=True, end="")
                full_word_tokens = []
            max_posid += 1

            token = model.forward_next_new(max_posid)
            tok_num += 1
        next_end = time.time()
        first_duration = first_end - first_start
        first_duration_mean += first_duration
        next_duration = next_end - first_end
        tps = tok_num / next_duration
        tps_mean += tps
        print(f"\ngenerate message: {message_duration:.3f} s")
        print(f"preprocess: {preprocess_duration:.3f} s")
        print(f"vision: {vision_duration:.3f} s")
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

    message_duration_mean /= len(file_paths)
    preprocess_duration_mean /= len(file_paths)
    vision_duration_mean /= len(file_paths)
    first_duration_mean /= len(file_paths)
    tps_mean /= len(file_paths)
    print(f"\ngenerate message mean({len(file_paths)}): {message_duration_mean:.3f} s")
    print(f"preprocess mean({len(file_paths)}): {preprocess_duration_mean:.3f} s")
    print(f"vision mean({len(file_paths)}): {vision_duration_mean:.3f} s")
    print(f"FTL mean({len(file_paths)}): {first_duration_mean:.3f} s")
    print(f"TPS mean({len(file_paths)}): {tps_mean:.3f} token/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--bmodel_path',
                        type=str,
                        default="../models/BM1684X/qwen2.5-vl-3b-instruct-awq_w4bf16_seq2048_bm1684x_1dev_20250428_143625.bmodel",
                        help='path to the bmodel file')
    parser.add_argument('-t',
                        '--tokenizer_path',
                        type=str,
                        default="./configs/token_config",
                        help='path to the tokenizer file')
    parser.add_argument('-p',
                        '--processor_path',
                        type=str,
                        default="./configs/processor_config",
                        help='path to the processor file')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default="./configs/config.json",
                        help='path to the model config file')
    parser.add_argument('-d', '--dev_id', type=int,
                        default=0, help='device ID to use')
    parser.add_argument('-vi',
                        '--vision_inputs',
                        type=str,
                        default="./datasets/images/test_frames",
                        help='path to the video or images and preprocess params, json format') 
    parser.add_argument('-ll',
                        '--log_level',
                        type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO",
                        help='log level, default: INFO, option[DEBUG, INFO, WARNING, ERROR]')
    args = parser.parse_args()
    main(args)