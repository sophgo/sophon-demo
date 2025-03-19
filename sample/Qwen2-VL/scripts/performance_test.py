import sys
import os 
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../python")
import json
from qwen2_vl import Qwen2VL
import copy
import time
import argparse


def main(args):
    model = Qwen2VL(dev_id=args.dev_id, bmodel_path=args.bmodel_path, log_level=args.log_level, processor_path=args.processor_path, tokenizer_path=args.tokenizer_path, config=args.config)
    vision_inputs = [{"type":"image_url", "image_url":{}, \
                                    "max_side":-1}]
    question = "请描述图片中的内容"
    file_paths = [os.path.join(args.vision_inputs, file_name) for file_name in os.listdir(args.vision_inputs)]
    message_duration_mean = 0
    preprocess_duration_mean = 0
    vision_duration_mean = 0
    first_duration_mean = 0
    tps_mean = 0
    
    for file_path in file_paths:
        message_start_time = time.time()
        cur_data = copy.deepcopy(vision_inputs)
        cur_data.append({"type": "text", "text": question})
        cur_data[0]["image_url"]["url"] = file_path
        messages = model.generate_message([], cur_data, "user")
        messages = [messages]
        message_duration = time.time() - message_start_time
        message_duration_mean += message_duration

        # preprocess text and images/video, get model inputs
        preprocess_start_time = time.time()
        position_ids, inputs, image_offset = model.preprocess(messages=messages, \
                        video_grid_thw=None, image_grid_thw=None)
        image_grid_thw = inputs.image_grid_thw if "image_grid_thw" in inputs else None
        video_grid_thw = inputs.video_grid_thw if "video_grid_thw" in inputs else None
        pixel_values_images = inputs.pixel_values if "pixel_values" in inputs else None
        pixel_values_videos = inputs.pixel_values_videos if "pixel_values_videos" in inputs else None
        preprocess_duration = time.time() - preprocess_start_time
        preprocess_duration_mean += preprocess_duration

        # vision
        vision_duration = -1
        vision_start = time.time()
        image_embeds, video_embeds = model.vision_process(
            pixel_values_images=pixel_values_images.numpy() if pixel_values_images is not None else None,
            pixel_values_videos=pixel_values_videos.numpy() if pixel_values_videos is not None else None, 
            image_grid_thw=image_grid_thw.numpy() if image_grid_thw is not None else None,
            video_grid_thw=video_grid_thw.numpy() if video_grid_thw is not None else None)
        vision_duration = time.time() - vision_start
        vision_duration_mean += vision_duration
        print(f"vision inference cost time(s): {time.time() - vision_start}")

        # Chat
        print("\nAnswer: ", end = '')
        first_start = time.time()
        token = model.forward_first(inputs.input_ids.numpy(), position_ids.numpy(), image_embeds, video_embeds)
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
            token = model.forward_next()
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
                        default="../models/BM1684X/qwen2-vl-7b_int4_seq1536_1dev.bmodel",
                        help='path to the bmodel file')
    parser.add_argument('-t',
                        '--tokenizer_path',
                        type=str,
                        default="../python/configs/token_config",
                        help='path to the tokenizer file')
    parser.add_argument('-p',
                        '--processor_path',
                        type=str,
                        default="../python/configs/processor_config",
                        help='path to the processor file')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default="../python/configs/config.json",
                        help='path to the model config file')
    parser.add_argument('-d', '--dev_id', type=int,
                        default=0, help='device ID to use')
    parser.add_argument('-g',
                        '--generation_mode',
                        type=str,
                        choices=["greedy", "penalty_sample"],
                        default="greedy",
                        help='mode for generating next token')
    parser.add_argument('-vi',
                        '--vision_inputs',
                        type=str,
                        default="../datasets/images/test_frames",
                        help='path to the video or images and preprocess params, json format') 
    parser.add_argument('-ll',
                        '--log_level',
                        type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO",
                        help='log level, default: INFO, option[DEBUG, INFO, WARNING, ERROR]')
    args = parser.parse_args()
    main(args)
