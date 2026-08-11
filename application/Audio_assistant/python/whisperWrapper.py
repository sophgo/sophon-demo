import argparse
import os
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import torch
import tqdm
import time

from bmwhisper.decoding import DecodingOptions, DecodingResult
from bmwhisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from bmwhisper.utils import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    exact_div,
    format_timestamp,
    get_writer,
    log_mel_spectrogram,
    make_safe,
    optional_float,
    optional_int,
    pad_or_trim,
    str2bool,
)
# 复用 sample/Whisper 的 sail 版 transcribe（不再自实现，不再依赖 libuntpu）
from bmwhisper.transcribe import transcribe

if TYPE_CHECKING:
    from bmwhisper.model import Whisper


class WhisperWrapper():
    def __init__(self, args):
        self.args = args.__dict__
        self.model = None
        self.load_model()

    def load_model(self):
        args = self.args
        args["model_name"] = args.pop("model")
        output_dir: str = args.pop("output_dir")
        output_format: str = args.pop("output_format")
        os.makedirs(output_dir, exist_ok=True)

        model_name = args["model_name"]
        if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
            if args["language"] is not None:
                warnings.warn(
                    f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead."
                )
            args["language"] = "en"
        temperature = args.pop("temperature")
        if (increment := args.pop("temperature_increment_on_fallback")) is not None:
            temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
        else:
            temperature = [temperature]

        self.temperature = temperature

        if (threads := args.pop("threads")) > 0:
            torch.set_num_threads(threads)

        # sample/Whisper 的 sail 版 load_model 读 args["dev_id"]（set_argparser 用 devid）
        args["dev_id"] = args.pop("devid")

        from bmwhisper import load_model

        self.model = load_model(args)
        # 这些 key 不能流进 transcribe -> DecodingOptions（sample 的 transcribe 不 pop dev_id）
        pop_list = ["model_name", "model_dir", "bmodel_dir", "dev_id",
                    "chip_mode", "chip", "profile"]
        for arg in pop_list:
            args.pop(arg, None)

        self.writer = get_writer(output_format, output_dir)
        word_options = ["highlight_words", "max_line_count", "max_line_width"]
        if not args["word_timestamps"]:
            for option in word_options:
                if args[option]:
                    raise ValueError("requires --word_timestamps True")
        if args["max_line_count"] and not args["max_line_width"]:
            warnings.warn("--max_line_count has no effect without --max_line_width")
        self.writer_args = {arg: args.pop(arg) for arg in word_options}
        self.loop_profile = self.args.pop("loop_profile")


    def transcribe(self, audio_path):
        os.environ["LOG_LEVEL"] = "-1"
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.model.init_cnt()
        self.model.init_time()
        print()
        print("{:=^100}".format(f" Start "))
        print(f"### audio_path: {os.path.basename(audio_path)}")
        audio_start_time = time.time()
        result = transcribe(self.model, audio_path, temperature=self.temperature, **self.args)
        cpu_time = time.time() - audio_start_time - self.model.inference_time
        if self.loop_profile:
            self.model.print_cnt()
        self.model.init_time()

        return result

os.environ["LOG_LEVEL"] = "-1"


def set_argparser():
    from bmwhisper import available_models
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action='store_true', help="print profiling result")
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--bmodel_dir", type=str, default="../models/whisper", help="the path to save the combined whisper bmodel; uses ../models/whisper by default")
    # parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["txt", "vtt", "srt", "tsv", "json", "all"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--word_timestamps", type=str2bool, default=False, help="(experimental) extract word-level timestamps and refine the results based on them")
    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'.。,，!！?？:：”)]}、", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of lines in a segment")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--padding_size", type=optional_int, default=448, help="max pre-allocation size for the key-value cache (must match the combined bmodel)")
    parser.add_argument("--chip_mode", default="soc", choices=["pcie", "soc"], help="name of the Whisper model to use")
    parser.add_argument("--loop_profile", action="store_true", help="whether to print loop times")
    parser.add_argument("--chip", default="bm1688", choices=["1684x", "bm1688"], help="chip platform name")
    parser.add_argument('-d', "--devid", default=0, type=int, help="device id")

    return parser
