try:
    from Llama3 import chat
except:
    import logging
    logging.warning("Llama3 not support! If need, please refer to python/README.md.")
import argparse
import time
from rich.console import Console
import wave
from transformers import AutoTokenizer
import time
import subprocess
import pyaudio
import collections
import webrtcvad
import soundfile as sf
import wave
from funasr import AutoModel
import numpy as np
import os
import multiprocessing

# vits
import math
from text import cleaned_text_to_sequence, pinyin_dict
import logging
from pypinyin import lazy_pinyin, Style
from pypinyin.core import load_phrases_dict
logging.basicConfig(level=logging.INFO)
from datetime import datetime

import whisperWrapper
import sys
sys.path.append("./XTTS")
from XTTS.api import TTS

# vad
VAD_FRAME_DURATION_MS = 30
PADDING_DURATION_MS = 300
num_padding_frames = int(PADDING_DURATION_MS / VAD_FRAME_DURATION_MS)
ring_buffer = collections.deque(maxlen=num_padding_frames)
is_voice_buffer = collections.deque(maxlen=num_padding_frames)
num_voiced = 0
num_unvoiced = 0
triggered = False
voiced_frames = []
vad = webrtcvad.Vad(2)

device = "cpu"


def isEmoji(content):
    if not content:
        return False
    if u"\U0001F600" <= content and content <= u"\U0001F64F":
        return True
    elif u"\U0001F300" <= content and content <= u"\U0001F5FF":
        return True
    elif u"\U0001F680" <= content and content <= u"\U0001F6FF":
        return True
    elif u"\U0001F1E0" <= content and content <= u"\U0001F1FF":
        return True
    else:
        return False

def get_wav_duration(filename):
    with wave.open(filename, 'r') as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        duration = frames / float(rate)
        return duration

class Timer:
    def __init__(self):
        self.run_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.run_time = time.time() - self.start_time


class Llama3Model:
    def __init__(self, llm_model_path=None, tokenizer_path=None, 
                 devid='0', temperature_llm=1.0, 
                 top_p=1.0, repeat_penalty=1.0, repeat_last_n=32, 
                 max_new_tokens=1024, generation_mode="greedy", 
                 prompt_mode="prompted", decode_mode="basic",
                 enable_history=True):
    
        self.args = argparse.Namespace(
            devices=[int(d) for d in devid.split(",")], # devid
            temperature=temperature_llm,
            top_p=top_p,
            model_path=llm_model_path,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            max_new_tokens=max_new_tokens,
            generation_mode=generation_mode,
            prompt_mode=prompt_mode,
            decode_mode=decode_mode,
            enable_history=enable_history
        )

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = '你是一个有用的人工智能助手。'
        self.EOS = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        self.system = {"role":"system","content":self.system_prompt}
        self.history = [self.system]
        self.enable_history = self.args.enable_history

        self._load_model()

    def _load_model(self):
        if self.args.decode_mode == "basic":
            self.model = chat.Llama3()
            self.model.init(self.args.devices, self.args.model_path)
            self.model.temperature = self.args.temperature
            self.model.top_p = self.args.top_p
            self.model.repeat_penalty = self.args.repeat_penalty
            self.model.repeat_last_n = self.args.repeat_last_n
            self.model.max_new_tokens = self.args.max_new_tokens
            self.model.generation_mode = self.args.generation_mode
            self.model.prompt_mode = self.args.prompt_mode
        else:
            raise ValueError("decode mode: {} is illegal!".format(self.args.decode_mode))
        
        self.SEQLEN = self.model.SEQLEN
    
    def gen_response(self, text:str, queue):
        """
        Chat with the model
        """
        self.input_str = text
        tokens = self._encode_tokens()
        print("\nAnswer:", end=" ")
        return self._stream_answer(tokens, queue)

    def _encode_tokens(self):
        self.history.append({"role":"user","content":self.input_str})
        return self.tokenizer.apply_chat_template(self.history, 
                    tokenize=True, add_generation_prompt=True)

    def _stream_answer(self, tokens, queue):
        """
        Stream the answer for the given tokens.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        # First token
        timer = Timer()
        # with timer:
        token = self.model.forward_first(tokens)
        # print(f"generation of the first token took {timer.run_time:.3f} seconds!")

        full_word_tokens = []
        answer = ""
        full_text_answer = ""
        sentence = []
        # Following tokens
        while token not in self.EOS and self.model.token_length < self.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next()
                tok_num += 1
                continue

            self.answer_token += [token]
            print(word, flush=True, end="")
            if args_global.streaming_output:
                if not isEmoji(word):
                    sentence.append(word.strip())
                    if "。" in word or "." in word or "?" in word or "？" in word or "!" in word or "！" in word or ":" in word or "：" in word:
                        queue.put("".join(sentence))
                        sentence = []
            answer += word
            if not isEmoji(word):
                full_text_answer += word.strip()

            token = self.model.forward_next()
            tok_num += 1
            full_word_tokens = []
        
        if not args_global.streaming_output:
            queue.put(full_text_answer)
        elif len(sentence) > 0:
            queue.put("".join(sentence))
        return answer


class MinicpmModel:
    # TODO: +prompt tokens
    def __init__(self, minicpm_model_path, minicpm_tokenizer_model_path, devid=0):
        self.process = subprocess.Popen(
            ['./MiniCPM/demo/minicpm', '--model', minicpm_model_path, '--tokenizer', minicpm_tokenizer_model_path, '--devid', devid],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    def gen_response_line(self, input:str):

        self.process.stdin.write(input + '\n')
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline().strip()
            if line[:6] == 'Answer':
                return line[9:]
            
    def gen_response(self, input:str, queue):
        # TODO: +prompt tokens
        st = time.time()
        self.process.stdin.write(input + '\n')
        self.process.stdin.flush()

        lines = []
        answer_flag = False
        while True:
            line=self.process.stdout.readline().strip()
            if line[:6] == 'Answer':
                answer_flag=True
                print("\n")
            if answer_flag:
                if line[:20]=='first token latency:':
                    if args_global.profile:
                        print(line)
                        line=self.process.stdout.readline().strip()
                        print(line)
                    if not args_global.streaming_output:
                        queue.put(''.join(lines))
                    return lines
                lines.append(line)
                print(line)
                if args_global.streaming_output:
                    print("llm output sentence cost time(s): ", time.time() - st)
                    queue.put(line)

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class VITS:
    def __init__(
        self,
        args,
    ):
        import sophon.sail as sail
        self.net = sail.Engine(args.vits_model, args.devid, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.vits_model))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_names[0])
        self.max_length = self.input_shape[1]

        self.tts_front = VITS_PinYin(args.bert_model, args.devid, hasBert=True)
        self.inference_time = 0.0
        self.sample_rate = 16000
        self.stage_factor = 900.0


    def init(self):
        self.inference_time = 0.0


    def estimate_silence_threshold(self, audio, sample_rate, duration=0.1):
        """
        Estimate the threshold of silence in an audio signal by calculating
        the average energy of the first 'duration' seconds of the audio.

        Args:
            audio: numpy array of audio data.
            sample_rate: the sample rate of the audio data.
            duration: duration (in seconds) of the initial audio to consider for silence.

        Returns:
            The estimated silence threshold.
        """
        # Calculate the number of samples to consider
        num_samples = int(sample_rate * duration)

        # Calculate the energy of the initial segment of the audio
        initial_energy = np.mean(np.abs(audio[-num_samples:]))

        # Return the average energy as the threshold
        return initial_energy


    def remove_silence_from_end(self, audio, sample_rate, threshold=0.005, frame_length=512):
        """
        Removes silence from the end of an audio signal using a specified energy threshold.
        If no threshold is provided, it estimates one based on the initial part of the audio.

        Args:
            audio: numpy array of audio data.
            sample_rate: the sample rate of the audio data.
            threshold: amplitude threshold to consider as silence. If None, will be estimated.
            frame_length: number of samples to consider in each frame.

        Returns:
            The audio signal with end silence removed.
        """
        if threshold is None:
            threshold = self.estimate_silence_threshold(audio, sample_rate)

        # Calculate the energy of audio by frame
        energies = [np.mean(np.abs(audio[i:i+frame_length])) for i in range(0, len(audio), frame_length)]

        # Find the last frame with energy above the threshold
        for i, energy in enumerate(reversed(energies)):
            if energy > threshold:
                last_non_silent_frame = len(energies) - i - 1
                break
        else:
            # In case the whole audio is below the threshold
            return np.array([])

        # Calculate the end index of the last non-silent frame
        end_index = (last_non_silent_frame + 1) * frame_length

        # Return the trimmed audio
        return audio[:end_index]


    def split_text_near_punctuation(self, text, max_length):
        # Define punctuation marks where the text can be split
        punctuation = "。！？，、；：,."
        # Initialize a list to hold the split text segments
        split_texts = []

        # Continue splitting the text until the remaining text is shorter than max_length
        while len(text) > max_length:
            # Assume we need to split at the max_length, then search left for the nearest punctuation
            split_pos = max_length
            # Search left for the nearest punctuation
            while split_pos > 0 and text[split_pos] not in punctuation:
                split_pos -= 1

            # If no punctuation is found to the left, split at the original max_length
            if split_pos == 0:
                split_pos = max_length

            # Split the text and add to the list
            split_texts.append(text[:split_pos + 1])
            # Update the remaining text
            text = text[split_pos + 1:].lstrip()

        # Add the remaining text segment
        split_texts.append(text)
        return split_texts

    def preprocess(self, x: np.ndarray):
        x = np.expand_dims(x, axis=0) if x.ndim == 1 else x
        if x.shape[1] < self.max_length:
            padding_size = self.max_length - x.shape[1]
            x = np.pad(x, [(0, 0), (0, padding_size)], mode='constant', constant_values=0)

        return x


    def inference(self, x: np.ndarray, char_embeds: np.ndarray):
        # Initialize an empty list to collect output tensors
        outputs = []

        # Extract a sequence of length `self.max_length` from x
        start_time = time.time()
        input_data = {self.input_names[0]: x, self.input_names[1]: char_embeds}
        output_data = self.net.process(self.graph_name, input_data)
        self.inference_time += time.time() - start_time
        y_max, y_segment = output_data.values()

        y_segment = y_segment[:math.ceil(y_max[0] / self.stage_factor * len(y_segment) + 1)]
        y_segment = self.remove_silence_from_end(y_segment, self.sample_rate)

        # Collect the output
        outputs.append(y_segment)

        # Concatenate all output segments along the sequence dimension
        y = np.concatenate(outputs, axis=-1)
        return y

    def __call__(self, x: np.ndarray, char_embeds: np.ndarray):
        """
        Args:
          x:
            A int32 array of shape (1, 128)
        """

        x = self.preprocess(x)
        y = self.inference(x, char_embeds)
        return y



def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def clean_chinese(text: str):
    text = text.strip()
    text_clean = []
    for char in text:
        if (is_chinese(char)):
            text_clean.append(char)
        else:
            if len(text_clean) > 1 and is_chinese(text_clean[-1]):
                text_clean.append(',')
    text_clean = ''.join(text_clean).strip(',')
    return text_clean


def load_pinyin_dict():
    my_dict={}
    with open("./text/pinyin-local.txt", "r", encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            cuts = line.strip().split()
            hanzi = cuts[0]
            phone = cuts[1:]
            tmp = []
            for p in phone:
                tmp.append([p])
            my_dict[hanzi] = tmp
    load_phrases_dict(my_dict)


class VITS_PinYin:
    def __init__(self, bert_model, dev_id, hasBert=True):
        load_pinyin_dict()
        from bert import TTSProsody
        self.hasBert = hasBert
        if self.hasBert:
            self.prosody = TTSProsody(bert_model, dev_id)
        self.inference_time = 0.0

    def get_phoneme4pinyin(self, pinyins):
        result = []
        count_phone = []
        for pinyin in pinyins:
            if pinyin[:-1] in pinyin_dict:
                tone = pinyin[-1]
                a = pinyin[:-1]
                a1, a2 = pinyin_dict[a]
                result += [a1, a2 + tone]
                count_phone.append(2)
        return result, count_phone

    def chinese_to_phonemes(self, text):
        text = clean_chinese(text)
        phonemes = ["sil"]
        chars = ['[PAD]']
        count_phone = []
        count_phone.append(1)
        for subtext in text.split(","):
            if (len(subtext) == 0):
                continue
            pinyins = self.correct_pinyin_tone3(subtext)
            sub_p, sub_c = self.get_phoneme4pinyin(pinyins)
            phonemes.extend(sub_p)
            phonemes.append("sp")
            count_phone.extend(sub_c)
            count_phone.append(1)
            chars.append(subtext)
            chars.append(',')
        phonemes.append("sil")
        count_phone.append(1)
        chars.append('[PAD]')
        chars = "".join(chars)
        char_embeds = None

        if self.hasBert:
            start_time = time.time()
            char_embeds = self.prosody.get_char_embeds(chars)
            self.inference_time +=  time.time() - start_time
            char_embeds = self.prosody.expand_for_phone(char_embeds, count_phone)
        return " ".join(phonemes), char_embeds

    def correct_pinyin_tone3(self, text):
        pinyin_list = lazy_pinyin(text,
                                  style=Style.TONE3,
                                  strict=False,
                                  neutral_tone_with_five=True,
                                  tone_sandhi=True)
        # , tone_sandhi=True -> 33变调
        return pinyin_list
 
def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n
 
 
def vad_collector(sample_rate, frame):
	global num_voiced, num_unvoiced, ring_buffer, is_voice_buffer, triggered, voiced_frames, vad, VAD_FRAME_DURATION_MS
	# sys.stdout.write(
	#     '1' if vad.is_speech(frame.bytes, sample_rate) else '0')
	if not triggered:
		if len(is_voice_buffer) == is_voice_buffer.maxlen and is_voice_buffer[0]:
			num_voiced -= 1
		ring_buffer.append(frame)
		is_voice_buffer.append(vad.is_speech(frame.bytes, sample_rate))
		# num_voiced = len([f for f in ring_buffer
		#                   if vad.is_speech(f.bytes, sample_rate)])
		if is_voice_buffer[-1]:
			num_voiced += 1
		if num_voiced > 0.9 * ring_buffer.maxlen:
			triggered = True
			voiced_frames.extend(ring_buffer)
			ring_buffer.clear()
			is_voice_buffer.clear()
			num_voiced = 0
	else:
		if len(is_voice_buffer) == is_voice_buffer.maxlen and not is_voice_buffer[0]:
			num_unvoiced -= 1
		voiced_frames.append(frame)
		ring_buffer.append(frame)
		is_voice_buffer.append(vad.is_speech(frame.bytes, sample_rate))
		# num_unvoiced = len([f for f in ring_buffer
		#                     if not vad.is_speech(f.bytes, sample_rate)])
		if not is_voice_buffer[-1]:
			num_unvoiced += 1
		if num_unvoiced > 0.9 * ring_buffer.maxlen:
			triggered = False
			res = b''.join([f.bytes for f in voiced_frames])
			voiced_frames = []
			num_unvoiced = 0
			ring_buffer.clear()
			is_voice_buffer.clear()
			return res
	return None


def run_tts(queue, play_queue, tts_model_config):
    output_text = ''
    min_output_text_len = args_global.min_tts_input_len
    if args_global.tts_type == "xtts":
        tts_model = TTS(model_path=tts_model_config["other_models_dir"], 
                    config_path=tts_model_config["config_pth"],
                    tpu_inference_config=tts_model_config).to("cpu")
    elif args_global.tts_type == "vits":
        tts_model = VITS(tts_model_config)
        tts_model.init()
    print("start T2S model ...")
    full_out_audio_data = []

    while True:
        try:
            llm_response = queue.get()
            with timer:
                whole_resp_text = ''
                if llm_response is not None:
                    if len(llm_response.strip()) == 0:
                        continue
                    elif 'Answer:' in llm_response:
                        whole_resp_text = llm_response[9:].strip()
                    else:
                        whole_resp_text = llm_response.strip()
                output_text += whole_resp_text
                if (len(output_text) < min_output_text_len and llm_response is not None) or len(output_text) == 0:
                    if llm_response is None:
                        if not args_global.streaming_output and len(full_out_audio_data) > 0:
                            play_queue.put(np.concatenate(full_out_audio_data, axis=-1))
                            full_out_audio_data = []
                        play_queue.put(None)
                    continue
                if llm_response is None:
                    is_final = True
                else:
                    is_final = False
                whole_resp_text = output_text
                print('\n------audio out content: ', whole_resp_text)
                if args_global.tts_type == "vits":
                    split_items = tts_model.split_text_near_punctuation(whole_resp_text, int(tts_model.max_length / 2 - 5))
                    for split_item in split_items:
                        print('\n------audio seg: ', split_item)
                        if len(split_item) == 0:
                            continue
                        phonemes, char_embeds = tts_model.tts_front.chinese_to_phonemes(split_item)
                        input_ids = cleaned_text_to_sequence(phonemes)
                        char_embeds = np.expand_dims(char_embeds, 0)
                        x = np.array(input_ids, dtype=np.int32)
                        if args_global.streaming_output:
                            play_queue.put(tts_model(x, char_embeds))
                        else:
                            full_out_audio_data.append(tts_model(x, char_embeds))
                elif args_global.tts_type == "xtts":
                    wav = tts_model.tts(text=whole_resp_text, speaker=args_global.tts_speaker, language=args_global.tts_out_language)
                    if args_global.streaming_output:
                        play_queue.put(wav)
                    else:
                        full_out_audio_data.append(np.stack(wav))

                output_text = ''
                if is_final:
                    if not args_global.streaming_output and len(full_out_audio_data) > 0:
                        play_queue.put(np.concatenate(full_out_audio_data, axis=-1))
                        full_out_audio_data = []
                    play_queue.put(None)
            tts_time = timer.run_time
            if args_global.profile:
                console.print(f"\ntts took {tts_time:.2f} seconds.") 
        except Exception as e:
            print(e)

class Pipeline():
    "pipeline for asr-->llm-->tts. "
    def __init__(self, asr_model=None, llm_model=None):
        self.asr_model = asr_model
        self.llm_model = llm_model
        self.chunk_ms = 200 # 1000 # ms
        self.model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", device='cpu', disable_pbar=True, log=False)
        self.cache = {}
        self.use_fsmn_vad = True
        self.speech_start = False
        self.audio_fs = 16000
        self.question_prefix = "，请用中文回答。"
        self.latency_start_time = None
    
    def run_asr(self, audio_path:str):
        return self.asr_model.transcribe(audio_path) 
    
    def run_llm(self, text:str, queue):
        return self.llm_model.gen_response(text, queue)
    
    def rum_tts(self, text:str):
        pass

    def load_bytes(self, input):
        middle_data = np.frombuffer(input, dtype=np.int16)
        middle_data = np.asarray(middle_data)
        if middle_data.dtype.kind not in "iu":
            raise TypeError("'middle_data' must be an array of integers")
        dtype = np.dtype("float32")
        if dtype.kind != "f":
            raise TypeError("'dtype' must be a floating point type")

        i = np.iinfo(middle_data.dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
        return array

    def inference(self, clip_audio, params, queue):
        # sf.write("whisper_input.wav", clip_audio, fs)
        try:
            st = time.time()
            with wave.open('whisper_input.wav','wb') as wavWrite:
                wavWrite.setparams(params) 
                wavWrite.writeframes(clip_audio) 
            et = time.time()
            print("write audio cost: ", et-st)

            with timer:
                asr_prompt = self.run_asr("whisper_input.wav")
            asr_time = timer.run_time
            print("\nasr cost: ", asr_time)

            if len(asr_prompt["text"].strip()) == 0:
                print("asr rec empty!")
                return

            # console.print(f"[green]ASR took {timer.run_time:.3f} seconds to process a {get_wav_duration(input_audio_path)} seconds audio!")
            # console.print(f"[blue]Input is:"+asr_prompt["text"])
            
            with timer:
                question = asr_prompt["text"] + self.question_prefix
                print("Question:"+question)
                llm_response = self.run_llm(question, queue)
            # print('Answer:'+llm_response)
            # console.print(f"\n[green]LLM took {timer.run_time:.4f} seconds to answer this question on BM1688!")
            llm_time = timer.run_time
            print("\nllm cost: ", llm_time, flush=True)

            if args_global.profile:
                console.print(f"\nwhisper took {asr_time:.2f} seconds, and LLM took {llm_time:.2f} seconds.") 
        except Exception as e:
            print(e)

    def play(self, queue, fs):
        if not args_global.output_file:
            p = pyaudio.PyAudio()
            out_stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=fs,
                output=True,
                output_device_index=args_global.audio_devid) 
        else:
            out_stream = None

        while True:
            try:
                audio_data = queue.get()
                if audio_data is None:
                    continue
                if self.latency_start_time.value != -1:
                    latency = time.time() - self.latency_start_time.value
                    print('latency time(s): ', latency)
                    self.latency_start_time.value = -1
                if args_global.output_file:
                    now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace(' ', '_').replace('/', '_').replace(':', '_')
                    audio_path = f"{now_time}.wav"
                    sf.write(
                        audio_path, audio_data, fs)
                    print("save audio to {}".format(audio_path))
                else:
                    data_int16 = audio_data * 32767
                    data_int16 = data_int16.astype('int16')
                    data_bytes = data_int16.tobytes()
                    out_stream.write(data_bytes)
            except Exception as e:
                print(e)
        if out_stream is not None:
            out_stream.stop_stream()
            out_stream.close()
    
    def forward(self, queue, audio_seq_queue, microphone_devid:int=None, output_audio_path:str=None, speaker_wav:str=None):
        global voiced_frames
        timer = Timer()

        self.latency_start_time = multiprocessing.Value('d', -1)
        play_process = multiprocessing.Process(target=self.play, args=(audio_seq_queue, self.audio_fs))
        play_process.daemon = True
        play_process.start()

        if args_global.audio_in is not None:
            with wave.open(args_global.audio_in, "rb") as wav_file:
                params = wav_file.getparams()
                fs = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                audio_bytes = bytes(frames)  
                stride = int(fs * (self.chunk_ms / 1000.0) * 2)     
        else:
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            # chunk_size = 60 * args.chunk_size[1] / args.chunk_interval
            fs = self.audio_fs
            CHUNK = int(fs * (self.chunk_ms / 1000.0)) # int(args.chunk_size[1] / args.chunk_interval * 2) # int(RATE / 1000 * chunk_size)

            p = pyaudio.PyAudio()

            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=fs,
                            input=True,
                            frames_per_buffer=CHUNK,
                            input_device_index=microphone_devid)
            params = wave._wave_params(nchannels=1, sampwidth=2, framerate=16000, nframes=160000, comptype='NONE', compname='not compressed')
            print("microphone running ...")
        
        beg = 0
        is_final = False
        while True:
            if args_global.audio_in is not None:
                if beg > len(audio_bytes):
                     break
                data = audio_bytes[beg : beg + stride]
                beg += stride
                if beg > len(audio_bytes):
                    is_final = True
            else:
                data = stream.read(CHUNK, exception_on_overflow=False)
            message  = data
            num_questions = 0
            if self.use_fsmn_vad:
                message_arr = self.load_bytes(message)
                res = self.model.generate(input=message_arr, cache=self.cache, is_final=is_final, chunk_size=self.chunk_ms)
                # cases: [beg, -1], [], [-1, end], [beg, end]
                if len(res[0]["value"]):
                    for seg in res[0]["value"]:
                        if seg[0] != -1 and seg[1] == -1:
                            self.speech_start = True
                            voiced_frames.append(message)
                        elif seg[0] != -1 and seg[1] != -1:
                            self.latency_start_time.value = time.time()
                            self.inference(message, params, queue)
                            self.cache = {}
                            num_questions += 1
                        elif seg[0] == -1 and seg[1] != -1:
                            self.speech_start = False
                            voiced_frames.append(message)
                            clip_audio = b''.join(voiced_frames)
                            self.latency_start_time.value = time.time()
                            self.inference(clip_audio, params, queue)
                            self.cache = {}
                            num_questions += 1
                            voiced_frames = []
                elif self.speech_start:
                    voiced_frames.append(message)
                else:
                    self.cache = {}
            else:
                sub_messages = list(frame_generator(VAD_FRAME_DURATION_MS, message, fs))
                for sub_message in sub_messages:
                    clip_audio = vad_collector(fs, sub_message)
                    if clip_audio is not None:
                        self.latency_start_time.value = time.time()
                        self.inference(clip_audio, params, queue)
                        num_questions += 1
            if num_questions > 0:
                if args_global.audio_in is None and not args_global.output_file:
                    queue.put("我还有什么可以帮您，您可以继续提问！")
                queue.put(None) # end
            while queue.qsize():
                time.sleep(1)
            while audio_seq_queue.qsize():
                time.sleep(1)
            if num_questions > 0:
                if args_global.audio_in is None:
                    print("microphone running ...")
                else:
                    print("next segment running ...")
        if len(voiced_frames) > 0:
            if self.use_fsmn_vad:
                clip_audio = b''.join(voiced_frames)
            else:
                clip_audio = b''.join([f.bytes for f in voiced_frames])
            self.latency_start_time.value = time.time()
            self.inference(clip_audio, params, queue)
            self.cache = {}
            queue.put(None) # end
            while queue.qsize():
                time.sleep(1)
            while audio_seq_queue.qsize():
                time.sleep(1)

        time.sleep(2) # for the last string (its length < min_output_text_len)


def set_argparser(name:str):
    def whisper_parser():
        return whisperWrapper.set_argparser()
    
    def llama3_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--llm_model_path', type=str, default = '../BM1684X/llama3/llama3-8b_int4_1dev_256.bmodel', help='path to the LLM bmodel file')
        parser.add_argument('--tokenizer_path', type=str, default="../BM1684X/llama3/token_config", help='path to the tokenizer file')
        parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
        parser.add_argument('--temperature_llm', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')    
        return parser
    
    def minicpm_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--minicpm_model_path', type=str, default = '../BM1688/minicpm/minicpm-2b_int4_1core.bmodel', help='path to the LLM bmodel file') 
        parser.add_argument('--minicpm_tokenizer_model_path', type=str, default = '../BM1688/minicpm/tokenizer.model', help='path to the tokenizer file') 
        parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
        return parser

    def vits_parser():
        parser = argparse.ArgumentParser(
            description='Inference code for bert vits models')
        parser.add_argument('--vits_model', type=str, default='../BM1688/vits/vits_chinese_128_bm1688_f16_1core.bmodel', help='path of bmodel')
        parser.add_argument('--bert_model', type=str, default='../BM1688/vits/bert_1688_f32_1core.bmodel', help='path of bert config')
        parser.add_argument('-d', '--devid', type=int, default=0, help='device id')
        return parser

    def xtts_parser():
        parser = argparse.ArgumentParser(
            description='Inference code for xtts models')
        parser.add_argument('--gpt_first_inference_path', type=str, default='../BM1684X/xtts/gpt_inference_first_bm1684x_f16.bmodel', help='path of gpt first step bmodel')
        parser.add_argument('--gpt_loop_inference_path', type=str, default='../BM1684X/xtts/gpt_inference_loop_bm1684x_f16.bmodel', help='path of gpt remain step bmodel')
        parser.add_argument('--gpt_inner_path', type=str, default='../BM1684X/xtts/gpt_inner_256_bm1684x_f16.bmodel', help='path of gpt inner bmodel')
        parser.add_argument('--waveform_decoder_path', type=str, default='../BM1684X/xtts/waveform_decoder_deconv2d_bm1684x_f16.bmodel', help='path of wavefrom decoder bmodel')
        parser.add_argument('--other_models_dir', type=str, default='../BM1684X/xtts', help='path of other models')
        parser.add_argument('--config_pth', type=str, default='../BM1684X/xtts/config.json', help='path of config')
        parser.add_argument('-d', '--devid', type=int, default=0, help='device id')
        return parser
    
    parser_dict = {
        'llama3': llama3_parser,
        'minicpm': minicpm_parser,
        'whisper': whisper_parser,
        'vits': vits_parser,
        'xtts': xtts_parser
    }

    parser_factory = parser_dict.get(name.lower())

    if parser_factory is None:
        raise ValueError(f"Invalid model type '{name}'")
    
    return parser_factory()


parser_global = argparse.ArgumentParser()
parser_global.add_argument("--profile", action='store_true', help="print profiling result")
parser_global.add_argument("--audio_in", type=str, default=None, help="input audio, None for microphone")
parser_global.add_argument("--output_file", action='store_true', default=False, help="whether to output file, otherwise audio")
parser_global.add_argument("--streaming_output", action='store_true', default=False, help="whether to streaming output for file or audio")
parser_global.add_argument("--llm_type", type=str, default="minicpm-2b", help="llm model type, support minicpm-2b, llama3-8b")
parser_global.add_argument("--tts_type", type=str, default="vits", help="tts model type, support vits, xtts")
parser_global.add_argument("--tts_out_language", type=str, default="zh-cn", help="tts output language, only valid for xtts")
parser_global.add_argument("--tts_speaker", type=str, default="Daisy Studious", help="tts speaker id or speaker wav path, only valid for xtts")
parser_global.add_argument("--microphone_devid", type=int, default=0, help="microphone device id, valid when --audio_in=None")
parser_global.add_argument("--audio_devid", type=int, default=None, help="play audio device id, valid when --output_file=False, default: use default device")
parser_global.add_argument("--min_tts_input_len", type=int, default=30, help="minimum TTS input text length")
args_global, _ = parser_global.parse_known_args()


if __name__ =="__main__":
    input_audios = [
        args_global.microphone_devid
    ]

    parser_whisper, parser_llama3, parser_minicpm, parser_vits, parser_xtts = set_argparser('whisper'), set_argparser('llama3'), set_argparser('minicpm'), set_argparser('vits'), set_argparser('xtts')

    
    args_whisper, _ = parser_whisper.parse_known_args()
    args_llama3, _ = parser_llama3.parse_known_args()
    args_minicpm, _ = parser_minicpm.parse_known_args()
    args_vits, _ = parser_vits.parse_known_args()
    args_xtts, _ = parser_xtts.parse_known_args()

    console, timer = Console(), Timer()

    # ------ init 
    timer = Timer()
    with timer:
        if args_global.tts_type == "xtts":
            tts_model_config = {
                "other_models_dir": args_xtts.other_models_dir,
                "config_pth": args_xtts.config_pth,
                "gpt_first_inference_path": args_xtts.gpt_first_inference_path,
                "gpt_loop_inference_path": args_xtts.gpt_loop_inference_path,
                "gpt_inner_path": args_xtts.gpt_inner_path,
                "waveform_decoder_path": args_xtts.waveform_decoder_path,
                "devid": args_xtts.devid
            }
        elif args_global.tts_type == "vits":
            tts_model_config = args_vits
        else:
            raise ValueError(args_global.tts_type + " not implement!")
        queue = multiprocessing.Queue()
        audio_seq_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=run_tts, args=(queue, audio_seq_queue, tts_model_config))
        process.daemon = True
        process.start()
        
        if args_global.llm_type == "minicpm-2b":
            assert os.path.exists(args_minicpm.minicpm_model_path)
            assert os.path.exists(args_minicpm.minicpm_tokenizer_model_path)
            llm_model = MinicpmModel(**args_minicpm.__dict__)
            # hot start~
            output = llm_model.gen_response_line("hello!")
        elif args_global.llm_type == "llama3-8b":
            llm_model = Llama3Model(**args_llama3.__dict__)
            # llm_model = None
        print("start LLM ...")
        # NOTE only support zh
        args_whisper.language = 'zh'
        whisper_model = whisperWrapper.WhisperWrapper(args_whisper)
        print("start S2T model ...")
        # whisper_model = None
        pipeline = Pipeline(asr_model=whisper_model, llm_model=llm_model)
    console.print(f"[blue]Initialization took {timer.run_time:.3f} seconds!")
    subprocess.run('clear', shell=True)
    # ----- process
    count:int = 0
    while input_audios:
        in_audio = input_audios.pop(0)
        count += 1
        out_audio = "out_" + str(count) + ".wav"

        pipeline.forward(queue, audio_seq_queue, in_audio, out_audio, None)

        if not input_audios:
            # TODO：allow user input 
            pass 
    print("\nTest end!")
