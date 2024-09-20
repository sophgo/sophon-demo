import sys
sys.path.append("../")
sys.path.append("../whisper-TPU_py")
try:
    from Llama3 import chat
except:
    import logging
    logging.warning("Llama3 not support! If need, please refer to python/README.md.")
import argparse
import time
import wave
from transformers import AutoTokenizer
import time
import subprocess
import soundfile as sf
import wave
import numpy as np
import multiprocessing

# vits
import math
from text import cleaned_text_to_sequence, pinyin_dict
import logging
from pypinyin import lazy_pinyin, Style
from pypinyin.core import load_phrases_dict
logging.basicConfig(level=logging.INFO)

import whisperWrapper


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
                 enable_history=True, args_global=None):
    
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
        self.args_global = args_global

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
            if self.args_global.streaming_output:
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
        
        if not self.args_global.streaming_output:
            queue.put(full_text_answer)
        elif len(sentence) > 0:
            queue.put("".join(sentence))
        return answer


class MinicpmModel:
    # TODO: +prompt tokens
    def __init__(self, minicpm_model_path, minicpm_tokenizer_model_path, devid=0, args_global=None):
        self.process = subprocess.Popen(
            ['../MiniCPM/demo/minicpm', '--model', minicpm_model_path, '--tokenizer', minicpm_tokenizer_model_path, '--devid', devid],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.args_global = args_global
    def gen_response_line(self, input:str):

        self.process.stdin.write(input + '\n')
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline().strip()
            if line[:6] == 'Answer':
                return line[9:]
            
    def gen_response(self, input:str, queue):
        # TODO: +prompt tokens
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
                    if self.args_global.profile:
                        print(line)
                        line=self.process.stdout.readline().strip()
                        print(line)
                    if not self.args_global.streaming_output:
                        queue.put(''.join(lines))
                    return lines
                lines.append(line)
                print(line)
                if self.args_global.streaming_output:
                    queue.put(line)


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
    with open("../text/pinyin-local.txt", "r", encoding='utf-8') as f:
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
            self.prosody = TTSProsody(bert_model, dev_id, bert_dir='../bert')
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


class Pipeline():
    "pipeline for asr-->llm-->tts. "
    def __init__(self, asr_model=None, tts_model=None, llm_model=None, args_global=None):
        self.asr_model = asr_model
        self.tts_model = tts_model
        self.llm_model = llm_model
        self.question_prefix = "，请用中文回答。"
        self.args_global = args_global
    
    def run_asr(self, audio_path:str):
        return self.asr_model.transcribe(audio_path) 
    
    def run_llm(self, text:str, queue):
        return self.llm_model.gen_response(text, queue)

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
        timer = Timer()
        st = time.time()
        with wave.open('whisper_input.wav','wb') as wavWrite:
            wavWrite.setparams(params) 
            wavWrite.writeframes(clip_audio) 
        et = time.time()
        print("write audio cost: ", et-st)

        with timer:
            st = time.time()
            asr_prompt = self.run_asr("whisper_input.wav")
            et = time.time()
        asr_time = timer.run_time
        print("\nasr cost: ", et-st)


        # console.print(f"[green]ASR took {timer.run_time:.3f} seconds to process a {get_wav_duration(input_audio_path)} seconds audio!")
        # console.print(f"[blue]Input is:"+asr_prompt["text"])
        
        with timer:
            question = asr_prompt["text"] + self.question_prefix
            print("Question:"+question)
            llm_response = self.run_llm(question, queue)
        # print('Answer:'+llm_response)
        # console.print(f"\n[green]LLM took {timer.run_time:.4f} seconds to answer this question on BM1688!")
        llm_time = timer.run_time
        print("\nllm cost: ", et-st, flush=True)

        if self.args_global.profile:
            console.print(f"\nwhisper took {asr_time:.2f} seconds, and LLM took {llm_time:.2f} seconds.") 

    def run_tts(self, queue, client_socket):
        output_text = ''
        min_output_text_len = self.args_global.min_tts_input_len
        timer = Timer()
        while True:
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
                        break
                    continue
                if llm_response is None:
                    is_final = True
                else:
                    is_final = False
                whole_resp_text = output_text
                print('\n------audio out content: ', whole_resp_text)
                split_items = self.tts_model.split_text_near_punctuation(whole_resp_text, int(self.tts_model.max_length / 2 - 5))
                full_out_audio_data = []
                for split_item in split_items:
                    print('\n------audio seg: ', split_item)
                    if len(split_item) == 0:
                        continue
                    phonemes, char_embeds = self.tts_model.tts_front.chinese_to_phonemes(split_item)
                    input_ids = cleaned_text_to_sequence(phonemes)
                    char_embeds = np.expand_dims(char_embeds, 0)
                    x = np.array(input_ids, dtype=np.int32)
                    if self.args_global.streaming_output:
                        play_message = self.tts_model(x, char_embeds)
                        play_message = play_message.tobytes()
                        # play_message += "</pend>".encode()
                        try:
                            client_socket.sendall(play_message)
                        except Exception as e:
                            print(e)
                    else:
                        full_out_audio_data.append(self.tts_model(x, char_embeds))

                if not self.args_global.streaming_output:
                    play_message = np.concatenate(full_out_audio_data, axis=-1)
                    play_message = play_message.tobytes()
                    # play_message += "</pend>".encode()
                    try:
                        client_socket.sendall(play_message)
                    except Exception as e:
                        print(e)
                    
                output_text = ''
                if is_final:
                    break
            tts_time = timer.run_time
            if self.args_global.profile:
                console.print(f"\ntts took {tts_time:.2f} seconds.") 
    
    def run_asr_llm(self, audio_in_queue, llm_out_queue):
        timer = Timer()
        params = wave._wave_params(nchannels=1, sampwidth=2, framerate=16000, nframes=160000, comptype='NONE', compname='not compressed')
    
        # while True:
        message = audio_in_queue.get()
        
        self.inference(message, params, llm_out_queue)
        llm_out_queue.put(None) # end
        while llm_out_queue.qsize():
            time.sleep(1)
        time.sleep(2) # for the last string (its length < min_output_text_len)

