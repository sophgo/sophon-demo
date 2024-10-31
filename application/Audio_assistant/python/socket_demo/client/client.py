# -*- encoding: utf-8 -*-
import socket
import time
# import threading
import argparse
import json
import collections
import webrtcvad
# from funasr.fileio.datadir_writer import DatadirWriter

import logging
import pyaudio
from funasr import AutoModel
import numpy as np
from datetime import datetime
import soundfile as sf
import multiprocessing
from threading import Thread

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    type=str,
                    default="localhost",
                    help="server ip, default: localhost")
parser.add_argument("--port",
                    type=int,
                    default=10095,
                    help="server port, default: 10095")
parser.add_argument("--chunk_duration_ms",
                    type=int,
                    default=200, # 1000,
                    help="chunk duration (ms) input to vad, must be multiple of 10/20/30ms & bigger than 300ms (when use webrtcvad)")
parser.add_argument("--vad_level",
					type=int,
					default=2,
					help="webrtcvad vad granularity, [0,3], the bigger the number is, the finer granularity is")
parser.add_argument("--vad_type",
                    type=str,
                    default='fsmn-vad',
                    help="vad model type, support webrtcvad,fsmn-vad")
parser.add_argument("--audio_in",
                    type=str,
                    default=None,
                    help="audio_in")
parser.add_argument("--audio_fs",
                    type=int,
                    default=16000,
                    help="audio_fs")
parser.add_argument("--microphone_devid",
                    type=int,
                    default=0,
                    help="microphone device id")
parser.add_argument("--audio_devid", 
                    type=int, 
                    default=None, 
                    help="play audio device id, valid when --output_file=False, default: use default device")
parser.add_argument("--output_file",
                    action='store_true', 
                    default=False, 
                    help="whether to output file, otherwise audio")


args = parser.parse_args()
if args.vad_type == "webrtcvad":
    assert (args.chunk_duration_ms % 10 == 0 or args.chunk_duration_ms % 20 == 0 or args.chunk_duration_ms % 30 == 0) and args.chunk_duration_ms > 300
assert args.vad_type in ["webrtcvad", "fsmn-vad"]

# vad
if args.vad_type == 'fsmn-vad':
    voiced_frames = []
    cache = {}
    speech_start = False
    vad = AutoModel(model="fsmn-vad", model_revision="v2.0.4", device='cpu', disable_pbar=True, log=False)
else:
    VAD_FRAME_DURATION_MS = 30
    PADDING_DURATION_MS = 300
    num_padding_frames = int(PADDING_DURATION_MS / VAD_FRAME_DURATION_MS)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    is_voice_buffer = collections.deque(maxlen=num_padding_frames)
    num_voiced = 0
    num_unvoiced = 0
    triggered = False
    voiced_frames = []
    vad = webrtcvad.Vad(args.vad_level)
print(args)


def load_bytes(input):
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

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n
 
 
def webrtcvad_collector(sample_rate, frame):
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

def vad_socket(client_socket, message, single_infer_done, latency_start_time, output_audio_queue):
    global voiced_frames, cache, speech_start, vad
    num_questions = 0
    if args.vad_type == 'fsmn-vad':
        message_arr = load_bytes(message)
        res = vad.generate(input=message_arr, cache=cache, is_final=False, chunk_size=args.chunk_duration_ms)
        # cases: [beg, -1], [], [-1, end], [beg, end]
        if len(res[0]["value"]):
            for seg in res[0]["value"]:
                if seg[0] != -1 and seg[1] == -1:
                    speech_start = True
                    voiced_frames.append(message)
                elif seg[0] != -1 and seg[1] != -1:
                    # send a sentence
                    message += bytes("</end>".encode())
                    latency_start_time.value = time.time()
                    client_socket.sendall(message)
                    print("get question, processing...")
                    get_message(client_socket, output_audio_queue)
                    while not single_infer_done.value:
                        time.sleep(1)                        
                    single_infer_done.value = 0
                    num_questions += 1
                elif seg[0] == -1 and seg[1] != -1:
                    speech_start = False
                    voiced_frames.append(message)
                    clip_audio = b''.join(voiced_frames)
                    # send a sentence
                    clip_audio += bytes("</end>".encode())
                    latency_start_time.value = time.time()
                    client_socket.sendall(clip_audio)
                    print("get question, processing...")
                    get_message(client_socket, output_audio_queue)
                    while not single_infer_done.value:
                        time.sleep(1)                        
                    single_infer_done.value = 0
                    num_questions += 1
                    voiced_frames = []
        elif speech_start:
            voiced_frames.append(message)
    else:
        sub_messages = list(frame_generator(VAD_FRAME_DURATION_MS, message, args.audio_fs))
        for sub_message in sub_messages:
            clip_audio = webrtcvad_collector(args.audio_fs, sub_message)
            if clip_audio is not None:
                # send a sentence
                clip_audio += bytes("</end>".encode())
                latency_start_time.value = time.time()
                client_socket.sendall(clip_audio)
                print("get question, processing...")
                get_message(client_socket, output_audio_queue)
                while not single_infer_done.value:
                    time.sleep(1)                        
                single_infer_done.value = 0
                num_questions += 1
    return num_questions

def record_microphone(client_socket, single_infer_done, latency_start_time, output_audio_queue):
    # init audio stream
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK = int(args.audio_fs * (args.chunk_duration_ms / 1000.0))
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=args.audio_fs,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=args.microphone_devid)

    # loop for reading data
    print("microphone running ...")
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        message = data
        num_questions = vad_socket(client_socket, message, single_infer_done, latency_start_time, output_audio_queue)
        if num_questions > 0:
            print("microphone running ...")

def record_from_wav(client_socket, wav_path, recv_dara_close, single_infer_done, latency_start_time, output_audio_queue):
    global voiced_frames
    is_finished = False

    # read file
    sample_rate = args.audio_fs
    wav_format = "pcm"
    if wav_path.endswith(".pcm"):
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
    elif wav_path.endswith(".wav"):
        import wave
        with wave.open(wav_path, "rb") as wav_file:
            params = wav_file.getparams()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            audio_bytes = bytes(frames)        
    else:
        wav_format = "others"
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

    stride = int(sample_rate * (args.chunk_duration_ms / 1000.0) * 2) 
    chunk_num = (len(audio_bytes) - 1) // stride + 1
    print('len(audio_bytes): ', len(audio_bytes), ', stride: ', stride)

    is_speaking = True
    for i in range(chunk_num):
        beg = i * stride
        data = audio_bytes[beg:beg + stride]
        message = data
        num_questions = vad_socket(client_socket, message, single_infer_done, latency_start_time, output_audio_queue)
        if num_questions > 0:
            print("next segment running ...")

        if i == chunk_num - 1:
            if voiced_frames:
                if args.vad_type == 'fsmn-vad':
                    audio_in = b''.join(voiced_frames)
                else:
                    audio_in = b''.join([f.bytes for f in voiced_frames])
                voiced_frames = []
                audio_in += bytes("</end>".encode())
                latency_start_time.value = time.time()
                client_socket.sendall(audio_in)
                print("get question, processing...")
                get_message(client_socket, output_audio_queue)
                while not single_infer_done.value:
                    time.sleep(1)                        
                single_infer_done.value = 0
    
    client_socket.close()
    print("connection close")
    recv_dara_close.value = 1

def play(output_audio_queue, recv_dara_close, single_infer_done, latency_start_time):
    # out stream
    if not args.output_file:
        p = pyaudio.PyAudio()
        out_stream = p.open(format=pyaudio.paInt16,
            channels=1,
            rate=args.audio_fs,
            output=True,
            output_device_index=args.audio_devid) 
    else:
        out_stream = None
    indx = 0
    while True:
        while output_audio_queue.empty():
            if recv_dara_close.value:
                break
            time.sleep(1)
        if recv_dara_close.value and output_audio_queue.empty():
            break
        audio_data = output_audio_queue.get()
        if audio_data is None:
            single_infer_done.value = 1
            continue
        if latency_start_time.value != -1:
            latency = time.time() - latency_start_time.value
            print('latency time(s): ', latency)
            latency_start_time.value = -1
        if args.output_file:
            now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_').replace('-', '_').replace(':', '_')
            audio_path = f"{now_time}.wav"
            sf.write(
                audio_path, audio_data, args.audio_fs)
            print("save audio to {}".format(audio_path))
        else:
            data_int16 = audio_data * 32767
            data_int16 = data_int16.astype('int16')
            data_bytes = data_int16.tobytes()
            out_stream.write(data_bytes)
    
          
def get_message(client_socket, output_audio_queue):
    min_len = 1024 * 200
    final = False
    try:
       while not final:
            full_data = b''
            while True:
                meg = client_socket.recv(1024)
                try:
                    if meg[-6:].decode() == "</end>":
                        full_data += meg[:-6]
                        if len(full_data) > 0:
                            meg_array = np.frombuffer(full_data, dtype=np.float32)
                            # play(meg_array)
                            output_audio_queue.put(meg_array)
                            full_data = b''
                        output_audio_queue.put(None)
                        final = True
                        break
                except Exception as e:
                    pass
                full_data += meg
                if len(full_data) > min_len:
                    full_data_arr = np.frombuffer(full_data, dtype=np.float32)
                    output_audio_queue.put(full_data_arr)
                    full_data = b""
                """
                try:
                    if meg[-7:].decode() == "</pend>":
                        full_data += meg[:-7]
                        print("pend ", len(full_data))
                        meg = np.frombuffer(full_data, dtype=np.float32)
                        play(meg)
                        full_data = b''
                        break
                    else:
                        full_data += meg
                except:
                    full_data += meg
                """
    except socket.error as e:
        print(e)
        exit(-1)
    except Exception as e:
        print("Exception:", e)
        exit(-1)


def ws_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.host, args.port))
    client_sockets = multiprocessing.Queue()
    single_infer_done = multiprocessing.Value("d", 0)
    output_audio_queue = multiprocessing.Queue()
    recv_dara_close = multiprocessing.Value("d", 0)
    latency_start_time = multiprocessing.Value('d', -1)


    p_send_data, p_recv_data, p_play_data = None, None, None
    if args.audio_in is not None:
        p_send_data = Thread(target=record_from_wav, args=(client_socket, args.audio_in, recv_dara_close, single_infer_done, latency_start_time, output_audio_queue))  
        p_send_data.start() 
    else:
        p_send_data = Thread(target=record_microphone, args=(client_socket, single_infer_done, latency_start_time, output_audio_queue))  
        p_send_data.start() 
    p_play_data = multiprocessing.Process(target=play, args=(output_audio_queue, recv_dara_close, single_infer_done, latency_start_time))  
    p_play_data.start() 
    p_play_data.join()

if __name__ == '__main__':
    ws_client()
    print('end')
