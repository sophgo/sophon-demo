# -*- encoding: utf-8 -*-
import os
import websockets, ssl
import asyncio
# import threading
import argparse
import json
from multiprocessing import Process
import platform
import collections
import webrtcvad
# from funasr.fileio.datadir_writer import DatadirWriter

import logging
import pyaudio
from funasr import AutoModel

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--hosts",
                    nargs='+',
                    type=str,
                    default=["localhost",],
                    required=False,
                    help="server ips only support not more than two elements, default: [localhost,], i.e. 0.0.0.0, two elements format[online server ip, offline server ip]")
parser.add_argument("--ports",
                    nargs='+',
                    type=int,
                    default=[10095,],
                    required=False,
                    help="server ports only support not more than two elements, default: [10095,], two elements format[online server port, offline server port]")
parser.add_argument("--chunk_duration_ms",
                    type=int,
                    default=1000,
                    help="sent chunk duration (ms), must be multiple of 10/20/30ms & bigger than 300ms")
parser.add_argument("--vad_level",
					type=int,
					default=2,
					help="webrtcvad vad granularity, [0,3], the bigger the number is, the finer granularity is")
parser.add_argument("--online_use_vad",
					action='store_true',
					default=False,
					help="whether to use vad for removing noise or empty audio segment during online"
					)
parser.add_argument("--vad_type",
                    type=str,
                    default='webrtcvad',
                    help="vad model type, support webrtcvad,fsmn-vad(only for offline)")
parser.add_argument("--encoder_chunk_look_back",
                    type=int,
                    default=4,
                    help="chunk")
parser.add_argument("--decoder_chunk_look_back",
                    type=int,
                    default=0,
                    help="chunk")
parser.add_argument("--chunk_interval",
                    type=int,
                    default=1,
                    help="chunk")
parser.add_argument("--hotword",
                    type=str,
                    default="",
                    help="hotword file path, one hotword perline (e.g.:阿里巴巴 20)")
parser.add_argument("--audio_in",
                    type=str,
                    default=None,
                    help="audio_in")
parser.add_argument("--audio_fs",
                    type=int,
                    default=16000,
                    help="audio_fs")
parser.add_argument("--microphone_dev_id",
                    type=int,
                    default=0,
                    help="microphone device id")
parser.add_argument("--thread_num",
                    type=int,
                    default=1,
                    help="thread_num, only support 1")
parser.add_argument("--words_max_print",
                    type=int,
                    default=10000,
                    help="chunk")
parser.add_argument("--output_dir",
                    type=str,
                    default=None,
                    help="output_dir")
parser.add_argument("--ssl",
                    type=int,
                    default=0,
                    help="1 for ssl connect, 0 for no ssl")
parser.add_argument("--use_itn",
                    type=int,
                    default=1,
                    help="1 for using itn, 0 for not itn")
parser.add_argument("--mode",
                    type=str,
                    default="parallel2pass",
                    help="offline, online, parallel2pass")

args = parser.parse_args()
assert len(args.hosts) == len(args.ports)
if args.mode == "parallel2pass":
    assert len(args.hosts) == 2
else:
    assert len(args.hosts) == 1
assert (args.chunk_duration_ms % 10 == 0 or args.chunk_duration_ms % 20 == 0 or args.chunk_duration_ms % 30 == 0) and args.chunk_duration_ms > 300
assert args.vad_type in ["webrtcvad", "fsmn-vad"]
if args.audio_in is None:
     assert args.vad_type == "webrtcvad"
assert args.thread_num == 1
if args.mode == "parallel2pass":
    lock = asyncio.Lock()
    text_print_parallel2pass_online = collections.deque()
    text_print_parallel2pass_offline = ""
    text_print_parallel2pass_online_frame_id = collections.deque()

# vad
VAD_FRAME_DURATION_MS = 30
PADDING_DURATION_MS = 300
num_padding_frames = int(PADDING_DURATION_MS / VAD_FRAME_DURATION_MS)
if args.vad_type == 'webrtcvad':
    if args.mode == 'parallel2pass' or args.mode == 'offline':
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        is_voice_buffer = collections.deque(maxlen=num_padding_frames)
        num_voiced = 0
        num_unvoiced = 0
        triggered = False
        voiced_frames = []
        vad = webrtcvad.Vad(args.vad_level)
    if (args.mode == 'parallel2pass' or args.mode == 'online') and args.online_use_vad:
        ring_buffer_online = collections.deque(maxlen=num_padding_frames)
        is_voice_buffer_online = collections.deque(maxlen=num_padding_frames)
        num_voiced_online = 0
        num_unvoiced_online = 0
        triggered_online = False
        voiced_frames_online = []
        vad_online = webrtcvad.Vad(args.vad_level)
elif args.vad_type == 'fsmn-vad':
    assert args.mode == 'offline'
    voiced_frames = []
    vad = AutoModel(model=args.vad_type,
                    device='cpu', disable_pbar=True, log=False)

print(args)
# voices = asyncio.Queue()
from queue import Queue

voices = Queue()
msg_done=[False] * len(args.hosts) 

if args.output_dir is not None:
    # if os.path.exists(args.output_dir):
    #     os.remove(args.output_dir)
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration
 
 
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


def vad_collector_online(sample_rate, frame):
	global num_voiced_online, num_unvoiced_online, ring_buffer_online, is_voice_buffer_online, triggered_online, voiced_frames_online, vad_online, VAD_FRAME_DURATION_MS
	# sys.stdout.write(
	#     '1' if vad.is_speech(frame.bytes, sample_rate) else '0')
	if not triggered_online:
		if len(is_voice_buffer_online) == is_voice_buffer_online.maxlen and is_voice_buffer_online[0]:
			num_voiced_online -= 1
		ring_buffer_online.append(frame)
		is_voice_buffer_online.append(vad_online.is_speech(frame.bytes, sample_rate))
		# num_voiced = len([f for f in ring_buffer
		#                   if vad.is_speech(f.bytes, sample_rate)])
		if is_voice_buffer_online[-1]:
			num_voiced_online += 1
		if num_voiced_online > 0.9 * ring_buffer_online.maxlen:
			triggered_online = True
			voiced_frames_online.extend(ring_buffer_online)
			ring_buffer_online.clear()
			is_voice_buffer_online.clear()
			num_voiced_online = 0
	else:
		if len(is_voice_buffer_online) == is_voice_buffer_online.maxlen and not is_voice_buffer_online[0]:
			num_unvoiced_online -= 1
		voiced_frames_online.append(frame)
		ring_buffer_online.append(frame)
		is_voice_buffer_online.append(vad_online.is_speech(frame.bytes, sample_rate))
		# num_unvoiced = len([f for f in ring_buffer
		#                     if not vad.is_speech(f.bytes, sample_rate)])
		if not is_voice_buffer_online[-1]:
			num_unvoiced_online += 1
		if num_unvoiced_online > 0.9 * ring_buffer_online.maxlen or (args.chunk_duration_ms is not None and len(voiced_frames_online) * VAD_FRAME_DURATION_MS >= args.chunk_duration_ms):
			triggered_online = False
			res = b''.join([f.bytes for f in voiced_frames_online])
			voiced_frames_online = []
			num_unvoiced_online = 0
			ring_buffer_online.clear()
			is_voice_buffer_online.clear()
			return res
	return None


async def record_microphone(mode, websocket):
    global voiced_frames
    # print("2")
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    # chunk_size = 60 * args.chunk_size[1] / args.chunk_interval
    CHUNK = int(args.audio_fs * (args.chunk_duration_ms / 1000.0)) # int(args.chunk_size[1] / args.chunk_interval * 2) # int(RATE / 1000 * chunk_size)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=args.audio_fs,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=args.microphone_dev_id)
    # hotwords
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        if os.path.exists(args.hotword):
            f_scp = open(args.hotword)
            hot_lines = f_scp.readlines()
            for line in hot_lines:
                words = line.strip().split(" ")
                if len(words) < 2:
                    print("Please checkout format of hotwords")
                    continue
                try:
                    fst_dict[" ".join(words[:-1])] = int(words[-1])
                except ValueError:
                    print("Please checkout format of hotwords")
            hotword_msg = json.dumps(fst_dict)
        else:
            hotword_msg = args.hotword

    use_itn = True
    if args.use_itn == 0:
        use_itn=False
    
    message = json.dumps({"mode": mode,
                          "chunk_size": CHUNK * 2,
                          "chunk_interval": args.chunk_interval,
                          "encoder_chunk_look_back": args.encoder_chunk_look_back,
                          "decoder_chunk_look_back": args.decoder_chunk_look_back,
                          "wav_name": "microphone",
                          "is_speaking": True,
                          "hotwords": hotword_msg,
                          "itn": use_itn,
                          "chunk_duration_ms": args.chunk_duration_ms,
                          "audio_fs": args.audio_fs
                          })
    #voices.put(message)
    await websocket.send(message)
    i = 0
    sleep_duration = args.chunk_duration_ms / 1000. / 4
    while True:
        data = stream.read(CHUNK)
        message = data
        sub_messages = list(frame_generator(VAD_FRAME_DURATION_MS, message, args.audio_fs))
        if mode == "offline":
            for sub_message in sub_messages:
                clip_audio = vad_collector(args.audio_fs, sub_message)
                if clip_audio is not None:
                    clip_audio += i.to_bytes(2, byteorder='little', signed=False)
                    await websocket.send(clip_audio)
        else:
            if args.online_use_vad:
                for sub_message in sub_messages:
                    clip_audio = vad_collector_online(args.audio_fs, sub_message)
                    if clip_audio is not None:
                        clip_audio += i.to_bytes(2, byteorder='little', signed=False)
                        await websocket.send(clip_audio)
            else:
                message += i.to_bytes(2, byteorder='little', signed=False)
                await websocket.send(message)
        i += 1
        #voices.put(message)
        await asyncio.sleep(sleep_duration)

async def record_from_scp(mode, websocket, chunk_begin, chunk_size, web_id):
    global voices, voiced_frames
    is_finished = False
    if args.audio_in.endswith(".scp"):
        f_scp = open(args.audio_in)
        wavs = f_scp.readlines()
    else:
        wavs = [args.audio_in]

    # hotwords
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        if os.path.exists(args.hotword):
            f_scp = open(args.hotword)
            hot_lines = f_scp.readlines()
            for line in hot_lines:
                words = line.strip().split(" ")
                if len(words) < 2:
                    print("Please checkout format of hotwords")
                    continue
                try:
                    fst_dict[" ".join(words[:-1])] = int(words[-1])
                except ValueError:
                    print("Please checkout format of hotwords")
            hotword_msg = json.dumps(fst_dict)
        else:
            hotword_msg = args.hotword
        print (hotword_msg)

    sample_rate = args.audio_fs
    wav_format = "pcm"
    use_itn=True
    if args.use_itn == 0:
        use_itn=False
     
    # print('chunk_size > 0: ', chunk_size > 0)
    # print('wavs: ', wavs, chunk_begin, chunk_begin + chunk_size)
    if chunk_size > 0:
        wavs = wavs[chunk_begin:chunk_begin + chunk_size]
    for wav in wavs:
        wav_splits = wav.strip().split()
 
        wav_name = wav_splits[0] if len(wav_splits) > 1 else "demo"
        wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
        if not len(wav_path.strip())>0:
           continue
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

        stride = int(sample_rate * (args.chunk_duration_ms / 1000.0) * 2) # int(args.chunk_size[1] / args.chunk_interval * 2) # int(60 * args.chunk_size[1] / args.chunk_interval / 1000 * sample_rate * 2)
        chunk_num = (len(audio_bytes) - 1) // stride + 1
        print('len(audio_bytes): ', len(audio_bytes), stride)
        # print(stride)

        # send first time
        message = json.dumps({"mode": mode,
                              "chunk_size": stride,
                              "chunk_interval": args.chunk_interval,
                              "encoder_chunk_look_back": args.encoder_chunk_look_back,
                              "decoder_chunk_look_back": args.decoder_chunk_look_back,
                              "audio_fs":sample_rate,
                              "wav_name": wav_name,
                              "wav_format": wav_format,
                              "is_speaking": True,
                              "hotwords": hotword_msg,
                              "chunk_duration_ms": args.chunk_duration_ms,
                              "itn": use_itn})

        #voices.put(message)
        await websocket.send(message)
        is_speaking = True
        if args.mode == "online" or args.mode == "parallel2pass":
            # TODO: update sleep_duration to adapt for microphone streaming
            sleep_duration = args.chunk_duration_ms / 1000. # input time - inference cost time(ignored)
        else:
             sleep_duration = 0
        if args.vad_type == 'webrtcvad':
            for i in range(chunk_num):

                beg = i * stride
                data = audio_bytes[beg:beg + stride]
                message = data
                #voices.put(message)
                sub_messages = list(frame_generator(VAD_FRAME_DURATION_MS, message, sample_rate))
                if mode == "offline":
                    for sub_message in sub_messages:
                        clip_audio = vad_collector(sample_rate, sub_message)
                        if clip_audio is not None:
                            clip_audio += i.to_bytes(2, byteorder='little', signed=False)
                            await websocket.send(clip_audio)
                else:
                    if args.online_use_vad:
                        for sub_message in sub_messages:
                            clip_audio = vad_collector_online(sample_rate, sub_message)
                            if clip_audio is not None:
                                clip_audio += i.to_bytes(2, byteorder='little', signed=False)
                                await websocket.send(clip_audio)
                    else:
                        message += i.to_bytes(2, byteorder='little', signed=False)
                        await websocket.send(message)
                if i == chunk_num - 1:
                    if mode == "offline" and voiced_frames:
                        audio_in = b''.join([f.bytes for f in voiced_frames])
                        voiced_frames = []
                        audio_in += i.to_bytes(2, byteorder='little', signed=False)
                        await websocket.send(audio_in)
                    is_speaking = False
                    message = json.dumps({"is_speaking": is_speaking})
                    #voices.put(message)
                    await websocket.send(message)
                
                await asyncio.sleep(sleep_duration)
        else:
            all_segs = vad.generate(input=wavs[0])[0]['value']
            for i, seg in enumerate(all_segs):
                seg = audio_bytes[seg[0] * 16 * 2 : seg[1] * 16 * 2]
                seg += i.to_bytes(2, byteorder='little', signed=False)
                await websocket.send(seg)
                await asyncio.sleep(sleep_duration)
            is_speaking = False
            message = json.dumps({"is_speaking": is_speaking})
            #voices.put(message)
            await websocket.send(message)
    
    # NOTE: ori
    # await asyncio.sleep(2)
    global msg_done
    while  not  msg_done[web_id]:
        await asyncio.sleep(1)
    
    await websocket.close()


          
async def message(id, websocket, web_id):
    global voices,msg_done,text_print_parallel2pass_online,text_print_parallel2pass_offline, text_print_parallel2pass_online_frame_id
    text_print = ""
    text_print_2pass_online = ""
    text_print_2pass_offline = ""
    if args.output_dir is not None:
        ibest_writer = open(os.path.join(args.output_dir, "text.{}".format(id)), "a", encoding="utf-8")
    else:
        ibest_writer = None
    try:
       while True:
        
            # NOTE: ori
            # meg = await websocket.recv()
            meg = await asyncio.wait_for(websocket.recv(), 10000000000000)
            meg = json.loads(meg)
            wav_name = meg.get("wav_name", "demo")
            text = meg["text"]
            timestamp=""
            msg_done[web_id] = meg.get("is_final", False)
            if "timestamp" in meg:
                timestamp = meg["timestamp"]

            if ibest_writer is not None:
                if timestamp !="":
                    text_write_line = "{}\t{}\t{}\n".format(wav_name, text, timestamp)
                else:
                    text_write_line = "{}\t{}\n".format(wav_name, text)
                ibest_writer.write(text_write_line)

            if 'mode' not in meg:
                continue
            if args.mode == "parallel2pass":
                async with lock:
                    if meg["mode"] == "online":
                        text_print_parallel2pass_online.append("{}".format(text))
                        text_print_parallel2pass_online_frame_id.append(meg["last_segment_id"])
                    else:
                        while len(text_print_parallel2pass_online_frame_id) > 0 and text_print_parallel2pass_online_frame_id[0] <= meg["last_segment_id"]:
                            text_print_parallel2pass_online_frame_id.popleft()
                            text_print_parallel2pass_online.popleft()
                        text_print_parallel2pass_offline += "{}".format(text)
                    text_print = text_print_parallel2pass_offline + "".join(text_print_parallel2pass_online)
                    if len(text_print) > args.words_max_print:
                        if len(text_print) - args.words_max_print <= len(text_print_parallel2pass_offline):
                            text_print_parallel2pass_offline = text_print_parallel2pass_offline[len(text_print) - args.words_max_print:]
                        else:
                            text_print_parallel2pass_offline = ""
                        text_print = text_print_parallel2pass_offline + "".join(text_print_parallel2pass_online)
                    if platform.system().lower() == 'linux':
                        os.system('clear')
                    elif platform.system().lower() == 'windows':
                        os.system('CLS')
                    print("\rpid" + str(id) + ": " + text_print)
            elif meg["mode"] == "online":
                text_print += "{}".format(text)
                text_print = text_print[-args.words_max_print:]
                if platform.system().lower() == 'linux':
                    os.system('clear')
                elif platform.system().lower() == 'windows':
                    os.system('CLS')
                print("\rpid" + str(id) + ": " + text_print)
            elif meg["mode"] == "offline":
                if timestamp !="":
                    text_print += "{} timestamp: {}".format(text, timestamp)
                else:
                    text_print += "{}".format(text)

                # text_print = text_print[-args.words_max_print:]
                if platform.system().lower() == 'linux':
                    os.system('clear')
                elif platform.system().lower() == 'windows':
                    os.system('CLS')
                print("\rpid" + str(id) + ": " + wav_name + ": " + text_print)
                # with open('results/' + args.audio_in.split('/')[-1].split('.')[0] + '.txt', 'w') as fw:
                #     fw.write(text_print)
                # offline_msg_done = True
            else:
                raise ValueError("not support mode!")

    except Exception as e:
            print("Exception:", e)
            #traceback.print_exc()
            #await websocket.close()
 



async def ws_client(id, chunk_begin, chunk_size):
  if args.audio_in is None:
       chunk_begin=0
       chunk_size=1
  global voices,msg_done
 
  print('chunk_begin,chunk_begin+chunk_size: ', chunk_begin,chunk_begin+chunk_size)
  for i in range(chunk_begin,chunk_begin+chunk_size):
    msg_done=[False] * len(msg_done)
    voices = Queue()
    uris = []
    if args.ssl == 1:
        ssl_context = ssl.SSLContext()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    else:
        ssl_context = None
    for j in range(len(args.hosts)):
        if args.ssl == 1:
            uri = "wss://{}:{}".format(args.hosts[j], args.ports[j])
        else:
            uri = "ws://{}:{}".format(args.hosts[j], args.ports[j])
        uris.append(uri)
        print("connect to", uri)
    
    if len(args.hosts) == 1:
        async with websockets.connect(uris[0], subprotocols=["binary"], ping_interval=None, ssl=ssl_context) as websocket:
            if args.audio_in is not None:
                task = asyncio.create_task(record_from_scp(args.mode, websocket, i, 1, 0))
            else:
                task = asyncio.create_task(record_microphone(args.mode, websocket))
            task3 = asyncio.create_task(message(str(id)+"_"+str(i), websocket, 0)) #processid+fileid
            await asyncio.gather(task, task3)
    elif len(args.hosts) == 2:
        async with websockets.connect(uris[0], subprotocols=["binary"], ping_interval=None, ssl=ssl_context) as websocket1:
            async with websockets.connect(uris[1], subprotocols=["binary"], ping_interval=None, ssl=ssl_context) as websocket2:
                if args.audio_in is not None:
                    task1 = asyncio.create_task(record_from_scp("online", websocket1, i, 1, 0))
                    task2 = asyncio.create_task(record_from_scp("offline", websocket2, i, 1, 1))
                else:
                    task1 = asyncio.create_task(record_microphone("online", websocket1))
                    task2 = asyncio.create_task(record_microphone("offline", websocket2))
                task3 = asyncio.create_task(message(str(id)+"_"+str(i), websocket1, 0)) #processid+fileid
                task4 = asyncio.create_task(message(str(id)+"_"+str(i), websocket2, 1)) #processid+fileid
                await asyncio.gather(task1, task2, task3, task4)
    else:
        raise ValueError("the number of host not more than 2")
  exit(0)
    

def one_thread(id, chunk_begin, chunk_size):
    asyncio.get_event_loop().run_until_complete(ws_client(id, chunk_begin, chunk_size))
    asyncio.get_event_loop().run_forever()

if __name__ == '__main__':
    # for microphone
    if args.audio_in is None:
        # p = Process(target=one_thread, args=(0, 0, 0))
        # p.start()
        # p.join()
        one_thread(0, 0, 0)
        print('end')
    else:
        # calculate the number of wavs for each preocess
        if args.audio_in.endswith(".scp"):
            f_scp = open(args.audio_in)
            wavs = f_scp.readlines()
        else:
            wavs = [args.audio_in]
        for wav in wavs:
            wav_splits = wav.strip().split()
            wav_name = wav_splits[0] if len(wav_splits) > 1 else "demo"
            wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
            audio_type = os.path.splitext(wav_path)[-1].lower()


        total_len = len(wavs)
        if total_len >= args.thread_num:
            chunk_size = int(total_len / args.thread_num)
            remain_wavs = total_len - chunk_size * args.thread_num
        else:
            chunk_size = 1
            remain_wavs = 0

        process_list = []
        chunk_begin = 0
        for i in range(args.thread_num):
            now_chunk_size = chunk_size
            if remain_wavs > 0:
                now_chunk_size = chunk_size + 1
                remain_wavs = remain_wavs - 1
            # process i handle wavs at chunk_begin and size of now_chunk_size
            # p = Process(target=one_thread, args=(i, chunk_begin, now_chunk_size))
            one_thread(i, chunk_begin, now_chunk_size)
            chunk_begin = chunk_begin + now_chunk_size
            # p.start()
            # process_list.append(p)

        # for i in process_list:
        #     p.join()

        print('end')
