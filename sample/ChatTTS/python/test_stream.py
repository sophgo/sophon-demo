import ChatTTS
import torch
import torchaudio
import numpy as np
import time as time
import os
import sounddevice as sd
import queue
import threading
audio_queue = queue.Queue()
stop_flag = False
def audio_player():
    global stop_flag
    while not stop_flag or not audio_queue.empty():
        try:
            wav = audio_queue.get(timeout=0.1)
            sd.play(wav, samplerate=sample_rate)
            sd.wait()
        except queue.Empty:
            time.sleep(0.1)
            continue
player_thread = threading.Thread(target=audio_player)
player_thread.start()

os.environ["OPENBLAS_NUM_THREADS"] = "16"
chat = ChatTTS.Chat()
chat.load(local_path='../models')
torch.set_num_threads(4)
inputs_en = """
chat T T S is a text to speech model designed for dialogue applications. 
[uv_break]it supports mixed language input [uv_break]and offers multi speaker 
capabilities with precise control over prosodic elements like 
[uv_break]laughter[uv_break][laugh], [uv_break]pauses, [uv_break]and intonation. 
[uv_break]it delivers natural and expressive speech,[uv_break]so please
[uv_break] use the project responsibly at your own risk.[uv_break]
""".replace('\n', '')
sample_rate = 24000
torch.manual_seed(1222)
start = time.time()
count = 0
wav_len = 0
wavs = None
for wav in chat.infer(inputs_en,
                  skip_refine_text=True, use_decoder=True, stream=True,
                  params_refine_text = ChatTTS.Chat.RefineTextParams(
                      prompt='[oral_2][laugh_0][break_4]',
                      ),
                  params_infer_code = ChatTTS.Chat.InferCodeParams(
                      prompt="[speed_5]",
                      temperature=0.3,
                      spk_emb=chat.sample_random_speaker_num())):
    wav_len += wav.shape[1] / sample_rate
    if wavs is None:
        wavs = wav
    else:
        wavs = torch.cat((wavs, wav), dim=1)
    wav = wav.squeeze()
    audio_queue.put(wav.numpy())
time_cost = time.time() - start

torchaudio.save("test_stream.wav", wavs, sample_rate=sample_rate)

print("Real-Time Factor(RTF): ", time_cost / wav_len )
stop_flag = True
player_thread.join()