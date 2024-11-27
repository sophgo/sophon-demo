import ChatTTS
import torch
import torchaudio
import numpy as np
import time as time
chat = ChatTTS.Chat()
chat.load(local_path='../models')

inputs_en = """
chat T T S is a text to speech model designed for dialogue applications. 
[uv_break]it supports mixed language input [uv_break]and offers multi speaker 
capabilities with precise control over prosodic elements like 
[uv_break]laughter[uv_break][laugh], [uv_break]pauses, [uv_break]and intonation. 
[uv_break]it delivers natural and expressive speech,[uv_break]so please
[uv_break] use the project responsibly at your own risk.[uv_break]
""".replace('\n', '')

torch.manual_seed(1222)
start = time.time()
wavs = chat.infer(inputs_en,
                  skip_refine_text=True, use_decoder=True, 
                  params_refine_text = ChatTTS.Chat.RefineTextParams(
                      prompt='[oral_2][laugh_0][break_4]',
                      ),
                  params_infer_code = ChatTTS.Chat.InferCodeParams(
                      prompt="[speed_5]",
                      temperature=0.3,
                      spk_emb=chat.sample_random_speaker_num()))
time_cost = time.time() - start 

sample_rate = 24000
wav_len = wavs.shape[1] / sample_rate 

print("Real-Time Factor(RTF): ", wav_len / time_cost)
torchaudio.save("test.wav", wavs, sample_rate=sample_rate)