import time
import torch
from api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tpu_inference_config = {
    "gpt_first_inference_path" : "../../BM1684X/xtts/gpt_inference_first_bm1684x_f16.bmodel",
    "gpt_loop_inference_path" :  "../../BM1684X/xtts/gpt_inference_loop_bm1684x_f16.bmodel",
    "gpt_inner_path" :           "../../BM1684X/xtts/gpt_inner_256_bm1684x_f16.bmodel",
    "waveform_decoder_path" :    "../../BM1684X/xtts/waveform_decoder_deconv2d_bm1684x_f16.bmodel",
    "devid":                     0
}
# tpu_inference_config = {
#     "chip_mode" : "soc",
#     "gpt_first_inference_path" : "./bmodel/gpt_inference_first_bm1688_f16.bmodel",
#     "gpt_loop_inference_path" :  "./bmodel/gpt_inference_loop_bm1688_f16.bmodel",
#     "gpt_inner_path" :           "./bmodel/gpt_inner_256_bm1688_f16.bmodel",
#     "waveform_decoder_path" :    "./bmodel/waveform_decoder_deconv2d_bm1688_f16.bmodel"
# }
start_time = time.time()
tts = TTS(model_path="../../BM1684X/xtts", 
          config_path="../../BM1684X/xtts/config.json",
          tpu_inference_config=tpu_inference_config).to(device)
# print(f"Init TTS took {time.time() - start_time:.2f} seconds")

text_prompts = [
    # "I'm very glad you could come",
    # "真的，天气很不错",
    "人工智能是指使用计算机或机器人来模拟人类的智能行为或思维过程的一种技术。",
    # "チュソクは私のお気に入りの祭りです",
    # "AI's accessibility can lead to innovation, but also raises concerns about data privacy and misuse"
]
# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file

def process(text, speaker, language):
    tts = TTS(model_path="../../BM1684X/xtts", 
          config_path="../../BM1684X/xtts/config.json",
          tpu_inference_config=tpu_inference_config).to(device)
    tts.tts_to_file(text=text, speaker=speaker, language=language, file_path=f"output_zh.wav")
    # tts.tts(text, speaker, language)
    

for idx, text in enumerate(text_prompts):
    # import pdb; pdb.set_trace()
    # tts.tts_to_file(text=text, speaker_wav='./XTTS-v2/samples/samples_en_sample.wav', language='en', file_path=f"output_en.wav", speaker='Daisy Studious')
    # tts.tts_to_file(text=text, language='en', file_path=f"output_en.wav", speaker='Daisy Studious')
    # tts.tts_to_file(text=text, speaker_wav='./XTTS-v2/samples/samples_zh-cn-sample.wav', language='zh-cn', file_path=f"output_zh.wav")
    tts.tts_to_file(text=text, speaker_wav='../../BM1684X/xtts/samples_zh-cn-sample.wav', language='zh-cn', file_path=f"output_zh.wav")
    # tts.tts_to_file(text=text, speaker='Daisy Studious', language='zh-cn', file_path=f"output_zh.wav")
    # tts.tts(text=text, speaker='Daisy Studious', language='zh-cn')
    """
    import multiprocessing
    # multiprocessing.set_start_method('spawn')
    play_process = multiprocessing.Process(target=process, args=(text, 'Daisy Studious', 'zh-cn'))
    play_process.daemon = True
    play_process.start()
    play_process.join()
    """

    # tts.tts_to_file(text=text, speaker_wav='./XTTS-v2/samples/samples_ja_sample.wav', language='ja', file_path=f"output_ja.wav")