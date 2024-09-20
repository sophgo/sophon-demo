import pyaudio
import multiprocessing
import wave
import time

# microphone device id
input_device_index = 0
def tts(queue, fs):
    print("tts start")
    if True:
        p = pyaudio.PyAudio()
        out_stream = p.open(format=pyaudio.paInt16,
            channels=1,
            rate=fs,
            output=True) 
    else:
        out_stream = None
    while True:
        chunk = queue.get()
        print(len(chunk))
        out_stream.write(chunk)

def stt(queue):
    global input_device_index
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    # chunk_size = 60 * args.chunk_size[1] / args.chunk_interval
    fs = 16000
    CHUNK = int(fs * (1000 / 1000.0)) # int(args.chunk_size[1] / args.chunk_interval * 2) # int(RATE / 1000 * chunk_size)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=fs,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=input_device_index)
    params = wave._wave_params(nchannels=1, sampwidth=2, framerate=16000, nframes=160000, comptype='NONE', compname='not compressed')

    while True:
        queue.put(stream.read(CHUNK))
        time.sleep(1)

queue = multiprocessing.Queue()
process = multiprocessing.Process(target=tts, args=(queue, 16000))
process.daemon = True
process.start()
stt(queue)


