#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import argparse
import logging
import librosa
import numpy as np
import soundfile as sf
import sophon.sail as sail
import time
import math

logging.basicConfig(level=logging.INFO)
 
class MP_SENET(object):
    def __init__(self, args):
        #self.net = Engine(args.mp_senet_model, device_id=args.dev_id,  graph_id=0, mode=sail.IOMode.SYSIO)
        self.net = sail.Engine(args.mp_senet_model, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.mp_senet_model))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_names[0])
        self.max_length = self.input_shape[2]
        self.wav_stft_L_list = []

        self.compress_factor = args.compress_factor
        self.n_fft = args.n_fft
        self.hop_size = args.hop_size
        self.win_size = args.win_size
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.sample_rate = args.sampling_rate
        self.noisy_end_len = 0 # The actual data length of the last speech segment
        self.norm_factor = 0
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)

        self.preprocess_time = 0
        self.postprocess_time = 0
        self.inference_time = 0
    
    def init(self): 
        pass
        
    def preprocess(self, wav_file):
        
        start_time = time.time()

        noisy_wav, _ = librosa.load(wav_file, sr=self.sample_rate)
        self.norm_factor = np.sqrt(len(noisy_wav) / np.sum(noisy_wav ** 2.0))
        noisy_wav = (noisy_wav * self.norm_factor).astype(np.float32)
        noisy_wav = noisy_wav[np.newaxis, :]
        max_wav_length = (self.max_length - 1) * self.hop_size + (self.hop_size - 1)
        noisy_wav_list = [noisy_wav[:, start_indx:start_indx+max_wav_length] for start_indx in range(0, noisy_wav.shape[1], max_wav_length)]

        noisy_amp_list = []
        noisy_pha_list = []
        self.wav_stft_L_list = []
        
        for noisy_wav in noisy_wav_list:
            try:
                if(noisy_wav.shape[1]<self.n_fft):
                    continue
                self.wav_stft_L_list.append(noisy_wav.shape[1])
                noisy_amp, noisy_pha, noisy_com = self.mag_pha_stft(noisy_wav)
            except Exception as e:
                print(e)
                continue
            self.noisy_end_len = noisy_amp.shape[-1]
            if noisy_amp.shape[-1] < self.max_length:
                # Create a zero array with the same data type as nois_amp and nois_pha
                zeros_amp = np.zeros((noisy_amp.shape[0], noisy_amp.shape[1], int(self.max_length*0.1)), dtype=noisy_amp.dtype)
                zeros_pha = np.zeros((noisy_pha.shape[0], noisy_pha.shape[1], int(self.max_length*0.1)), dtype=noisy_pha.dtype)
                # Connect the array along the third dimension (axis=2)
                add_noisy_amp = np.concatenate((zeros_amp, noisy_amp), axis=2)
                add_noisy_pha = np.concatenate((zeros_pha, noisy_pha), axis=2)
                repeat_n = math.ceil((self.max_length-noisy_amp.shape[-1])/add_noisy_amp.shape[-1])

                for i in range(repeat_n):
                    noisy_amp = np.concatenate((noisy_amp, add_noisy_amp), axis=2)
                    noisy_pha = np.concatenate((noisy_pha, add_noisy_pha), axis=2)
                # Prevent exceeding the maximum audio processing length   
                noisy_amp = noisy_amp[:, :, :self.max_length]
                noisy_pha = noisy_pha[:, :, :self.max_length]

            noisy_amp_list.append(noisy_amp)
            noisy_pha_list.append(noisy_pha)
        
        self.preprocess_time += time.time() - start_time
        
        return noisy_amp_list, noisy_pha_list
    
    def inference(self, noisy_amp_list, noisy_pha_list):

        start_time = time.time()
        result_list = []

        for i in range(len(noisy_amp_list)):
            res_dic = self.net.process(self.graph_name, {self.input_names[0] : noisy_amp_list[i],self.input_names[1] :  noisy_pha_list[i]})
            result_list.append(res_dic)

        self.inference_time += time.time() - start_time

        return result_list

    def postprocess(self, result_list):

        start_time = time.time()
        clean_wav_list = []
        real_len = self.max_length
        for i in range(len(result_list)):

            if i == len(result_list) - 1:
                real_len = self.noisy_end_len

            amp_g = result_list[i][self.output_names[0]][:, :, :real_len]
            pha_g = result_list[i][self.output_names[1]][:, :, :real_len]

            audio_g = self.mag_pha_istft(amp_g, pha_g ,self.wav_stft_L_list[i])
            audio_g = audio_g / self.norm_factor
            clean_wav_list.append(audio_g)
        
        clean_wav = np.concatenate(clean_wav_list, axis=1)
        self.postprocess_time += time.time() - start_time

        return clean_wav

    def mag_pha_stft(self, y):
        if 0:
            spec_left = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_size,win_length=self.win_size,window='hann',center=True,pad_mode='reflect')
            real_part = spec_left.real
            imag_part = spec_left.imag
            stft_spec = np.stack((real_part, imag_part), axis=-1)
        else:
            stft_R, stft_I = self.bmcv.stft(y, y, True, False, self.n_fft, self.hop_size, 1, 0)
            stft_spec = np.stack((stft_R, stft_I), axis=-1)

        mag = np.sqrt(np.sum(np.square(stft_spec), axis=-1) + 1e-9)
        pha = np.arctan2(stft_spec[:, :, :, 1] + 1e-10, stft_spec[:, :, :, 0] + 1e-5)
        # Magnitude Compression
        mag = np.power(mag, self.compress_factor)
        
        # Calculate the real and imaginary parts of complex numbers
        real = mag * np.cos(pha)
        imag = mag * np.sin(pha)

        # Composite complex array
        com = np.stack((real, imag), axis=-1)
        return mag, pha, com

    def mag_pha_istft(self, mag, pha, wav_L):

        mag = np.power(mag, 1.0 / self.compress_factor)
        real_part = mag * np.cos(pha)
        imag_part = mag * np.sin(pha)

        # Create a complex array
        com = real_part + 1j * imag_part
        wav = self.bmcv.istft(com.real, com.imag, True, False, wav_L, self.hop_size, 1, 0)

        return wav[0]

    def __call__(self):
        pass
        return 0


def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser(
        description='Inference code for mp_senet models')
    parser.add_argument('--mp_senet_model', type=str, default='./models/BM1684X/mpsenet_vb_1b_bf16.bmodel', help='path of bmodel')
    parser.add_argument('--wav_files', type=str, default='./datasets/', help='path of wav files')
    parser.add_argument('--result_files', type=str, default='./python/results/', help='path of result files')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--compress_factor',type=float, default=0.30, help='compress factor')
    parser.add_argument('--sampling_rate',type=int, default=16000, help='sampling rate')
    parser.add_argument('--n_fft',type=int, default=400, help='n_fft')
    parser.add_argument('--hop_size',type=int, default=100, help='hop_size')
    parser.add_argument('--win_size',type=int, default=400, help='win_size')
    args = parser.parse_args()
    

    if 'bf16' in args.mp_senet_model:
        args.compress_factor = 0.27
        args.hop_size = 150
    
    mp_senet=MP_SENET(args)
    test_indexes = os.listdir(args.wav_files)
    os.makedirs(args.result_files, exist_ok=True)
    #The first startup takes a long time, so it needs to be preloaded once. Approximately takes 3 seconds.
    librosa.load(os.path.join(args.wav_files, test_indexes[0]))
    
    n=0
    total_time = time.time()

    for index in test_indexes:
        
        noisy_amp_list, noisy_pha_list = mp_senet.preprocess(os.path.join(args.wav_files, index))
        
        result_list = mp_senet.inference(noisy_amp_list, noisy_pha_list)

        clean_wav = mp_senet.postprocess(result_list)

        n+=1
        output_file = os.path.join(args.result_files, index)
        sf.write(output_file, clean_wav.squeeze(), args.sampling_rate, 'PCM_16')
    
    total_time = time.time() - total_time

    # calculate speed
    logging.info("------------------ Predict Time Info ----------------------")
    logging.info("wav nums: {}, preprocess_time(ms): {:.2f}".format(n, mp_senet.preprocess_time * 1000))
    logging.info("wav nums: {}, inference_time(ms): {:.2f}".format(n, mp_senet.inference_time * 1000))
    logging.info("wav nums: {}, postprocess_time(ms): {:.2f}".format(n, mp_senet.postprocess_time * 1000))
    logging.info("wav nums: {}, total_time(ms): {:.2f}".format(n, total_time * 1000))

if __name__ == "__main__":
    main()
