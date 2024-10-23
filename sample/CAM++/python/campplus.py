#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import torchaudio
import torchaudio.compliance.kaldi as Kaldi
import numpy as np
import sophon.sail as sail
import os
import time
import argparse

class FBank(object):
    def __init__(self, n_mels, sample_rate, mean_nor: bool = False,):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr==self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0]==1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=sr, dither=dither)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat

class Campplus:
    def __init__(self, args):
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.decode_time = 0.0

    def load_wav(self, wav_file, obj_fs=16000):
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
            wav, fs = torchaudio.sox_effects.apply_effects_tensor(
                wav, fs, effects=[['rate', str(obj_fs)]]
            )
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        return wav

    def __call__(self, wav_file):
        # load wav
        start_decode = time.time()
        wav = self.load_wav(wav_file)
        self.decode_time += time.time() - start_decode
        # compute feat
        start_preprocess = time.time()
        feat = self.feature_extractor(wav).unsqueeze(0)
        self.preprocess_time += time.time() - start_preprocess
        # compute embedding
        start_inference = time.time()
        output = self.net.process("campplus", {"feature":feat.numpy()})
        self.inference_time += time.time() - start_inference
        if 'embedding_BatchNormalization_f32' in output:
            embedding = output['embedding_BatchNormalization_f32'][0]
        else:
            embedding = output['embedding_BatchNormalization'][0]

        return embedding

def cosine_similarity(embedding1, embedding2, eps):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    scores = dot_product / (norm1 * norm2 + eps)
    return scores

def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # initialize net
    campplus = Campplus(args)
    # calculate embedding and save into output_dir
    outputs = []
    filenames = []
    print(f'[INFO]: Extracting embeddings...')
    for input_file in os.listdir(args.input):
        filenames.append(input_file)
        wav_file = os.path.join(args.input,input_file)
        embedding = campplus(wav_file)
        outputs.append(embedding)
        save_path = os.path.join(output_dir,os.path.basename(input_file).rsplit('.', 1)[0]+".npy")
        np.save(save_path, embedding)
        print(f'[INFO]: The extracted embedding from {input_file} is saved to {save_path}')

    # compute similarity score
    print('[INFO]: Computing the similarity score...')
    for i in range(len(outputs)):
        for j in range(i+1,len(outputs)):
            scores = cosine_similarity(outputs[i],outputs[j],1e-6)
            print('[INFO]: The similarity score between ' + filenames[i] + ' and ' + filenames[j] + ' is %7.4f' % scores)

    cn = len(outputs)
    decode_time = campplus.decode_time / cn *1000
    preprocess_time = campplus.preprocess_time / cn *1000
    inference_time = campplus.inference_time / cn *1000
    postprocess_time = campplus.postprocess_time / cn *1000
    print('#'*28)
    print('SUMMARY: Campplus detect')
    print('#'*28)
    print('[     Campplus decode_time]  loops: '+format(cn,'4d')+' avg: '+format(decode_time,'.3f')+' ms')
    print('[ Campplus preprocess_time]  loops: '+format(cn,'4d')+' avg: '+format(preprocess_time,'.3f')+' ms')
    print('[       Campplus inference]  loops: '+format(cn,'4d')+' avg: '+format(inference_time,'.3f')+' ms')
    print('[Campplus postprocess_time]  loops: '+format(cn,'4d')+' avg: '+format(postprocess_time,'.3f')+' ms')

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/campplus_bm1684x_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
