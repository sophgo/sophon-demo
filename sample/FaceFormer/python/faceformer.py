#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sophon.sail as sail
import torch
import torch.nn.functional as F
import os,argparse,pickle
import librosa
import numpy as np
import time
from transformers import Wav2Vec2Processor

# 生成mask，并将布尔数组转换为0和1的数组
def enc_dec_mask(dataset, T, S):
    mask = np.ones((T, S), dtype=bool) 
    if dataset == "BIWI":
        for i in range(T):
            if (i * 2 + 2) <= S:  
                mask[i, i*2:i*2+2] = False
    elif dataset == "vocaset":
        for i in range(T):
            if i < S: 
                mask[i, i] = False
    return mask.astype(np.float32)

class Faceformer:
    def __init__(self, handle, engine, args):
        self.args = args
        self.handle = handle

        # load bmodels
        self.net = engine

        # initialize faceformer parameters
        self.dataset = self.args.dataset
        self.name_encoder_1 = "audio_encoder_1"
        self.name_encoder_2 = "audio_encoder_2"
        self.name_ppe = "ppe"
        self.name_decoder = "decoder"

        # faceformer preprocess
        self.preprocess_time_start = time.time()
        self.audio_feature, self.one_hot, self.template = self.audioPreprocess(self.args)
        self.preprocess_time = time.time() - self.preprocess_time_start
        print('preprocess_time: ', self.preprocess_time)


    def linear_interpolation(self, features, input_fps, output_fps, output_len=None):
        features = features.transpose(1, 2)
        seq_len = features.shape[2] / float(input_fps)
        if output_len is None:
            output_len = int(seq_len * output_fps)
        output_features = F.interpolate(features,size=output_len,align_corners=True,mode='linear')
        return output_features.transpose(1, 2)
        
    def audioPreprocess(self, args):
        template_file = os.path.join('../tools/',args.dataset, args.template_path)
        with open(template_file, 'rb') as fin:
            templates = pickle.load(fin,encoding='latin1')
        train_subjects_list = [i for i in args.train_subjects.split(" ")]
        one_hot_labels = np.eye(len(train_subjects_list))
        iter = train_subjects_list.index(args.condition)
        one_hot = one_hot_labels[iter]
        one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
        temp = templates[args.subject]
        template = temp.reshape((-1))
        template = np.reshape(template,(-1,template.shape[0]))
        wav_path = args.wav_path
        speech_array, _ = librosa.load(os.path.join(wav_path), sr=16000)
        processor = Wav2Vec2Processor.from_pretrained("../tools/wav2vec2-base-960h")
        processor_output = processor(speech_array,sampling_rate=16000).input_values
        audio_feature = np.squeeze(processor_output)
        audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
        return audio_feature.astype(np.float32), one_hot.astype(np.float32), template.astype(np.float32)
    
    def predict(self):
        input_encoder_1_tensors = {'audio_feature': self.audio_feature}
        frame_num = self.audio_feature.shape[1] // 534
        output_encoder_1_names = self.net.get_output_names(self.name_encoder_1)
        output_encoder_1_tensors = self.net.process(self.name_encoder_1, input_encoder_1_tensors)
        hidden_states_Transpose = torch.from_numpy(output_encoder_1_tensors[output_encoder_1_names[0]])
        hidden_states_out1 = self.linear_interpolation(hidden_states_Transpose, 50, 30, output_len=None)
        hidden_states_out1 = hidden_states_out1.numpy()
        input_encoder_2_tensors = {'hidden_states': hidden_states_out1, 'one_hot': self.one_hot}
        output_encoder_2_names = self.net.get_output_names(self.name_encoder_2)
        output_encoder_2_tensors = self.net.process(self.name_encoder_2, input_encoder_2_tensors)
        hidden_states = output_encoder_2_tensors[output_encoder_2_names[0]]
        obj_embedding = output_encoder_2_tensors[output_encoder_2_names[1]]
        output_ppe_names = self.net.get_output_names(self.name_ppe)
        output_decoder_names = self.net.get_output_names(self.name_decoder)
        for i in range(frame_num):
            if i==0:
                vertice_emb = np.expand_dims(obj_embedding, axis=1) # (1,1,64)
                style_emb = vertice_emb
                input_ppe_tensors_first = {'embedding': style_emb}
                output_ppe_tensors = self.net.process(self.name_ppe, input_ppe_tensors_first)
                vertice_input = output_ppe_tensors[output_ppe_names[0]]
            else:
                input_ppe_tensors_next = {'embedding': vertice_emb}
                output_ppe_tensors = self.net.process(self.name_ppe, input_ppe_tensors_next)
                vertice_input = output_ppe_tensors[output_ppe_names[0]]
            memory_mask = enc_dec_mask("vocaset", vertice_input.shape[1], frame_num)
            input_decoder_tensors = {'vertice_input': vertice_input, 'hidden_states': hidden_states, 'memory_mask': memory_mask}
            output_decoder_tensors = self.net.process(self.name_decoder, input_decoder_tensors)
            vertice_out = output_decoder_tensors[output_decoder_names[0]]
            new_output = output_decoder_tensors[output_decoder_names[1]]
            new_output = new_output + style_emb
            vertice_emb = np.concatenate((vertice_emb, new_output), axis=1)
        template_in = np.expand_dims(self.template, axis=1)
        vertice_out = vertice_out + template_in
        prediction = vertice_out
        prediction = np.squeeze(prediction)
        return prediction

def main(args):
    handle = sail.Handle(args.dev_id)
    engine = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
    
    ret = Faceformer(handle, engine, args)
    time_start = time.time()
    result = ret.predict()
    inference_time = time.time() - time_start
    print('inference_time: ', inference_time)
    print('result.shape: ', result.shape)

def argsparser():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--model_name", type=str, default="vocaset")
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--condition", type=str, default="FaceTalk_170913_03279_TA", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="FaceTalk_170809_00138_TA", help='FaceTalk_170809_00138_TA select a subject from test_subjects or train_subjects')
    parser.add_argument("--wav_path", type=str, default="../Data/wav/test2.mp3", help='path of the input audio signal')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/faceformer_f32.bmodel', help='path of encoder and ppe bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done')