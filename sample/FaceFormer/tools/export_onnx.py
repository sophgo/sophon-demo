#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
import librosa
import os,argparse,pickle
from faceformer import Faceformer
from transformers import Wav2Vec2Processor
import torch
import torch.nn
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
parser.add_argument("--model_name", type=str, default="vocaset")
parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
parser.add_argument("--fps", type=float, default=30, help='frame rate - 30 for vocaset; 25 for BIWI')
parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")
parser.add_argument("--output_path", type=str, default="../Data/output", help='path of the rendered video sequence')
parser.add_argument("--wav_path", type=str, default="../Data/wav/test1.wav", help='path of the input audio signal')
parser.add_argument("--result_path", type=str, default="../Data/result", help='path of the predictions')
parser.add_argument("--condition", type=str, default="FaceTalk_170913_03279_TA", help='select a conditioning subject from train_subjects')
parser.add_argument("--subject", type=str, default="FaceTalk_170809_00138_TA", help='FaceTalk_170809_00138_TA select a subject from test_subjects or train_subjects')
parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
parser.add_argument("--render_template_path", type=str, default="templates", help='path of the mesh in BIWI/FLAME topology')
args = parser.parse_args()

device = torch.device("cpu")

model = Faceformer(args)
model_path = os.path.join(args.dataset, '{}.pth'.format(args.model_name))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


# 设置目标文件夹路径
export_path = '../models/onnx'

# 检查文件夹是否存在
if not os.path.exists(export_path):
    # 文件夹不存在，创建文件夹
    os.makedirs(export_path)
    print("文件夹已创建：", export_path)
else:
    print("文件夹已存在：", export_path)

def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

class FaceFormerAudioEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = model.audio_encoder
        self.audio_feature_map = model.audio_feature_map
        self.obj_vector = model.obj_vector

    def forward(self, audio, one_hot, dataset):
        obj_embedding = self.obj_vector(one_hot)
        hidden_states = self.audio_encoder(audio, dataset).last_hidden_state
        hidden_states = self.audio_feature_map(hidden_states)
        return obj_embedding, hidden_states
    
class FaceFormerPPE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.PPE = model.PPE

    def forward(self, emb):
        return self.PPE(emb)
    
class FaceFormerDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.biased_mask = model.biased_mask
        self.transformer_decoder = model.transformer_decoder
        self.vertice_map_r = model.vertice_map_r
        self.vertice_map = model.vertice_map
        
    
    def forward(self, vertice_input, hidden_states, memory_mask):
        tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device)
        # memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
        vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
        vertice_out = self.vertice_map_r(vertice_out)
        new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
        return vertice_out, new_output

# linear interpolation layer
def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features,size=output_len,align_corners=True,mode='linear')
    return output_features.transpose(1, 2)

class AudioModelFirst(torch.nn.Module):
    def __init__(self, audio_encoder:torch.nn.Module):
        super().__init__()
        self.audio_encoder = audio_encoder

    def forward(self, input_values, output_attentions=None, output_hidden_states=None, return_dict=None):
        self.audio_encoder.config.output_attentions = True
        output_attentions = output_attentions if output_attentions is not None else self.audio_encoder.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.audio_encoder.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.audio_encoder.config.use_return_dict

        hidden_states = self.audio_encoder.feature_extractor(input_values)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

class AudioModelSecond(torch.nn.Module):
    def __init__(self, audio_encoder:torch.nn.Module):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.audio_feature_map = model.audio_feature_map
        self.obj_vector = model.obj_vector

    def forward(self, hidden_states, one_hot, output_attentions=None, output_hidden_states=None, return_dict=True, attention_mask=None):
        obj_embedding = self.obj_vector(one_hot)
        hidden_states = self.audio_encoder.feature_projection(hidden_states)
        encoder_outputs = self.audio_encoder.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        hidden_states = self.audio_feature_map(hidden_states)

        return hidden_states, obj_embedding

template_file = os.path.join(args.dataset, args.template_path)
with open(template_file, 'rb') as fin:
    templates = pickle.load(fin,encoding='latin1')

train_subjects_list = [i for i in args.train_subjects.split(" ")]

one_hot_labels = np.eye(len(train_subjects_list))
iter = train_subjects_list.index(args.condition)
one_hot = one_hot_labels[iter]
one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
one_hot = torch.FloatTensor(one_hot).to(device=args.device)

temp = templates[args.subject]
            
template = temp.reshape((-1))
template = np.reshape(template,(-1,template.shape[0]))
template = torch.FloatTensor(template).to(device=args.device)

wav_path = args.wav_path
speech_array, _ = librosa.load(os.path.join(wav_path), sr=16000)
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-base-960h")
processor_output = processor(speech_array,sampling_rate=16000).input_values
audio_feature = np.squeeze(processor_output)
audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

#### test net ####
audio_encoder = FaceFormerAudioEncoder()
PPE = FaceFormerPPE()
decoder = FaceFormerDecoder()
dataset = 'vocaset'

mymodel1 = AudioModelFirst(audio_encoder.audio_encoder)
# 这里使用最大的输入来进行导出模型
# 如果您想测试模型的输入输出，可以将下面的行注释
audio_feature = torch.rand(1, 262144)

test_out = mymodel1(audio_feature)
hidden_states_out1 = linear_interpolation(test_out, 50, 30,output_len=None)

torch.onnx.export(
    mymodel1, 
    (audio_feature), 
    os.path.join(export_path, 'audio_encoder_1.onnx'), 
    input_names=['audio_feature'],output_names = ['hidden_states'], dynamic_axes={'audio_feature': {1: 'audio_length'}}, verbose=True)

mymodel2 = AudioModelSecond(audio_encoder.audio_encoder)
# random_tensor2 = torch.rand(1, 490, 512)
# one_hot = torch.rand(1, 8)
# hidden_states, obj_embedding = mymodel2(random_tensor2, one_hot_random_tensor)
hidden_states, obj_embedding = mymodel2(hidden_states_out1, one_hot)

torch.onnx.export(
    mymodel2, 
    (hidden_states_out1, one_hot), 
    os.path.join(export_path, 'audio_encoder_2.onnx'), 
    input_names=['hidden_states', 'one_hot'], dynamic_axes={'hidden_states': {1: 'frame_num'}}, verbose=True)

frame_num = hidden_states.shape[1]

for i in range(frame_num):
    if i==0:
        vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
        style_emb = vertice_emb
        vertice_input = PPE(style_emb)
    else:
        vertice_input = PPE(vertice_emb)
        if i == 489: #vertice_emb大小（[1,490,64]）
            torch.onnx.export(PPE, (vertice_emb), os.path.join(export_path, 'ppe.onnx'), input_names=['embedding'], dynamic_axes={'embedding': {1:'frame_num'}})
    memory_mask = enc_dec_mask(device, "vocaset", vertice_input.shape[1], hidden_states.shape[1])
    vertice_out, new_output = decoder(vertice_input, hidden_states, memory_mask)
    if i == 489:
        torch.onnx.export(decoder, (vertice_input, hidden_states, memory_mask), os.path.join(export_path, 'decoder.onnx'), input_names=['vertice_input', 'hidden_states', 'memory_mask'], dynamic_axes={'vertice_input': {1:'frame_num'}, 'hidden_states': {1:'frame_num'}, 'memory_mask': {0:'past_frame', 1:'all_frame'}})
    new_output = new_output + style_emb
    vertice_emb = torch.cat((vertice_emb, new_output), 1)
template = template.unsqueeze(1)
vertice_out = vertice_out + template
#### test net ####
prediction = vertice_out
prediction = prediction.squeeze() # (seq_len, V*3)
# print(prediction.shape)
# print(prediction)




