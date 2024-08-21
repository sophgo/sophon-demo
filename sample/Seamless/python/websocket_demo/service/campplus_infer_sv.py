import os
import sys
import re
import pathlib
import numpy as np
import argparse
import torch
import torchaudio

import torchaudio.compliance.kaldi as Kaldi
class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
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

import importlib
def dynamic_import(import_path):
    module_name, obj_name = import_path.rsplit('.', 1)
    m = importlib.import_module(module_name)
    return getattr(m, obj_name)

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path


CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2NetV2_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
        'baseWidth': 26,
        'scale': 2,
        'expansion': 2,
    },
}

ERes2NetV2_w24s4ep4_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
        'baseWidth': 24,
        'scale': 4,
        'expansion': 4,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_base_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Base_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Large_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 64,
    },
}

ECAPA_CNCeleb = {
    'obj': 'speakerlab.models.ecapa_tdnn.ECAPA_TDNN.ECAPA_TDNN',
    'args': {
        'input_size': 80,
        'lin_neurons': 192,
        'channels': [1024, 1024, 1024, 1024, 3072],
    },
}

supports = {
    # CAM++ trained on 200k labeled speakers
    'iic/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    # ERes2Net trained on 200k labeled speakers
    'iic/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.5', 
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    # ERes2NetV2 trained on 200k labeled speakers
    'iic/speech_eres2netv2_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1', 
        'model': ERes2NetV2_COMMON,
        'model_pt': 'pretrained_eres2netv2.ckpt',
    },
    # ERes2NetV2_w24s4ep4 trained on 200k labeled speakers
    'iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1', 
        'model': ERes2NetV2_w24s4ep4_COMMON,
        'model_pt': 'pretrained_eres2netv2w24s4ep4.ckpt',
    },
    # ERes2Net_Base trained on 200k labeled speakers
    'iic/speech_eres2net_base_200k_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': ERes2Net_base_COMMON,
        'model_pt': 'pretrained_eres2net.pt',
    },
    # CAM++ trained on a large-scale Chinese-English corpus
    'iic/speech_campplus_sv_zh_en_16k-common_advanced': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_en_common.pt',
    },
    # CAM++ trained on VoxCeleb
    'iic/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
    },
    # ERes2Net trained on VoxCeleb
    'iic/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    # ERes2Net_Base trained on 3dspeaker
    'iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.1', 
        'model': ERes2Net_Base_3D_Speaker,
        'model_pt': 'eres2net_base_model.ckpt',
    },
    # ERes2Net_large trained on 3dspeaker
    'iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0', 
        'model': ERes2Net_Large_3D_Speaker,
        'model_pt': 'eres2net_large_model.ckpt',
    },
    # ECAPA-TDNN trained on CNCeleb
    'iic/speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k': {
        'revision': 'v1.0.0', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
    },
    # ECAPA-TDNN trained on 3dspeaker
    'iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
    },
    # ECAPA-TDNN trained on VoxCeleb
    'iic/speech_ecapa-tdnn_sv_en_voxceleb_16k': {
        'revision': 'v1.0.1', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa_tdnn.bin',
    },
}

class CampplusSV:
    def __init__(self, model_path, thresh=0.5, local_model_dir='pretrained'):
        """
        assert isinstance(model_id, str) and \
            is_official_hub_path(model_id), "Invalid modelscope model id."
        if model_id.startswith('damo/'):
            model_id = model_id.replace('damo/','iic/', 1)
        assert model_id in supports, "Model id not currently supported."
        save_dir = os.path.join(local_model_dir, model_id.split('/')[1])
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        conf = supports[model_id]
        # download models from modelscope according to model_id
        cache_dir = snapshot_download(
                    model_id,
                    revision=conf['revision'],
                    )
        cache_dir = pathlib.Path(cache_dir)

        self.embedding_dir = save_dir / 'embeddings'
        self.embedding_dir.mkdir(exist_ok=True, parents=True)

        # link
        download_files = ['examples', conf['model_pt']]
        for src in cache_dir.glob('*'):
            if re.search('|'.join(download_files), src.name):
                dst = save_dir / src.name
                try:
                    dst.unlink()
                except FileNotFoundError:
                    pass
                except IsADirectoryError:
                    pass
                try:
                    dst.symlink_to(src)
                except Exception:
                    pass
        """
        self.embedding_dir = pathlib.Path('embeddings')
        self.embedding_dir.mkdir(exist_ok=True, parents=True)
        pretrained_state = torch.load(model_path, map_location='cpu')

        self.device = torch.device('cpu')

        # load model
        model = {'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus', 'args': {'feat_dim': 80, 'embedding_size': 192}}
        self.embedding_model = dynamic_import(model['obj'])(**model['args'])
        self.embedding_model.load_state_dict(pretrained_state)
        self.embedding_model.to(self.device)
        self.embedding_model.eval()

        self.cache = []
        self.thresh = thresh

    def reset(self):
        self.cache = []

    def infer_sv(self, wav):
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
            array = torch.from_numpy(array)
            return array

        feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        def compute_embedding(wav, save=True):
            # load wav
            wav = load_bytes(wav)
            # compute feat
            feat = feature_extractor(wav).unsqueeze(0).to(self.device)
            # compute embedding
            with torch.no_grad():
                embedding = self.embedding_model(feat).detach().squeeze(0).cpu().numpy()
            
            if save:
                save_path = self.embedding_dir / (
                '%s.npy' % ("speaker_embeddings"))
                np.save(save_path, embedding)
                print(f'[INFO]: The extracted embedding from audio is saved to {save_path}.')
            
            return embedding

        # extract embeddings
        print(f'[INFO]: Extracting embeddings...')

        # input one wav file
        embedding = compute_embedding(wav, False)
        sp_id, _ = self.get_speaker_id(embedding)
        if sp_id == -1:
            sp_id = len(self.cache)
            self.cache.append(embedding)
        else:
            self.cache[sp_id] = (self.cache[sp_id] + embedding) / 2
        return self.cache[sp_id], sp_id
        
    def get_speaker_id(self, embedding):
        max_id, max_score = -1, 0
        for sp_id, sp in enumerate(self.cache):
            # compute similarity score
            print('[INFO]: Computing the similarity score...')
            similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            scores = similarity(torch.from_numpy(embedding).unsqueeze(0), torch.from_numpy(sp).unsqueeze(0)).item()
            if scores > self.thresh and scores > max_score:
                max_score = scores
                max_id = sp_id
        print('[INFO]: The similarity score between two input wavs is %.4f' % max_score)
        return max_id, max_score
    
