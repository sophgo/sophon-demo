import torch
import re
import os

from vita.model.vita_tts.audioLLM import AudioLLM

from vita.model.vita_tts.encoder.cmvn import GlobalCMVN, load_cmvn
from vita.model.vita_tts.encoder.encoder import speechEncoder

def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        print('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        print('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')
    
    # load parm from checkpoint
    model.load_state_dict(checkpoint, strict=False)

    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    # get configs
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.safe_load(fin)
    return configs

def init_encoder_llm(configs):
    if configs['cmvn_file'] is not None:
        # read cmvn
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        # init cmvn layer
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    # init speech encoder
    encoder = speechEncoder(input_dim, global_cmvn=global_cmvn, **configs['encoder_conf'])
    # init audioLLM
    model = AudioLLM(encoder=encoder, **configs['model_conf'])

    return model
