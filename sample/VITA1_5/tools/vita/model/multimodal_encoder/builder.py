import os

import yaml
import torch
from transformers.utils.hub import get_file_from_repo

from .clip.clip_encoder import CLIPVisionTower
from .eva_clip.eva_clip_encoder import EvaClipVisionTower
from .internvit.internvit_encoder import InternViTVisionTower
from .siglip.siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2
from .whale.init_model import init_model


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None)
    )
    use_s2 = getattr(vision_tower_cfg, "use_s2", False)

    if "sig" in vision_tower.lower():
        if use_s2:
            return SiglipVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "eva" in vision_tower.lower():
        if use_s2:
            raise ValueError(f"Currently not supporting S2 for EVA-CLIP")
        else:
            return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    elif "clip" in vision_tower.lower():
        if use_s2:
            raise ValueError(f"Currently not supporting S2 for CLIP")
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "internvit" in vision_tower.lower():
        if use_s2:
            raise ValueError(f"Currently not supporting S2 for InternViT")
        else:
            return InternViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")


def build_audio_encoder(audio_encoder_config, **kwargs):
    with open(get_file_from_repo(audio_encoder_config.mm_audio_encoder, "train.yaml"), "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    configs["cmvn_file"] = get_file_from_repo(audio_encoder_config.mm_audio_encoder, "global_cmvn")

    configs["model_conf"]["freeze_encoder"] = getattr(
        audio_encoder_config, "freeze_audio_encoder", True
    )
    configs["model_conf"]["freeze_adpter"] = getattr(
        audio_encoder_config, "freeze_audio_encoder_adapter", True
    )
    configs["model_conf"]["audio_prompt_finetune"] = getattr(
        audio_encoder_config, "audio_prompt_finetune", False
    )
    configs["model_conf"]["audio_prompt_num"] = getattr(
        audio_encoder_config, "audio_prompt_num", 0
    )

    audio_encoder = init_model(configs)

    checkpoint = torch.load(get_file_from_repo(audio_encoder_config.mm_audio_encoder, "final.pt"), map_location="cpu")
    model_dict = audio_encoder.state_dict()
    for key in model_dict.keys():
        if key in checkpoint.keys():
            if model_dict[key].shape == checkpoint[key].shape:
                model_dict[key] = checkpoint[key]
            else:
                print(
                    "Key {} has different shape, {} VS {}".format(
                        key, model_dict[key].shape, checkpoint[key].shape
                    )
                )
        else:
            print("Key {} has not in resume model".format(key))
    audio_encoder.load_state_dict(model_dict)

    return audio_encoder
