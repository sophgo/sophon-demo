import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from vita.constants import AUDIO_TOKEN_INDEX, IGNORE_INDEX, IMAGE_TOKEN_INDEX

from .multimodal_encoder.builder import build_audio_encoder, build_vision_tower
from .multimodal_projector.builder import build_vision_projector
import numpy as np

class VITAMetaModel:
    def __init__(self, config):
        super(VITAMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(
                config, delay_load=False#not getattr(config, "continuous_training", False)
            )
            if getattr(config, "continuous_training", False):
                config.continuous_training = False
            self.mm_projector = build_vision_projector(config)

        if hasattr(config, "mm_audio_encoder"):
            self.audio_encoder = build_audio_encoder(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_audio_encoder(self):
        audio_encoder = getattr(self, "audio_encoder", None)
        return audio_encoder

    def initialize_vision_modules(self, model_args):
        vision_tower = model_args.vision_tower

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        else:
            vision_tower = self.vision_tower
            #vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type")
        self.config.mm_hidden_size = vision_tower.hidden_size

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))

    def initialize_audio_modules(self, model_args):
        audio_encoder = model_args.audio_encoder

        pretrain_audio_mlp_adapter = model_args.pretrain_audio_mlp_adapter

        setattr(self.config, "mm_audio_encoder", audio_encoder)

        audio_encoder = build_audio_encoder(self.config)
        self.audio_encoder = audio_encoder

        load_audio_ckpt_from_mllm = True
        if load_audio_ckpt_from_mllm:
            from safetensors.torch import load_file
            import os
            audio_weights = {}
            for file_name in os.listdir(model_args.model_name_or_path):
                if file_name.endswith('safetensors'):
                    audio_weights.update(
                        {k[20:]: v for k, v in load_file(os.path.join(model_args.model_name_or_path, file_name)).items() if
                            k.startswith('model.audio_encoder.')})
            self.audio_encoder.load_state_dict(audio_weights, strict=True) 

        #load_audio_ckpt = True
        #if self.get_audio_encoder() is None or load_audio_ckpt or model_args.audio_prompt_finetune:
        #    audio_encoder = build_audio_encoder(self.config)
        #    self.audio_encoder = audio_encoder

        #load_audio_prompt_weight = False #True
        #if load_audio_prompt_weight:
        #    from safetensors.torch import load_file
        #    import os
        #    audio_weights = {}
        #    for file_name in os.listdir(model_args.model_name_or_path):
        #        if file_name.endswith('safetensors'):
        #            audio_weights.update(
        #                {k[38:]: v for k, v in load_file(os.path.join(model_args.model_name_or_path, file_name)).items() if
        #                    k.startswith('model.audio_encoder.prompt_embeddings')})
        #    self.audio_encoder.prompt_embeddings.load_state_dict(audio_weights, strict=True)

        #checkpoint = torch.load(model_args.audio_encoder + "/final.pt", map_location="cpu")
        #model_dict = self.audio_encoder.state_dict()
        #for key in model_dict.keys():
        #    if key in checkpoint.keys():
        #        if model_dict[key].shape == checkpoint[key].shape:
        #            model_dict[key] = checkpoint[key]
        #        else:
        #            print(
        #                "Key {} has different shape, {} VS {}".format(
        #                    key, model_dict[key].shape, checkpoint[key].shape
        #                )
        #            )
        #    else:
        #        print("Key {} has not in resume model".format(key))
        #self.audio_encoder.load_state_dict(model_dict)

        if pretrain_audio_mlp_adapter is not None:
            audio_projector_weights = torch.load(pretrain_audio_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.audio_encoder.adpter.load_state_dict(get_w(audio_projector_weights, "audio_encoder.adpter"))


class VITAMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_audio_encoder(self):
        return self.get_model().get_audio_encoder()

    def pool_feats(self, x, out_size):
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0)
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        num_tokens = x.shape[2] * x.shape[3]  # Recalculate the number of tokens after pooling
        x = x.reshape(b, c, num_tokens).permute(0, 2, 1)
        if ndim == 2:
            x = x.squeeze(0)
        return x

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        #image_features = self.pool_feats(image_features)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_frameCat(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        assert len(image_features) % 5 == 0

        concatenated_features = []
        for i in range(0, len(image_features), 5):
            tensors_to_concat = [image_features[j] for j in range(i, i + 5)]
            concatenated_tensor = torch.cat(tensors_to_concat, dim=-1)
            concatenated_features.append(concatenated_tensor)
        concatenated_features = torch.stack(concatenated_features)
        image_features = concatenated_features

        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def slow_fast_pooling0(self, temp_img_feats):
        num_frame = len(temp_img_feats)
        if num_frame <= 30:
            slow_token_num = max([e for e in [256, 225, 196, 169] if e <= 5200/num_frame]) 
            fast_token_num = slow_token_num
        elif num_frame <= 45:
            slow_token_num = 169
            fast_token_num = 81
        elif num_frame <= 64:
            slow_token_num = 169
            fast_token_num = 49
        else:
            raise ValueError("The number of frames is too large!")
        
        if num_frame <= 30:
            num_slow = num_frame
        else:
            num_slow = int((5200 - fast_token_num * num_frame) / (slow_token_num - fast_token_num))
        num_fast = num_frame - num_slow
        slow_index = list(np.linspace(0, num_frame, num=num_slow, dtype=int))

        new_img_feats = []
        for i, feat in enumerate(temp_img_feats):
            if i in slow_index:
                sqrt_len = int(math.sqrt(slow_token_num))
            else:
                sqrt_len = int(math.sqrt(fast_token_num))
            if sqrt_len != 16:
                feat = self.pool_feats(feat, out_size=(sqrt_len, sqrt_len))
            new_img_feats.append(feat)

        return new_img_feats

    def slow_fast_pooling1(self, temp_img_feats):
        num_frame = len(temp_img_feats)
        if num_frame <= 28:
            slow_token_num = max([e for e in [256, 225, 196, 169, 144] if e <= 4096/num_frame]) 
            fast_token_num = slow_token_num
        elif num_frame <= 40:
            slow_token_num = 144
            fast_token_num = 81
        elif num_frame <= 64:
            slow_token_num = 144
            fast_token_num = 49
        else:
            raise ValueError("The number of frames is too large!")
        
        if num_frame <= 28:
            num_slow = num_frame
        else:
            num_slow = int((4096 - fast_token_num * num_frame) / (slow_token_num - fast_token_num))
        num_fast = num_frame - num_slow
        slow_index = list(np.linspace(0, num_frame, num=num_slow, dtype=int))

        new_img_feats = []
        for i, feat in enumerate(temp_img_feats):
            if i in slow_index:
                sqrt_len = int(math.sqrt(slow_token_num))
            else:
                sqrt_len = int(math.sqrt(fast_token_num))
            if sqrt_len != 16:
                feat = self.pool_feats(feat, out_size=(sqrt_len, sqrt_len))
            new_img_feats.append(feat)

        return new_img_feats

    def slow_fast_pooling(self, temp_img_feats):
        num_frame = len(temp_img_feats)
        slow_token_num = 144
        fast_token_num = 49
        
        slow_index = list(range(0, num_frame, 4))

        new_img_feats = []
        for i, feat in enumerate(temp_img_feats):
            if i in slow_index:
                sqrt_len = int(math.sqrt(slow_token_num))
            else:
                sqrt_len = int(math.sqrt(fast_token_num))
            if sqrt_len != 16:
                feat = self.pool_feats(feat, out_size=(sqrt_len, sqrt_len))
            new_img_feats.append(feat)

        return new_img_feats

    def slow_fast_pooling3(self, temp_img_feats):
        num_frame = len(temp_img_feats)
        slow_token_num = 144
        fast_token_num = 36
        
        slow_index = list(range(0, num_frame, 16))

        new_img_feats = []
        for i, feat in enumerate(temp_img_feats):
            if i in slow_index:
                sqrt_len = int(math.sqrt(slow_token_num))
            else:
                sqrt_len = int(math.sqrt(fast_token_num))
            if sqrt_len != 16:
                feat = self.pool_feats(feat, out_size=(sqrt_len, sqrt_len))
            new_img_feats.append(feat)

        return new_img_feats

    def slow_fast(self, image_features, sf_masks):
        new_image_features = []
        temp_img_feats = []  # 初始化 temp_img_feats 在循环外
        for i, img_feat in enumerate(image_features):
            if i == 0 or sf_masks[i] != sf_masks[i-1]:
                if temp_img_feats:  # 如果 temp_img_feats 不为空，则添加到 new_image_features
                    if sf_masks[i-1] > 0:
                        temp_img_feats = self.slow_fast_pooling(temp_img_feats)
                    new_image_features.append(temp_img_feats)
                temp_img_feats = [img_feat]  # 重新初始化 temp_img_feats
            else:
                temp_img_feats.append(img_feat)
        if temp_img_feats:  # 处理最后一个子列表
            if sf_masks[-1] > 0:
                temp_img_feats = self.slow_fast_pooling(temp_img_feats)
            new_image_features.append(temp_img_feats)
        
        output_features = []
        for e in new_image_features:
            output_features += e

        return output_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, audios, sf_masks, shared_v_pid_stride=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)

        image_features = [e for e in image_features]
        if sf_masks is not None:
            assert len(image_features) == len(sf_masks)
            image_features = self.slow_fast(image_features, sf_masks) 

        audio_encoder = self.get_audio_encoder()
        if audios is not None:
            audio_features = audio_encoder(audios["audios"], audios["lengths"])
            state_labels = audios.get("state_labels", None)
            lengths_for_llm = audios["lengths_for_llm"]
            if state_labels is not None:
                assert len(audio_features["inputs_embeds"]) == len(state_labels) == len(lengths_for_llm)
        else:
            audio_features, state_labels, lengths_for_llm = None, None, None        

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        v_start_end = []
        cur_image_idx = 0
        cur_audio_idx = 0
        assert (
            sum([(cur == IMAGE_TOKEN_INDEX).sum() for cur in input_ids])
            + sum([(IMAGE_TOKEN_INDEX not in cur) for cur in input_ids])
            == len(image_features)
        ), input_ids
        assert (
            sum([(cur == AUDIO_TOKEN_INDEX).sum() for cur in input_ids])
            + sum([(AUDIO_TOKEN_INDEX not in cur) for cur in input_ids])
            == audio_features["inputs_embeds"].shape[0]
        ), input_ids

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_audio_frames = (cur_input_ids == AUDIO_TOKEN_INDEX).sum()
            if num_images == 0 and num_audio_frames == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0], cur_audio_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                cur_audio_idx += 1
                continue

            image_audio_token_indices = (
                [-1]
                + torch.where(
                    (cur_input_ids == IMAGE_TOKEN_INDEX) | (cur_input_ids == AUDIO_TOKEN_INDEX)
                )[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim_noau = []
            cur_labels = labels[batch_idx]
            cur_labels_noim_noau = []
            for i in range(len(image_audio_token_indices) - 1):
                cur_input_ids_noim_noau.append(
                    cur_input_ids[
                        image_audio_token_indices[i] + 1 : image_audio_token_indices[i + 1]
                    ]
                )
                cur_labels_noim_noau.append(
                    cur_labels[image_audio_token_indices[i] + 1 : image_audio_token_indices[i + 1]]
                )

            split_sizes = [x.shape[0] for x in cur_labels_noim_noau]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim_noau))
            cur_input_embeds_no_im_no_au = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_v_start_end = []
            for i in range(num_images + num_audio_frames + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im_no_au[i])
                cur_new_labels.append(cur_labels_noim_noau[i])
                if i < num_images + num_audio_frames:
                    if cur_input_ids[image_audio_token_indices[i + 1]] == IMAGE_TOKEN_INDEX:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                        if shared_v_pid_stride:
                            start = sum([x.shape[0] for x in cur_new_labels[:-1]])
                            end = start + cur_new_labels[-1].shape[0]
                            cur_v_start_end.append((start, end))
                    elif cur_input_ids[image_audio_token_indices[i + 1]] == AUDIO_TOKEN_INDEX:
                        cur_lengths_for_llm = lengths_for_llm[cur_audio_idx]
                        cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                        if getattr(self.config, "audio_prompt_num", None):#self.config.audio_prompt_num:
                            cur_lengths_for_llm = cur_lengths_for_llm + self.config.audio_prompt_num
                        cur_audio_features = cur_audio_features[:cur_lengths_for_llm]
                        if state_labels is not None:
                            cur_state_label = state_labels[cur_audio_idx]
                        cur_audio_idx += 1
                        cur_new_input_embeds.append(cur_audio_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_audio_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                        if state_labels is not None:
                            cur_new_labels[-1][-1] = cur_state_label
                    else:
                        raise ValueError

            if num_images != 0 and num_audio_frames == 0:
                cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                cur_audio_idx += 1
                cur_new_input_embeds.append(cur_audio_features[0:0])
            elif num_images == 0 and num_audio_frames != 0:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features[0:0])
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

            if shared_v_pid_stride:
                cur_v_start_end = merge_consecutive_tuples(cur_v_start_end)
                v_start_end.append(cur_v_start_end)

        assert cur_image_idx == len(image_features)
        assert cur_audio_idx == audio_features["inputs_embeds"].shape[0]
        if state_labels is not None:
            assert cur_audio_idx == len(state_labels)
        if state_labels is not None:
            assert (
                sum([(cur == AUDIO_TOKEN_INDEX).sum() for cur in input_ids])
                == sum([(cur == -101).sum() for cur in new_labels]) + sum([(cur == -102).sum() for cur in new_labels])
            ), (input_ids, sum([(cur == AUDIO_TOKEN_INDEX).sum() for cur in input_ids]),  sum([(cur == -101).sum() for cur in new_labels]), sum([(cur == -102).sum() for cur in new_labels]), new_labels.shape)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    if shared_v_pid_stride is None:
                        position_ids[i, :cur_len] = torch.arange(
                            0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                        )
                    else:
                        cur_v_start_end = v_start_end[i]
                        cur_shared_position_ids = make_shared_position_ids(cur_v_start_end, cur_len, shared_v_pid_stride)
                        position_ids[i, :cur_len] = cur_shared_position_ids

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None and shared_v_pid_stride is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


def merge_consecutive_tuples(tuples_list):
    if not tuples_list:
        return []

    # 首先对列表按照起点索引进行排序
    sorted_tuples = sorted(tuples_list, key=lambda x: x[0])
    
    # 初始化合并后的列表
    merged_tuples = [sorted_tuples[0]]
    
    for current_start, current_end in sorted_tuples[1:]:
        last_merged_start, last_merged_end = merged_tuples[-1]
        if current_start <= last_merged_end:  # 如果当前元组的起点小于等于上一个合并元组的终点
            # 合并这两个元组
            new_start, new_end = merged_tuples[-1][0], max(last_merged_end, current_end)
            merged_tuples[-1] = (new_start, new_end)
        else:
            # 如果当前元组不连续，直接添加到合并后的列表中
            merged_tuples.append((current_start, current_end))
    
    return merged_tuples


def make_shared_position_ids(cur_v_start_end, cur_len, shared_v_pid_stride):
    position_ids = torch.tensor([1.0] * cur_len)

    for start, end in cur_v_start_end:
        position_ids[start:end] = 1/shared_v_pid_stride
        v_mod = (end - start) % shared_v_pid_stride
        if v_mod != 0:
            position_ids[end-v_mod:end] = 1 / v_mod
    position_ids = position_ids.cumsum(dim=0)
    position_ids = torch.ceil(position_ids).long() - 1

    return position_ids
