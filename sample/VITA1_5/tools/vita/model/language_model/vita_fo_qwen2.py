from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..vita_arch import VITAMetaForCausalLM, VITAMetaModel
from ...constants import IGNORE_INDEX
from .vita_qwen2 import custom_forward


Qwen2ForCausalLM.forward = custom_forward


class VITAFOQwen2Config(Qwen2Config):
    model_type = "vita-fo-Qwen2"


class VITAFOQwen2Model(VITAMetaModel, Qwen2Model):
    config_class = VITAFOQwen2Config

    def __init__(self, config: Qwen2Config):
        super(VITAFOQwen2Model, self).__init__(config)


class VITAFOQwen2ForCausalLM(Qwen2ForCausalLM, VITAMetaForCausalLM):
    config_class = VITAFOQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = VITAFOQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.predict_usr_state = 0#2
        if self.predict_usr_state:
            self.predictor_head = torch.nn.Linear(config.hidden_size, self.predict_usr_state + 1) # +1 for the dummy class
        else:
            self.predictor_head = None
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        audios: Optional[dict] = None,
        sf_masks: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, audios, sf_masks
            )
        if labels is not None:
            state_labels = labels
            labels = torch.where(labels>=0, labels, IGNORE_INDEX)
        output_hidden_states = True
        outputs =  super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        # state loss
        if self.predictor_head is not None:
            state_logits = self.predictor_head(outputs[2][-1]).view(-1, self.predict_usr_state+1) # +1 for the dummy class
            if labels is not None:
                loss = outputs[0]
                weight = torch.Tensor([1, 5, 1]).to(torch.bfloat16).to(inputs_embeds.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
                s_labels= torch.where(
                            state_labels < IGNORE_INDEX, 
                            IGNORE_INDEX-state_labels-1, 
                            IGNORE_INDEX).view(-1)
                #assert all(label in [0, 1, IGNORE_INDEX] for label in s_labels), "s_labels must contain only 0, 1, or -100"
                state_loss = loss_fct(state_logits, s_labels)
                loss = loss + state_loss
                outputs['loss'] = loss
        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        sf_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or audios is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                audios,
                sf_masks,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        audios = kwargs.pop("audios", None)
        sf_masks = kwargs.pop("sf_masks", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

        if images is not None:
            _inputs["images"] = images
        if audios is not None:
            _inputs["audios"] = audios
        if sf_masks is not None:
            _inputs["sf_masks"] = sf_masks
        return _inputs

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def process_images(self, images, model_cfg):
        vision_tower = self.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = self.expand2square(
                    image, tuple(int(x * 255) for x in image_processor.image_mean)
                )
                image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images


AutoConfig.register("vita-fo-Qwen2", VITAFOQwen2Config)
AutoModelForCausalLM.register(VITAFOQwen2Config, VITAFOQwen2ForCausalLM)




