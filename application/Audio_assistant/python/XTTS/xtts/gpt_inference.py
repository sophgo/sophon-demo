import os
import math
import time

# import onnx
import numpy as np
import torch
from torch import nn
# import torch.nn.functional as F
from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers import GPT2Config
import sophon.sail as sail


class GPT2InferenceModel(GPT2PreTrainedModel):
    """Override GPT2LMHeadModel to allow for prefix conditioning."""

    def __init__(self, pos_emb, embeddings, norm, linear, kv_cache, tpu_inference_config=None):
        super().__init__(GPT2Config())
        self.pos_embedding = pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache
        self.inference_time = 0
        self.gpt_inference_first = None
        self.gpt_inference_loop  = None
        self.gpt_inference_first_times = 0
        self.gpt_inference_loop_times = 0
        self.gpt_inference_first_profile = 0
        self.gpt_inference_loop_profile = 0

        start_time = time.time()
        os.environ["LOG_LEVEL"] = "-1"
        gpt_inference_first_bmodel_path        = tpu_inference_config["gpt_first_inference_path"]
        gpt_inference_loop_bmodel_path         = tpu_inference_config["gpt_loop_inference_path"]
        assert os.path.exists(gpt_inference_first_bmodel_path), f"{gpt_inference_first_bmodel_path} not found"
        assert os.path.exists(gpt_inference_loop_bmodel_path), f"{gpt_inference_loop_bmodel_path} not found"
        self.devid = tpu_inference_config["devid"]

        # load bmodel
        self.gpt_inference_first = sail.Engine(gpt_inference_first_bmodel_path, self.devid, sail.IOMode.DEVIO)
        self.handle = sail.Handle(0)
        self.gpt_first_graph_name = self.gpt_inference_first.get_graph_names()[0]
        self.gpt_first_input_names = self.gpt_inference_first.get_input_names(self.gpt_first_graph_name)
        # get output
        self.gpt_first_output_names = self.gpt_inference_first.get_output_names(self.gpt_first_graph_name)

        self.gpt_first_output_tensors = {}
        self.gpt_first_output_scales = {}
        self.gpt_first_output_shapes = []
        for output_idx, output_name in enumerate(self.gpt_first_output_names):
            output_shape = self.gpt_inference_first.get_output_shape(self.gpt_first_graph_name, output_name)
            output_dtype = self.gpt_inference_first.get_output_dtype(self.gpt_first_graph_name, output_name)
            output_scale = self.gpt_inference_first.get_output_scale(self.gpt_first_graph_name, output_name)
            if output_idx == 0:
                output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            else:
                output = sail.Tensor(self.handle, output_shape, output_dtype, False, True)
            self.gpt_first_output_tensors[output_name] = output
            self.gpt_first_output_scales[output_name] = output_scale
            self.gpt_first_output_shapes.append(output_shape)

        # load bmodel
        self.gpt_inference_loop = sail.Engine(gpt_inference_loop_bmodel_path, self.devid, sail.IOMode.DEVIO)
        self.gpt_loop_graph_name = self.gpt_inference_loop.get_graph_names()[0]
        self.gpt_loop_input_names = self.gpt_inference_loop.get_input_names(self.gpt_loop_graph_name)
        # get output
        self.gpt_loop_output_names = self.gpt_inference_loop.get_output_names(self.gpt_loop_graph_name)

        self.gpt_loop_output_tensors = {}
        self.gpt_loop_output_scales = {}
        self.gpt_loop_output_shapes = []
        for output_idx, output_name in enumerate(self.gpt_loop_output_names):
            output_shape = self.gpt_inference_loop.get_output_shape(self.gpt_loop_graph_name, output_name)
            output_dtype = self.gpt_inference_loop.get_output_dtype(self.gpt_loop_graph_name, output_name)
            output_scale = self.gpt_inference_loop.get_output_scale(self.gpt_loop_graph_name, output_name)
            if output_idx == 0:
                output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            else:
                output = sail.Tensor(self.handle, output_shape, output_dtype, False, True)
            self.gpt_loop_output_tensors[output_name] = output
            self.gpt_loop_output_scales[output_name] = output_scale
            self.gpt_loop_output_shapes.append(output_shape)
        self.layer_num = (len(self.gpt_loop_output_tensors) - 1) // 2

        self.init_time = time.time() - start_time
        print(f"\nTPU bmodel init time: {self.init_time}s")

    def store_prefix_emb(self, prefix_emb):
        self.cached_prefix_emb = prefix_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None

        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        start_time = time.time()
        assert self.cached_prefix_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # assert len(past_key_values) + len(input_ids) == attention_mask.shape[1]

        # Create embedding
        prefix_len = self.cached_prefix_emb.shape[1]
        if input_ids.shape[1] != 1:
            gen_inputs = input_ids[:, prefix_len:]
            gen_emb = self.embeddings(gen_inputs)
            gen_emb = gen_emb + self.pos_embedding(gen_emb)
            if self.cached_prefix_emb.shape[0] != gen_emb.shape[0]:
                prefix_emb = self.cached_prefix_emb.repeat_interleave(
                    gen_emb.shape[0] // self.cached_prefix_emb.shape[0], 0
                )
            else:
                prefix_emb = self.cached_prefix_emb.to(gen_emb.dtype)
            #　emb = torch.cat([prefix_emb, gen_emb], dim=1)
            emb = np.concatenate([prefix_emb.numpy(), gen_emb.numpy()], axis=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - (prefix_len + 1), attention_mask.device
            )
            emb = emb.numpy()
        
        # import pdb; pdb.set_trace()
        static_seq_len = 256
        #　attention_mask_template = torch.empty(static_seq_len, static_seq_len).fill_(-2000).triu_(1)
        attention_mask_template = np.triu(np.full((static_seq_len, static_seq_len), -2000.), 1).astype(np.float32)
        if input_ids.shape[1] != 1:
            real_seq_len = emb.shape[1]
            # emb_padding = F.pad(emb_t,(0, 0, static_seq_len - real_seq_len, 0))
            emb_padding = np.pad(emb, ((0, 0), (static_seq_len - real_seq_len, 0), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            #　attention_mask_template = torch.empty(static_seq_len, static_seq_len).fill_(-2000).triu_(1)
            attention_mask_template = np.triu(np.full((static_seq_len, static_seq_len), -2000), 1).astype(np.float32)
            # mask = F.pad(attention_mask_template_t[:real_seq_len, :real_seq_len], (static_seq_len - real_seq_len, 0, 0, 0), value=-2000)
            mask = np.pad(attention_mask_template[:real_seq_len, :real_seq_len], ((0, 0), (static_seq_len - real_seq_len, 0)), 'constant', constant_values=((-2000, -2000), (-2000, -2000)))
            # mask = F.pad(mask_t, (0, 0, static_seq_len - real_seq_len, 0), value=-2000)
            mask = np.pad(mask, ((static_seq_len - real_seq_len, 0), (0, 0)), 'constant', constant_values=((-2000, -2000), (-2000, -2000)))
            mask = mask.reshape(1, 1, *mask.shape)
            position_ids = None
            st = time.time()
            # mask = mask.numpy()
            # emb_padding = emb_padding.numpy()
            mask = sail.Tensor(self.handle, mask)
            mask.sync_s2d()
            emb_padding = sail.Tensor(self.handle, emb_padding)
            emb_padding.sync_s2d()
            bmodel_inputs = {self.gpt_first_input_names[0]: mask, self.gpt_first_input_names[1]: emb_padding}
            self.gpt_inference_first.process(self.gpt_first_graph_name, bmodel_inputs, self.gpt_first_output_tensors)
            self.gpt_first_output_tensors[self.gpt_first_output_names[0]].sync_d2s()

            transformer_outputs = (
                torch.from_numpy(self.gpt_first_output_tensors[self.gpt_first_output_names[0]].asnumpy()),
                tuple((self.gpt_first_output_tensors[self.gpt_first_output_names[2*i+1]], self.gpt_first_output_tensors[self.gpt_first_output_names[2*i+2]]) for i in range(self.layer_num)),
                None,
                None,
                None
            )
            self.gpt_inference_first_profile += time.time() - st
            self.gpt_inference_first_times += 1
            # import pdb; pdb.set_trace()
        else:
            real_seq_len = position_ids.item()
            position_ids = None
            if real_seq_len >= static_seq_len:
                mask = np.flip(attention_mask_template[-1:], 1)
            else:
                mask = np.flip(attention_mask_template[real_seq_len:real_seq_len+1], 1)
            mask = mask.reshape(1, 1, *mask.shape)
            st = time.time()
            bmodel_inputs = {}
            for i in range(len(past_key_values)):
                bmodel_inputs[self.gpt_loop_input_names[i * 2]] = past_key_values[i][0]
                bmodel_inputs[self.gpt_loop_input_names[i * 2 + 1]] = past_key_values[i][1]

            # mask = mask.numpy()
            # emb = emb.numpy()
            bmodel_inputs[self.gpt_loop_input_names[-2]] = sail.Tensor(self.handle, mask)
            bmodel_inputs[self.gpt_loop_input_names[-2]].sync_s2d()
            bmodel_inputs[self.gpt_loop_input_names[-1]] = sail.Tensor(self.handle, emb)
            bmodel_inputs[self.gpt_loop_input_names[-1]].sync_s2d()

            self.gpt_inference_loop.process(self.gpt_loop_graph_name, bmodel_inputs, self.gpt_loop_output_tensors)
            self.gpt_loop_output_tensors[self.gpt_loop_output_names[0]].sync_d2s()
            transformer_outputs = (
                torch.from_numpy(self.gpt_loop_output_tensors[self.gpt_loop_output_names[0]].asnumpy()),
                tuple((self.gpt_loop_output_tensors[self.gpt_loop_output_names[2*i+1]], self.gpt_loop_output_tensors[self.gpt_loop_output_names[2*i+2]]) for i in range(self.layer_num)),
                None,
                None,
                None
            )
            self.gpt_inference_loop_profile += time.time() - st
            self.gpt_inference_loop_times += 1
            # import pdb; pdb.set_trace()
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        self.inference_time += time.time() - start_time

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs[1],
            hidden_states=transformer_outputs[2],
            attentions=transformer_outputs[3],
            cross_attentions=transformer_outputs[4],
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
