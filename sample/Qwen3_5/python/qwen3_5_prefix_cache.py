import time
import os
import argparse
import logging
import numpy as np
import torch

from qwen3_5 import Qwen3_5


class PrefixCacheQwen3_5(Qwen3_5):
    """Qwen3.5 with a reusable text-prefix cache.

    For a workload whose user text is fixed while the image changes each call,
    the prompt splits into:
      prefix : chat-template head up to (and including) <|vision_start|>
      image  : <|image_pad|>xN (varies with the image)
      tail   : <|vision_end|> + fixed user text + generation prompt

    The prefix tokens are embedded once and run through the blocks once; the
    resulting FA-layer KV entries and linear-layer conv/recurrent states are
    snapshotted. Each subsequent call restores the snapshot, runs the vision
    tower for the new image, and only prefills image + tail tokens. Works on
    top of any use_history_kv bmodel (needs the block_kv_<i> graphs, probed by
    Qwen3_5.__init__ via support_history).
    """

    def build_prefix_inputs(self, prefix_text):
        """Tokenize the prefix-only prompt (no image placeholders)."""
        # render with the fixed text BEFORE the image slot so the whole text
        # ends up in the cacheable prefix; a text-only rendering would not
        # contain the vision markers at all
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prefix_text},
                {"type": "image"},
            ],
        }]
        text = self.processor.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        # keep everything up to and including <|vision_start|> as the prefix;
        # the image pads and the tail are rebuilt per call
        vision_start = text.find("<|vision_start|>")
        if vision_start < 0:
            raise RuntimeError(
                "chat template has no <|vision_start|>; cannot split prefix")
        prefix = text[:vision_start + len("<|vision_start|>")]
        return self.tokenizer(prefix, add_special_tokens=False,
                              return_tensors="pt").input_ids.numpy()

    def snapshot_states(self):
        """Copy FA past-KV prefix region + linear conv/recurrent states.

        Uses per-layer lists indexed by layer id (pop-based drain in
        restore_states would be consumed by the first restore call, breaking
        reuse of the snapshot across multiple image calls).
        """
        n = self.prefix_kv_len
        snap = {
            "fa_key": {}, "fa_value": {},
            "lin_conv": {}, "lin_recurrent": {},
        }
        for i in range(self.num_layers):
            if self.is_FA(i):
                snap["fa_key"][i] = \
                    self.past_key_input[i].asnumpy()[:n].copy()
                snap["fa_value"][i] = \
                    self.past_value_input[i].asnumpy()[:n].copy()
            else:
                snap["lin_conv"][i] = self.past_key_input[i].asnumpy().copy()
                snap["lin_recurrent"][i] = \
                    self.past_value_input[i].asnumpy().copy()
        return snap

    def restore_states(self, snap):
        fa_n = self.prefix_kv_len
        for i in range(self.num_layers):
            if self.is_FA(i):
                k = self.past_key_input[i]
                v = self.past_value_input[i]
                k_np = k.asnumpy()
                k_np[:fa_n] = snap["fa_key"][i]
                k_np[fa_n:] = 0  # no stale KV from a previous longer request
                k.update_data(k_np)
                v_np = v.asnumpy()
                v_np[:fa_n] = snap["fa_value"][i]
                v_np[fa_n:] = 0
                v.update_data(v_np)
            else:
                c = self.past_key_input[i]
                c_np = c.asnumpy()
                c_np[:] = snap["lin_conv"][i]
                c.update_data(c_np)
                r = self.past_value_input[i]
                r_np = r.asnumpy()
                r_np[:] = snap["lin_recurrent"][i]
                r.update_data(r_np)

    def warmup_prefix(self, prefix_ids, prefix_pos_ids):
        """Embed + prefill the fixed text prefix once; snapshot the states."""
        self.forward_embed(prefix_ids)
        # text-only prefix: 3 x len positions 0..len-1
        position_ids = np.tile(
            np.arange(prefix_ids.shape[1], dtype=np.int32), 3)
        self.forward_first_with_kv(position_ids.reshape(3, -1))
        self.prefix_kv_len = self.history_length - 1  # lm_head slot not written yet
        self.prefix_snapshot = self.snapshot_states()
        self.prefix_pos = int(position_ids.max())
        self.prefix_tokens = prefix_ids.shape[1]

    def vit_process_image_shift(self, inputs, shift):
        """vit_process_image with image embeddings written `shift` slots
        earlier: in the cached path the rest tokens start at dev_buffer[0],
        not at their absolute position in the full prompt."""
        vit_token_list = torch.where(inputs.input_ids == self.ID_VISION_START)[1].tolist()
        pre_patches = 0
        for idx, vit_offset in enumerate(vit_token_list):
            grid_thw = inputs.image_grid_thw[idx].unsqueeze(0)
            num_patches = int(torch.prod(grid_thw))
            hidden_states = inputs.pixel_values[pre_patches:pre_patches + num_patches, :]
            position_ids = self.rot_pos(grid_thw)
            pos_ids, pos_weights = self.fast_pos_embed_interpolate(grid_thw.tolist())
            self.forward_vit(hidden_states.numpy(), position_ids.numpy(), pos_ids.numpy(),
                             pos_weights.numpy(), grid_thw.numpy(),
                             vit_offset + 1 - shift)
            pre_patches += num_patches

    def generate_with_image(self, image_path, max_tokens=50):
        """One image call: restore prefix, vit + image/tail prefill, decode."""
        self.restore_states(self.prefix_snapshot)
        # forward_first_with_kv computes old_kvlen = history_length - 1;
        # the prefix wrote prefix_kv_len KV entries, so enter with +1
        self.history_length = self.prefix_kv_len + 1
        self.max_posid = self.prefix_pos
        self.tokens = []

        messages = [{
            "role": "user",
            "content": [
                # MUST match build_prefix_inputs ordering (text before image)
                # so the full prompt literally starts with the prefix tokens
                {"type": "text", "text": self.input_str},
                {"type": "image", "image": image_path,
                 "min_pixels": 4 * 32 * 32,
                 "max_pixels": self.MAX_PIXELS},
            ],
        }]
        inputs = self.process(messages, "image")
        token_len = inputs.input_ids.numel()
        # prefix part is inside input_ids; embed only tokens after it
        full_ids = inputs.input_ids.numpy()
        assert token_len > self.prefix_tokens, "image added no tokens?"
        assert full_ids[0, self.prefix_tokens - 1] == self.ID_VISION_START, \
            "input_ids[prefix_tokens-1] is not <|vision_start|>; template mismatch"
        rest_ids = full_ids[:, self.prefix_tokens:]
        self.forward_embed(rest_ids)
        t_vit = time.time()
        # forward_embed placed the rest embeddings at dev_buffer[0 ...]; vit's
        # absolute write offset is the index of the first <|image_pad|>
        # (= prefix_tokens), so shift it down by exactly that to overwrite the
        # image-pad embedding slots within the rest region
        self.vit_process_image_shift(inputs, self.prefix_tokens)
        vit_time = time.time() - t_vit

        position_ids = self.get_rope_index(inputs.input_ids,
                                           inputs.image_grid_thw,
                                           self.ID_IMAGE_PAD)
        # decode positions continue after the last image/tail position
        self.max_posid = int(position_ids.numpy().max())
        t_pre = time.time()
        # strip prefix positions; image/tail positions continue from prefix.
        # get_rope_index returns [3, batch, seq]; flatten to [3, seq] first.
        rest_pos = position_ids.numpy().reshape(3, -1)[:, self.prefix_tokens:]
        token = self.forward_first_with_kv(rest_pos)
        prefill_time = time.time() - t_pre
        # forward_embed(rest_ids) set token_len to the rest length; report the
        # full prompt length (prefix + image + tail) for the user
        token_len_full = self.prefix_tokens + self.token_len

        tok_num = 0
        full_word_tokens = []
        text = ""
        t_dec = time.time()
        while token not in [self.ID_IM_END, self.ID_END] and \
                self.history_length < self.seq_len and tok_num < max_tokens:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens,
                                         skip_special_tokens=True)
            if "�" not in word:
                if len(full_word_tokens) == 1:
                    pre_word = word
                    word = self.tokenizer.decode([token, token],
                                                 skip_special_tokens=True)[len(pre_word):]
                text += word
                full_word_tokens = []
            self.max_posid += 1
            pos = np.array([self.max_posid] * 3, dtype=np.int32)
            token = self.forward_next(pos)
            tok_num += 1
        decode_time = time.time() - t_dec
        return {
            "text": text, "vit": vit_time, "prefill": prefill_time,
            "decode": decode_time, "tok_num": tok_num,
            "token_len": token_len_full,
        }


def main(args):
    model = PrefixCacheQwen3_5(args)
    if not model.support_history:
        raise RuntimeError(
            "prefix cache requires a use_history_kv bmodel "
            "(block_kv_<i> graphs not found)")

    model.input_str = args.question
    print(f"[prefix] tokenizing fixed text prefix ...")
    t0 = time.time()
    prefix_ids = model.build_prefix_inputs(args.question)
    prefix_pos = np.tile(np.arange(prefix_ids.shape[1], dtype=np.int32), 3)
    print(f"[prefix] prefix tokens: {prefix_ids.shape[1]}, "
          f"build+prefill one-time cost below")
    t1 = time.time()
    model.warmup_prefix(prefix_ids, prefix_pos)
    print(f"[prefix] warmup (one-time): {time.time() - t1:.3f} s "
          f"(embed+build_prefix_inputs={t1 - t0:.3f} s incl.)")

    images = args.images
    results = []
    for img in images:
        if not os.path.exists(img):
            print(f"skip missing {img}")
            continue
        t_start = time.time()
        r = model.generate_with_image(img, args.max_tokens)
        total = time.time() - t_start
        r["total"] = total
        results.append((img, r))
        print(f"\n===== {img} =====")
        print(r["text"])
        print(f"----- tokens_in={r['token_len']} (prefix {model.prefix_tokens} "
              f"+ image/tail {r['token_len'] - model.prefix_tokens}), "
              f"tokens_out={r['tok_num']}")
        print(f"FTL(cached): {r['vit'] + r['prefill']:.3f} s  "
              f"[vit {r['vit']:.3f} | prefill {r['prefill']:.3f}]")
        print(f"decode: {r['decode']:.3f} s  "
              f"TPS: {r['tok_num'] / r['decode']:.3f} token/s")

    if len(results) >= 2:
        avg = sum(r["vit"] + r["prefill"] for _, r in results[1:]) / (len(results) - 1)
        tps = sum(r["tok_num"] / r["decode"] for _, r in results[1:]) / (len(results) - 1)
        print(f"\n===== summary (excl. first call) =====")
        print(f"avg FTL(cached): {avg:.3f} s   avg TPS: {tps:.3f} token/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3.5 fixed-text prefix KV cache demo")
    # yapf: disable
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file (must be --use_history_kv build)')
    parser.add_argument('-c', '--config_path', type=str, default="config",
                        help='path to the processor config dir')
    parser.add_argument('-vr', '--video_ratio', type=float, default=0.25,
                        help='Set video ratio, default is 0.25')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    parser.add_argument('--question', type=str,
                        default='请描述图片中的内容',
                        help='fixed text question used for every image')
    parser.add_argument('--images', nargs='+', required=True,
                        help='image paths to query one by one')
    parser.add_argument('--max_tokens', type=int, default=50,
                        help='max new tokens per image')
    parser.add_argument('--do_sample', action='store_true',
                        help='enable sampling (default greedy)')
    parser.add_argument('-ll', '--log_level', type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help='log level')
    # yapf: enable
    args = parser.parse_args()
    main(args)
