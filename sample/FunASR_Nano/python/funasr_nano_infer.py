#!/usr/bin/env python3
# ==============================================================================
# FunASR Nano TPU Inference — End-to-End Speech Recognition (all NN on TPU)
#
# Pipeline:
#   WAV (16kHz) → FBank+LFR (CPU, torchaudio) → SANM Encoder (TPU)
#               → Audio Adapter (TPU) → audio_embedding
#               → Qwen3-0.6B LLM prefill+decode (TPU, sail.EngineLLM) → Text
#
# Every neural network (encoder, adapter, LLM decoder) runs on TPU.
# Only the FBank/LFR feature extraction runs on CPU.
#
# Usage:
#   python3 funasr_nano_infer.py --input test.wav \
#       --encoder ../models/BM1684X/funasr_encoder_f16_1b.bmodel \
#       --adapter ../models/BM1684X/funasr_adapter_f16_1b.bmodel \
#       --llm ../models/BM1684X/qwen3_0.6b_llm_w4bf16_seq512_bm1684x.bmodel \
#       --config config/ --dev_id 0
#
# RTF (real-time factor) = total_inference_time / audio_duration.
# ==============================================================================

import argparse, logging, os, time
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def type_convert(sail_dtype):
    """sail.Dtype -> numpy dtype (bfloat16 has no numpy type, use uint16 bits)."""
    from sophon import sail
    if sail_dtype == sail.Dtype.BM_FLOAT32:
        return np.float32
    if sail_dtype == sail.Dtype.BM_FLOAT16:
        return np.float16
    if sail_dtype == sail.Dtype.BM_BFLOAT16:
        return np.uint16
    if sail_dtype == sail.Dtype.BM_INT32:
        return np.int32
    return np.float32


class SophonEngine:
    """SYSIO sail.Engine wrapper for the encoder/adapter bmodels."""
    def __init__(self, path, dev_id=0):
        from sophon import sail
        self.eng = sail.Engine(path, dev_id, sail.IOMode.SYSIO)
        self.graph = self.eng.get_graph_names()[0]
        self.in_names = self.eng.get_input_names(self.graph)
        self.out_names = self.eng.get_output_names(self.graph)

    def run(self, **feed):
        out = self.eng.process(self.graph, feed)
        return tuple(out[n] for n in self.out_names)


# ============================================================================
# Feature extraction (FunASR WavFrontend: kaldi-fbank + LFR(7,6), no CMVN)
# ============================================================================

def _apply_lfr(mat: torch.Tensor, lfr_m: int = 7, lfr_n: int = 6) -> torch.Tensor:
    """Low Frame Rate: concatenate lfr_m frames with stride lfr_n.

    Mirrors funasr.frontends.wav_frontend.apply_lfr exactly."""
    T = mat.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = mat[0:1].repeat((lfr_m - 1) // 2, 1)
    mat = torch.vstack((left_padding, mat))
    T = T + (lfr_m - 1) // 2
    feat_dim = mat.shape[-1]
    strides = (lfr_n * feat_dim, 1)
    sizes = (T_lfr, lfr_m * feat_dim)
    last_idx = (T - lfr_m) // lfr_n + 1
    num_padding = lfr_m - (T - last_idx * lfr_n)
    if num_padding > 0:
        num_padding = (2 * lfr_m - 2 * T + (T_lfr - 1 + last_idx) * lfr_n) / 2 * (T_lfr - last_idx)
        mat = torch.vstack([mat] + [mat[-1:]] * int(num_padding))
    out = mat.as_strided(sizes, strides)
    return out.clone().type(torch.float32)


def extract_features(wav_path: str, fs: int = 16000):
    """WAV -> FBank(80,25ms,10ms,hamming) -> LFR(7,6) -> float32 [T, 560]."""
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    waveform, sr = torchaudio.load(wav_path)          # [C, N]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # -> mono
    if sr != fs:
        waveform = torchaudio.functional.resample(waveform, sr, fs)
    waveform = waveform[0] * (1 << 15)                 # upscale_samples
    mat = kaldi.fbank(waveform.unsqueeze(0),
                      num_mel_bins=80, frame_length=25, frame_shift=10,
                      dither=0.0, energy_floor=0.0, window_type="hamming",
                      sample_frequency=fs, snip_edges=True)
    mat = _apply_lfr(mat, 7, 6)                          # [T, 560]
    return mat.numpy().astype(np.float32)


def compute_fake_token_len(T: int) -> int:
    """Number of LLM speech tokens from LFR length T (use_low_frame_rate path)."""
    olens = 1 + (T - 1) // 2
    olens = 1 + (olens - 1) // 2
    return (olens - 1) // 2 + 1


# ============================================================================
# Qwen3-0.6B LLM on TPU (sail.EngineLLM)
# ============================================================================

class FunASRLlm:
    """Qwen3-0.6B LLM decoder running entirely on TPU.

    Audio embeddings (adapter output) are spliced into the token embedding
    buffer at the speech-placeholder positions, exactly like a VLM splices
    image embeddings (see sample/Qwen2-VL)."""
    MASK_VALUE = -10000.0

    def __init__(self, bmodel_path, config_dir, dev_id=0):
        from sophon import sail
        self.dev_id = dev_id
        self.handle = sail.Handle(dev_id)
        self.net = sail.EngineLLM(bmodel_path, [dev_id])

        self.NUM_LAYERS = 28
        self.HIDDEN_SIZE = 1024
        self.KV_HEADS = 8
        self.HEAD_DIM = 128
        self.MAX_INPUT_LENGTH = self.net.get_input_shape("embedding", 0)[1]   # 256
        self.SEQLEN = self.net.get_input_shape("block_cache_0", 3)[1]         # 512

        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = [f"block_{i}" for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = [f"block_cache_{i}" for i in range(self.NUM_LAYERS)]

        # embedding / embedding_cache / lm_head tensors
        self.input_tensors = {}
        self.output_tensors = {}
        for n in (self.name_embed, self.name_embed_cache, self.name_lm):
            self.input_tensors[n] = self.net.create_max_input_tensors(n)
            self.output_tensors[n] = self.net.create_max_output_tensors(n)

        # block_0 inputs (shared hidden/pos/mask across all prefill blocks)
        self.input_tensors[self.name_blocks[0]] = \
            self.net.create_max_input_tensors(self.name_blocks[0])
        self.first_hidden_out = self._dev_tensor(self.name_blocks[0], 0, False)

        # block_cache_0 tensors
        self.next_hidden_in = self._dev_tensor(self.name_blocks_cache[0], 0, True)
        self.next_pos_in = self._dev_tensor(self.name_blocks_cache[0], 1, True)
        self.next_mask_in = self._dev_tensor(self.name_blocks_cache[0], 2, True)
        self.next_hidden_out = self._dev_tensor(self.name_blocks_cache[0], 0, False)
        self.present_k = self._dev_tensor(self.name_blocks_cache[0], 1, False)
        self.present_v = self._dev_tensor(self.name_blocks_cache[0], 2, False)

        # per-layer KV cache buffers (block_cache history_k/history_v inputs)
        self.past_k, self.past_v = [], []
        for i in range(self.NUM_LAYERS):
            self.past_k.append(self._dev_tensor(self.name_blocks_cache[0], 3, True))
            self.past_v.append(self._dev_tensor(self.name_blocks_cache[0], 4, True))
        self.kv_stride = self.KV_HEADS * self.HEAD_DIM  # per-token elements

        # tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config_dir, trust_remote_code=True)
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        self.token_length = 0
        self.token_pos_length = 0
        self.step = 0
        self.tokens = []

    def _dev_tensor(self, graph, idx, is_input):
        from sophon import sail
        if is_input:
            name = self.net.get_input_names(graph)[idx]
            shape = self.net.get_input_shape(graph, idx)
            dtype = self.net.get_input_dtype(graph, idx)
        else:
            name = self.net.get_output_names(graph)[idx]
            shape = self.net.get_output_shape(graph, idx)
            dtype = self.net.get_output_dtype(graph, idx)
        return sail.Tensor(self.handle, shape, dtype, False, True)

    # ---- prompt construction (mirrors FunASR fun_asr_nano data_load_speech) ----
    def build_prompt(self, fake_token_len):
        prefix = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                  "<|im_start|>user\n语音转写：")
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        pre_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
        suf_ids = self.tokenizer(suffix, add_special_tokens=False).input_ids
        # speech placeholder = token id 0 (its embedding is overwritten by audio_emb)
        ids = pre_ids + [0] * fake_token_len + suf_ids
        fbank_beg = len(pre_ids)
        return ids, fbank_beg

    def _bf16_view(self, arr_float):
        """float{32,16} numpy -> uint16 bit pattern of bfloat16."""
        t = torch.from_numpy(np.ascontiguousarray(arr_float)).to(torch.bfloat16)
        # torch<2.0 lacks torch.uint16; view as int16 (same 16-bit width), reinterpret in numpy
        return t.view(torch.int16).numpy().view(np.uint16)

    def forward_first(self, input_ids, fbank_beg, audio_emb):
        """Prefill. audio_emb: [fake_token_len, 1024] float (from adapter)."""
        token_len = len(input_ids)
        assert token_len <= self.MAX_INPUT_LENGTH, f"prompt {token_len} > {self.MAX_INPUT_LENGTH}"
        seq = self.MAX_INPUT_LENGTH

        # 1. embedding graph: input_ids -> embeddings (bfloat16, as uint16)
        ids_pad = np.zeros((1, seq), dtype=np.int32)
        ids_pad[0, :token_len] = input_ids
        self.input_tensors[self.name_embed][0].update_data(ids_pad)
        self.net.process(self.name_embed, self.input_tensors[self.name_embed],
                         self.output_tensors[self.name_embed])
        inputs_embeds = self.output_tensors[self.name_embed][0].asnumpy()   # [1, seq, H] uint16

        # 2. splice audio embeddings into the speech-placeholder slots
        fake = audio_emb.shape[0]
        audio_u16 = self._bf16_view(audio_emb)                              # [fake, H] uint16
        inputs_embeds[0, fbank_beg:fbank_beg + fake, :] = audio_u16

        # 3. position_ids [1, seq]: 0..token_len-1
        pos = np.zeros((1, seq), dtype=np.int32)
        pos[0, :token_len] = np.arange(token_len)

        # 4. causal mask [1, 1, seq, seq] bfloat16: 0 where j<=i<token_len else MASK_VALUE
        mask = np.ones((seq, seq), dtype=np.float32) * self.MASK_VALUE
        for i in range(token_len):
            for j in range(i + 1):
                mask[i, j] = 0.0
        mask_u16 = self._bf16_view(mask).reshape(1, 1, seq, seq)

        # 5. feed block_0 inputs and run all prefill blocks
        self.input_tensors[self.name_blocks[0]][0].update_data(inputs_embeds)
        self.input_tensors[self.name_blocks[0]][1].update_data(pos)
        self.input_tensors[self.name_blocks[0]][2].update_data(mask_u16)
        for i in range(self.NUM_LAYERS):
            block_out = {0: self.first_hidden_out,
                         1: self.past_k[i],
                         2: self.past_v[i]}
            self.net.process(self.name_blocks[i], self.input_tensors[self.name_blocks[0]], block_out)
            self.input_tensors[self.name_blocks[0]][0].sync_d2d(
                self.first_hidden_out, 0, 0, len(self.first_hidden_out))

        # 6. lm_head on the last real token (greedy argmax baked into lm_head)
        self.step = token_len
        self.token_pos_length = token_len
        self.input_tensors[self.name_lm][0].sync_d2d(
            self.first_hidden_out, (token_len - 1) * self.HIDDEN_SIZE, 0, self.HIDDEN_SIZE)
        self.net.process(self.name_lm, self.input_tensors[self.name_lm],
                         self.output_tensors[self.name_lm])
        self.last_id = int(self.output_tensors[self.name_lm][0].asnumpy().item())
        self.tokens = list(input_ids) + [self.last_id]
        return self.last_id

    def forward_next(self):
        # embedding_cache: last token -> embedding
        tok = np.array([[self.last_id]], dtype=np.int32)
        self.input_tensors[self.name_embed_cache][0].update_data(tok)
        self.net.process(self.name_embed_cache, self.input_tensors[self.name_embed_cache],
                         self.output_tensors[self.name_embed_cache])

        # attention_mask [1,1,1,SEQLEN+1]: mask future history positions
        mask = np.zeros(self.SEQLEN + 1, dtype=np.float32)
        for i in range(self.step, self.SEQLEN):
            mask[i] = self.MASK_VALUE
        mask_u16 = self._bf16_view(mask).reshape(1, 1, 1, self.SEQLEN + 1)
        pos_id = np.array([[self.token_pos_length]], dtype=np.int32)

        self.next_hidden_in.sync_d2d(self.output_tensors[self.name_embed_cache][0],
                                     0, 0, self.HIDDEN_SIZE)
        self.next_pos_in.update_data(pos_id)
        self.next_mask_in.update_data(mask_u16)

        block_out = {0: self.next_hidden_out, 1: self.present_k, 2: self.present_v}
        for i in range(self.NUM_LAYERS):
            block_in = {0: self.next_hidden_in, 1: self.next_pos_in, 2: self.next_mask_in,
                        3: self.past_k[i], 4: self.past_v[i]}
            self.net.process(self.name_blocks_cache[i], block_in, block_out)
            self.next_hidden_in.sync_d2d(self.next_hidden_out, 0, 0, self.HIDDEN_SIZE)
            self.past_k[i].sync_d2d(self.present_k, 0, self.step * self.kv_stride, self.kv_stride)
            self.past_v[i].sync_d2d(self.present_v, 0, self.step * self.kv_stride, self.kv_stride)

        self.input_tensors[self.name_lm][0].sync_d2d(self.next_hidden_out, 0, 0, self.HIDDEN_SIZE)
        self.net.process(self.name_lm, self.input_tensors[self.name_lm],
                         self.output_tensors[self.name_lm])
        self.last_id = int(self.output_tensors[self.name_lm][0].asnumpy().item())
        self.tokens.append(self.last_id)
        self.step += 1
        self.token_pos_length += 1
        return self.last_id

    def generate(self, audio_emb, max_new_tokens=128):
        """audio_emb: [fake_token_len, 1024] (adapter output, float)."""
        fake = audio_emb.shape[0]
        input_ids, fbank_beg = self.build_prompt(fake)
        t0 = time.time()
        token = self.forward_first(input_ids, fbank_beg, audio_emb)
        first_tok_time = time.time() - t0
        n_gen = 1
        while token != self.ID_IM_END and n_gen < max_new_tokens:
            token = self.forward_next()
            n_gen += 1
        decode_time = time.time() - t0 - first_tok_time
        # decode the generated tokens (exclude the prompt + the trailing <|im_end|>)
        text_ids = self.tokens[len(input_ids):]
        if text_ids and text_ids[-1] == self.ID_IM_END:
            text_ids = text_ids[:-1]
        text = self.tokenizer.decode(text_ids, skip_special_tokens=True)
        return text, first_tok_time, decode_time, n_gen


# ============================================================================
# Pipeline
# ============================================================================

class FunASRNanoInfer:
    ENC_GRID = 200
    ADAPT_GRID = 200

    def __init__(self, enc_eng, adapt_eng, llm, fs=16000):
        self.enc_eng = enc_eng
        self.adapt_eng = adapt_eng
        self.llm = llm
        self.fs = fs

    @classmethod
    def create(cls, encoder_path, adapter_path, llm_path, config_dir, dev_id=0):
        enc = SophonEngine(encoder_path, dev_id)
        adapt = SophonEngine(adapter_path, dev_id)
        llm = FunASRLlm(llm_path, config_dir, dev_id)
        logger.info(f"Encoder+Adapter+LLM loaded on TPU dev {dev_id}")
        return cls(enc, adapt, llm)

    def _encode(self, wav_path):
        """Frontend -> Encoder -> Adapter -> audio_emb [fake, 1024] float32."""
        feats = extract_features(wav_path, self.fs)       # [T, 560]
        T = feats.shape[0]

        # Encoder
        if T < self.ENC_GRID:
            feats_pad = np.pad(feats, ((0, self.ENC_GRID - T), (0, 0)), mode='constant')
        else:
            feats_pad = feats[:self.ENC_GRID, :]
            T = self.ENC_GRID
        enc_out = self.enc_eng.run(
            **{self.enc_eng.in_names[0]: feats_pad[None, :, :].astype(np.float32),
               self.enc_eng.in_names[1]: np.array([T], dtype=np.int32)})
        encoder_out = enc_out[0]                          # [1, GRID, 512]

        # Adapter
        if encoder_out.shape[1] < self.ADAPT_GRID:
            enc_pad = np.pad(encoder_out, ((0, 0), (0, self.ADAPT_GRID - encoder_out.shape[1]), (0, 0)),
                             mode='constant')
        else:
            enc_pad = encoder_out[:, :self.ADAPT_GRID, :]
        adapt_out = self.adapt_eng.run(
            **{self.adapt_eng.in_names[0]: enc_pad.astype(np.float32),
               self.adapt_eng.in_names[1]: np.array([T], dtype=np.int32)})
        adaptor_out = adapt_out[0][:, :T, :]              # [1, T, 1024]

        fake = compute_fake_token_len(T)
        audio_emb = adaptor_out[0, :fake, :].astype(np.float32)   # [fake, 1024]
        return audio_emb, T, fake

    def infer(self, wav_path, max_new_tokens=128):
        timings = {}
        # audio duration for RTF
        import torchaudio
        info = torchaudio.info(wav_path)
        dur = info.num_frames / info.sample_rate

        t0 = time.time()
        audio_emb, T, fake = self._encode(wav_path)
        t1 = time.time()
        timings['encode'] = (t1 - t0) * 1000
        logger.info(f"[1/2] Encode: T={T} fake_tokens={fake} emb={audio_emb.shape} "
                     f"({timings['encode']:.0f}ms)")

        t0 = time.time()
        text, ftl, dtime, n_gen = self.llm.generate(audio_emb, max_new_tokens)
        t1 = time.time()
        timings['llm'] = (t1 - t0) * 1000
        logger.info(f"[2/2] LLM: '{text[:50]}' ftl={ftl:.2f}s decode={dtime:.2f}s "
                     f"({n_gen} tok, {timings['llm']:.0f}ms)")

        timings['total'] = timings['encode'] + timings['llm']
        rtf = timings['total'] / 1000.0 / dur if dur > 0 else 0
        return {'text': text, 'timings': timings, 'rtf': rtf, 'duration': dur,
                'fake_tokens': fake}


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description='FunASR Nano full-TPU Inference')
    p.add_argument('--input', required=True)
    p.add_argument('--encoder', default='models/BM1684X/funasr_encoder_f16_1b.bmodel')
    p.add_argument('--adapter', default='models/BM1684X/funasr_adapter_f16_1b.bmodel')
    p.add_argument('--llm', default='models/BM1684X/qwen3_0.6b_llm_w4bf16_seq512_bm1684x.bmodel')
    p.add_argument('--config', default='config/')
    p.add_argument('--dev_id', type=int, default=0)
    p.add_argument('--max_new_tokens', type=int, default=128)
    args = p.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    enc = args.encoder if os.path.isabs(args.encoder) else os.path.join(root, args.encoder)
    adapt = args.adapter if os.path.isabs(args.adapter) else os.path.join(root, args.adapter)
    llm = args.llm if os.path.isabs(args.llm) else os.path.join(root, args.llm)
    cfg = args.config if os.path.isabs(args.config) else os.path.join(root, args.config)

    engine = FunASRNanoInfer.create(enc, adapt, llm, cfg, args.dev_id)
    result = engine.infer(args.input, args.max_new_tokens)
    t = result['timings']
    print(f"\n{'='*60}")
    print(f"Text:       {result['text']}")
    print(f"Timings:    encode={t['encode']:.0f}ms  llm={t['llm']:.0f}ms  total={t['total']:.0f}ms")
    print(f"Audio:      {result['duration']:.2f}s   fake_tokens={result['fake_tokens']}")
    print(f"RTF:        {result['rtf']:.3f}  (total_time / audio_duration)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
