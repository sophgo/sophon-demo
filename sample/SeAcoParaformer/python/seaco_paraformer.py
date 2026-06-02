#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# -*- coding: utf-8 -*-
"""
SeACoParaformer: Non-autoregressive ASR with hotword customization on BM1684X.

Reference:
    SeACo-Paraformer: A Non-Autoregressive ASR System with Flexible and
    Effective Hotword Customization Ability (https://arxiv.org/abs/2308.03266)

Inference pipeline:
    1. Preprocess (CPU):  audio -> fbank -> LFR -> CMVN -> [B, T, 560]
    2. Encoder   (TPU):   [speech, speech_len] -> enc_out, hidden, alphas, token_num
    3. CIF       (CPU):   cif(hidden, alphas) -> pre_acoustic_embeds, token_len
    4. Decoder   (TPU):   [enc_out, enc_len, embeds, token_len] -> logits, hidden
    5. Predictor (TPU):   [enc_out, enc_len] -> us_alphas, token_num
    6. Timestamp (CPU):   us_alphas -> us_peaks -> ts_prediction_lfr6
    7. Decode    (CPU):   argmax -> token_ids -> text

Usage:
    python3 seaco_paraformer.py --model_dir ../models/BM1684X --input test.wav
"""

import time
import os
import json
import argparse
import logging
import numpy as np
import sophon.sail as sail

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# CIF (Continuous Integrate-and-Fire) -- pure NumPy
# Ported from FunASR funasr/models/bicif_paraformer/cif_predictor.py
# ---------------------------------------------------------------------------

def cif(hidden, alphas, threshold=1.0):
    """
    CIF with hidden states.

    Args:
        hidden:    (B, T, D)  float32
        alphas:    (B, T)     float32
        threshold: float

    Returns:
        acoustic_embeds: (B, N, D)  float32   fired frame embeddings
        fires:           (B, T)     float32   accumulated fire values
    """
    B, T, D = hidden.shape
    integrate = np.zeros(B, dtype=np.float32)
    frame = np.zeros((B, D), dtype=np.float32)
    list_fires = []
    list_frames = []

    for t in range(T):
        alpha = alphas[:, t]
        distribution_completion = np.ones(B, dtype=np.float32) - integrate

        integrate += alpha
        list_fires.append(integrate.copy())

        fire_place = integrate >= threshold
        integrate = np.where(fire_place, integrate - threshold, integrate)
        cur = np.where(fire_place, distribution_completion, alpha)
        remains = alpha - cur

        frame += cur[:, np.newaxis] * hidden[:, t, :]
        list_frames.append(frame.copy())
        frame = np.where(
            fire_place[:, np.newaxis],
            remains[:, np.newaxis] * hidden[:, t, :],
            frame,
        )

    fires = np.stack(list_fires, axis=1)       # (B, T)
    frames = np.stack(list_frames, axis=1)      # (B, T, D)

    # Collect fired frames
    batch_embeds = []
    max_len = 0
    for b in range(B):
        idxs = np.nonzero(fires[b] >= threshold)[0]
        if len(idxs) == 0:
            batch_embeds.append(np.zeros((0, D), dtype=np.float32))
        else:
            batch_embeds.append(frames[b, idxs, :])
        max_len = max(max_len, len(idxs))

    acoustic_embeds = np.zeros((B, max_len, D), dtype=np.float32)
    for b in range(B):
        n = batch_embeds[b].shape[0]
        if n > 0:
            acoustic_embeds[b, :n, :] = batch_embeds[b]

    return acoustic_embeds, fires


def cif_wo_hidden(alphas, threshold=1.0):
    """CIF without hidden states -- peak detection only."""
    B, T = alphas.shape
    integrate = np.zeros(B, dtype=np.float32)
    list_fires = []
    for t in range(T):
        integrate += alphas[:, t]
        list_fires.append(integrate.copy())
        fire_place = integrate >= threshold
        integrate = np.where(fire_place, integrate - threshold, integrate)
    return np.stack(list_fires, axis=1)


# ---------------------------------------------------------------------------
# Timestamp prediction  (ported from FunASR utils/timestamp_tools.py)
# ---------------------------------------------------------------------------

def ts_prediction_lfr6(us_alphas, us_peaks, char_list,
                       vad_offset=0.0, force_time_shift=-1.5, upsample_rate=3):
    """
    Predict word-level timestamps from upsampled CIF alphas and peaks.

    Returns:
        timestamp_list: list[[start_ms, end_ms], ...]  per-token in ms
        new_char_list:  list[str]  tokens + optional <sil>
    """
    START_END_THRESHOLD = 5
    MAX_TOKEN_DURATION = 12
    # seconds per frame; multiply by 1000 to get ms
    TIME_RATE = 10.0 * 6 / upsample_rate   # = 20 ms per frame

    if len(us_alphas.shape) == 2:
        alphas, peaks = us_alphas[0], us_peaks[0]
    else:
        alphas, peaks = us_alphas, us_peaks

    if char_list and char_list[-1] == "</s>":
        char_list = char_list[:-1]

    fire_place = np.where(peaks >= 1.0 - 1e-4)[0] + force_time_shift

    # Length mismatch → renormalize
    if len(fire_place) != len(char_list) + 1:
        alphas = alphas / (alphas.sum() / (len(char_list) + 1))
        peaks = cif_wo_hidden(alphas[np.newaxis, :], threshold=1.0 - 1e-4)[0]
        fire_place = np.where(peaks >= 1.0 - 1e-4)[0] + force_time_shift

    num_frames = peaks.shape[0]
    timestamp_list = []
    new_char_list = []

    # Leading silence
    if fire_place[0] > START_END_THRESHOLD:
        timestamp_list.append([0.0, round(fire_place[0] * TIME_RATE)])
        new_char_list.append("<sil>")

    # Token timestamps
    for i in range(len(fire_place) - 1):
        new_char_list.append(char_list[i])
        if MAX_TOKEN_DURATION < 0 or fire_place[i + 1] - fire_place[i] <= MAX_TOKEN_DURATION:
            timestamp_list.append([round(fire_place[i] * TIME_RATE),
                                    round(fire_place[i + 1] * TIME_RATE)])
        else:
            _split = fire_place[i] + MAX_TOKEN_DURATION
            timestamp_list.append([round(fire_place[i] * TIME_RATE),
                                    round(_split * TIME_RATE)])
            timestamp_list.append([round(_split * TIME_RATE),
                                    round(fire_place[i + 1] * TIME_RATE)])
            new_char_list.append("<sil>")

    # Trailing silence
    if num_frames - fire_place[-1] > START_END_THRESHOLD:
        _end = (num_frames + fire_place[-1]) * 0.5
        if timestamp_list:
            timestamp_list[-1][1] = int(_end * TIME_RATE)
        timestamp_list.append([int(_end * TIME_RATE),
                                int(num_frames * TIME_RATE)])
        new_char_list.append("<sil>")
    else:
        if timestamp_list:
            timestamp_list[-1][1] = int(num_frames * TIME_RATE)

    # VAD offset (vad_offset is in ms)
    if vad_offset:
        for ts in timestamp_list:
            ts[0] += vad_offset
            ts[1] += vad_offset

    return timestamp_list, new_char_list


# ---------------------------------------------------------------------------
# SeACoParaformer
# ---------------------------------------------------------------------------

class SeacoParaformer:
    """SeACoParaformer ASR on BM1684X using sophon.sail."""

    SAMPLE_RATE = 16000
    N_MELS = 80
    FRAME_LENGTH_MS = 25
    FRAME_SHIFT_MS = 10
    LFR_M = 7
    LFR_N = 6
    FEAT_DIM = N_MELS * LFR_M   # 560
    SOS_ID = 1
    EOS_ID = 2
    BLANK_ID = 0

    def __init__(self, model_dir, dev_id=0):
        self.model_dir = model_dir
        self.dev_id = dev_id

        # ------------------------- bmodels -------------------------
        self.encoder_net = sail.Engine(
            os.path.join(model_dir, "encoder_fp32_10b.bmodel"),
            dev_id, sail.IOMode.SYSIO)
        self.decoder_net = sail.Engine(
            os.path.join(model_dir, "decoder_fp32_10b.bmodel"),
            dev_id, sail.IOMode.SYSIO)
        self.predictor_net = sail.Engine(
            os.path.join(model_dir, "predictor_fp32_10b.bmodel"),
            dev_id, sail.IOMode.SYSIO)

        # Graph names
        self.enc_graph = self.encoder_net.get_graph_names()[0]
        self.dec_graph = self.decoder_net.get_graph_names()[0]
        self.pred_graph = self.predictor_net.get_graph_names()[0]

        # I/O names
        self.enc_in = self.encoder_net.get_input_names(self.enc_graph)
        self.enc_out = self.encoder_net.get_output_names(self.enc_graph)
        self.dec_in = self.decoder_net.get_input_names(self.dec_graph)
        self.dec_out = self.decoder_net.get_output_names(self.dec_graph)
        self.pred_in = self.predictor_net.get_input_names(self.pred_graph)
        self.pred_out = self.predictor_net.get_output_names(self.pred_graph)

        logging.info("Encoder graph: %s", self.enc_graph)
        logging.info("  inputs:  %s", [(n, self.encoder_net.get_input_shape(self.enc_graph, n))
                                        for n in self.enc_in])
        logging.info("  outputs: %s", [(n, self.encoder_net.get_output_shape(self.enc_graph, n))
                                        for n in self.enc_out])
        logging.info("Decoder graph: %s", self.dec_graph)
        logging.info("  inputs:  %s", [(n, self.decoder_net.get_input_shape(self.dec_graph, n))
                                        for n in self.dec_in])
        logging.info("Predictor graph: %s", self.pred_graph)
        logging.info("  inputs:  %s", [(n, self.predictor_net.get_input_shape(self.pred_graph, n))
                                        for n in self.pred_in])

        # ------------------------- tokenizer -------------------------
        with open(os.path.join(model_dir, "tokens.json"), "r", encoding="utf-8") as f:
            self.tokens = json.load(f)
        logging.info("Vocabulary size: %d", len(self.tokens))

        # ------------------------- CMVN -------------------------
        cmvn_path = os.path.join(model_dir, "am.mvn")
        if os.path.exists(cmvn_path):
            self.cmvn_means, self.cmvn_vars = self._load_cmvn(cmvn_path)
        else:
            self.cmvn_means = self.cmvn_vars = None
            logging.warning("CMVN not found: %s", cmvn_path)

        # Timing
        self.t_pre = 0.0
        self.t_enc = 0.0
        self.t_cif = 0.0
        self.t_dec = 0.0
        self.t_pred = 0.0
        self.t_tok = 0.0

    # -------------------------------------------------------------------
    # CMVN
    # -------------------------------------------------------------------

    @staticmethod
    def _load_cmvn(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        means_l, vars_l = [], []
        for i, line in enumerate(lines):
            items = line.split()
            if items[0] == "<AddShift>":
                ni = lines[i + 1].split()
                if ni[0] == "<LearnRateCoef>":
                    means_l = list(ni[3:len(ni) - 1])
            elif items[0] == "<Rescale>":
                ni = lines[i + 1].split()
                if ni[0] == "<LearnRateCoef>":
                    vars_l = list(ni[3:len(ni) - 1])
        return np.array(means_l, dtype=np.float32), np.array(vars_l, dtype=np.float32)

    # -------------------------------------------------------------------
    # Preprocessing  (fbank + LFR + CMVN)
    # -------------------------------------------------------------------

    def preprocess(self, audio):
        """
        audio: 1D float32 numpy array, 16 kHz mono
        Returns: speech (1, T, 560), speech_len (int)
        """
        t0 = time.time()

        import torch
        import torchaudio.compliance.kaldi as kaldi

        waveform = torch.from_numpy(audio).float().unsqueeze(0) * (1 << 15)

        fbank = kaldi.fbank(
            waveform,
            num_mel_bins=self.N_MELS,
            frame_length=self.FRAME_LENGTH_MS,
            frame_shift=self.FRAME_SHIFT_MS,
            dither=0.0,
            energy_floor=0.0,
            window_type="hamming",
            sample_frequency=self.SAMPLE_RATE,
        ).numpy()                                                      # (T_fbank, 80)

        lfr = self._apply_lfr(fbank, self.LFR_M, self.LFR_N)           # (T_lfr, 560)

        if self.cmvn_means is not None:
            d = lfr.shape[1]
            lfr = lfr + self.cmvn_means[:d]
            lfr = lfr * self.cmvn_vars[:d]

        speech_len = lfr.shape[0]
        speech = np.zeros((1, speech_len, self.FEAT_DIM), dtype=np.float32)
        speech[0, :speech_len, :] = lfr

        self.t_pre += time.time() - t0
        return speech, speech_len

    @staticmethod
    def _apply_lfr(inputs, lfr_m, lfr_n):
        """Low Frame Rate: stack consecutive frames (FunASR wav_frontend)."""
        import torch
        inputs = torch.from_numpy(inputs)
        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))

        left_pad = inputs[0].repeat((lfr_m - 1) // 2, 1)
        inputs = torch.vstack((left_pad, inputs))
        T = T + (lfr_m - 1) // 2
        D = inputs.shape[-1]

        strides = (lfr_n * D, 1)
        sizes = (T_lfr, lfr_m * D)

        last_idx = (T - lfr_m) // lfr_n + 1
        num_pad = lfr_m - (T - last_idx * lfr_n)
        if num_pad > 0:
            num_pad = (2 * lfr_m - 2 * T + (T_lfr - 1 + last_idx) * lfr_n) / 2 * (T_lfr - last_idx)
            inputs = torch.vstack([inputs] + [inputs[-1:]] * int(num_pad))

        return inputs.as_strided(sizes, strides).clone().type(torch.float32).numpy()

    # -------------------------------------------------------------------
    # TPU helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _build_inputs(names, arrays, predicates=None):
        """Build {name: np.ascontiguousarray(arr)} dict.

        If predicates is given, each array is matched to the first name whose
        predicate returns True.  Otherwise arrays are matched 1:1 by position.
        """
        if predicates is None:
            return {n: np.ascontiguousarray(a) for n, a in zip(names, arrays)}
        used = [False] * len(arrays)
        result = {}
        for name in names:
            for i, (pred, arr) in enumerate(zip(predicates, arrays)):
                if used[i]:
                    continue
                if pred(arr):
                    result[name] = np.ascontiguousarray(arr)
                    used[i] = True
                    break
        return result

    @staticmethod
    def _extract(outputs, predicate, skip=None):
        """Extract first output tensor matching predicate."""
        for name, arr in outputs.items():
            if skip is not None and arr is skip:
                continue
            a = arr.copy()
            if predicate(a):
                return a
        return None

    # -------------------------------------------------------------------
    # Encoder (TPU)
    # -------------------------------------------------------------------

    def _encoder_forward(self, speech, speech_len):
        t0 = time.time()

        inp = self._build_inputs(
            self.enc_in,
            [speech, np.array([speech_len], dtype=np.int32)],
            predicates=[lambda a: a.ndim == 3, lambda a: a.ndim == 1],
        )
        out = self.encoder_net.process(self.enc_graph, inp)
        self.t_enc += time.time() - t0

        # Expected shapes:
        #   enc_out:  (1, T,   512)  -- LayerNormalization
        #   hidden:   (1, T+1, 512)  -- Concat  (tail padding)
        #   alphas:   (1, T+1)       -- Add
        #   token_num:(1,)           -- Floor
        enc_out = self._extract(out, lambda a: a.ndim == 3 and a.shape[1] == speech_len)
        hidden = self._extract(out, lambda a: a.ndim == 3 and a.shape[1] == speech_len + 1)
        alphas = self._extract(out, lambda a: a.ndim == 2 and a.shape[1] == speech_len + 1)
        token_num = self._extract(out, lambda a: a.ndim == 1)

        return enc_out, hidden, alphas, token_num

    # -------------------------------------------------------------------
    # Decoder (TPU)
    # -------------------------------------------------------------------

    def _decoder_forward(self, encoder_out, encoder_out_lens,
                         pre_embeds, pre_token_len):
        t0 = time.time()

        inp = self._build_inputs(
            self.dec_in,
            [encoder_out,
             encoder_out_lens.astype(np.int32),
             pre_embeds,
             pre_token_len.astype(np.int32)],
        )
        out = self.decoder_net.process(self.dec_graph, inp)
        self.t_dec += time.time() - t0

        # Output 0: logits (1, N, vocab_size)   Output 1: hidden (1, N, 512)
        logits = self._extract(out, lambda a: a.ndim == 3 and a.shape[-1] > 512)
        dec_hidden = self._extract(out, lambda a: a.ndim == 3 and a.shape[-1] == 512)
        return logits, dec_hidden

    # -------------------------------------------------------------------
    # Predictor V3 (TPU)
    # -------------------------------------------------------------------

    def _predictor_forward(self, encoder_out, encoder_out_lens):
        """Returns us_alphas (1, T_up) and token_num (1,)."""
        t0 = time.time()

        inp = self._build_inputs(
            self.pred_in,
            [encoder_out,
             encoder_out_lens.astype(np.int32)],
        )
        out = self.predictor_net.process(self.pred_graph, inp)
        self.t_pred += time.time() - t0

        # Output 0: us_alphas (1, T_up)    Output 1: token_num (1,)
        us_alphas = self._extract(out, lambda a: a.ndim == 2)
        pred_token_num = self._extract(out, lambda a: a.ndim == 1, skip=us_alphas)
        return us_alphas, pred_token_num

    # -------------------------------------------------------------------
    # Full inference
    # -------------------------------------------------------------------

    def infer(self, audio_or_path):
        """
        Args:
            audio_or_path:  float32 numpy array (1D, 16 kHz) or path to WAV.

        Returns:
            dict: {"text", "tokens", "token_ids", "sentence_info"}
        """
        # 1. Load audio
        if isinstance(audio_or_path, str):
            audio = read_audio(audio_or_path)
        else:
            audio = audio_or_path
        logging.info("Audio: %.2f s  (%d samples)", len(audio) / self.SAMPLE_RATE, len(audio))

        # 2. Preprocess
        speech, speech_len = self.preprocess(audio)
        logging.info("Features: %d frames  shape=%s", speech_len, speech.shape)

        # 3. Encoder (TPU)
        enc_out, hidden, alphas, token_num = self._encoder_forward(speech, speech_len)
        logging.info("Encoder: enc_out=%s hidden=%s alphas=%s token_num=%s",
                     enc_out.shape, hidden.shape, alphas.shape, token_num.shape)

        # 4. CIF (CPU)
        t0 = time.time()
        pre_embeds, _ = cif(hidden, alphas, threshold=1.0)
        token_num_int = int(np.max(token_num).item())
        pre_embeds = pre_embeds[:, :token_num_int, :]                    # (1, N, 512)
        pre_token_len = np.round(token_num).astype(np.int64)             # (1,)
        self.t_cif += time.time() - t0
        logging.info("CIF: embeds=%s token_len=%s", pre_embeds.shape, pre_token_len)

        if pre_embeds.shape[1] == 0:
            return {"text": "", "tokens": [], "token_ids": [], "sentence_info": []}

        # 5. Decoder (TPU)
        enc_lens = np.array([enc_out.shape[1]], dtype=np.int64)
        logits, _ = self._decoder_forward(enc_out, enc_lens, pre_embeds, pre_token_len)
        logging.info("Decoder: logits=%s", logits.shape)

        # 6. Predictor V3 (TPU) → timestamps
        alphas2, pred_token_num = self._predictor_forward(enc_out, enc_lens)

        # 7. Greedy decode (CPU)
        t0 = time.time()
        N = pre_token_len[0].item()
        token_ids = np.argmax(logits[0, :N, :], axis=-1).tolist()
        token_ids = [t for t in token_ids
                     if t != self.SOS_ID and t != self.EOS_ID and t != self.BLANK_ID]
        tokens = [self.tokens[tid] if tid < len(self.tokens) else "<unk>"
                  for tid in token_ids]
        text = "".join(tokens).replace("@@", "").replace(" ", "")
        self.t_tok += time.time() - t0

        # 8. Timestamps
        sentence_info = []
        if alphas2 is not None and pred_token_num is not None:
            # Normalize us_alphas sum to match pre_token_length
            us_alphas = (alphas2 *
                         (pre_token_len / pred_token_num)[:, np.newaxis]
                         .repeat(alphas2.shape[1], axis=1))           # (1, T_up)
            us_peaks = cif_wo_hidden(us_alphas, threshold=1.0 - 1e-4) # (1, T_up)

            enc_len = enc_out.shape[1]
            ts_list, new_char_list = ts_prediction_lfr6(
                us_alphas[0, :enc_len * 3],
                us_peaks[0, :enc_len * 3],
                list(tokens),
                vad_offset=0.0,
            )
            for i, (start_ms, end_ms) in enumerate(ts_list):
                tok = new_char_list[i] if i < len(new_char_list) else ""
                sentence_info.append({
                    "start": int(start_ms),
                    "end": int(end_ms),
                    "text": tok,
                })

        return {"text": text, "tokens": tokens, "token_ids": token_ids,
                "sentence_info": sentence_info}

    # -------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------

    def print_timing(self, audio_dur_s):
        logging.info("------------------ Inference Time ----------------------")
        logging.info("  preprocess:  %.3f s", self.t_pre)
        logging.info("  encoder:     %.3f s", self.t_enc)
        logging.info("  cif:         %.3f s", self.t_cif)
        logging.info("  decoder:     %.3f s", self.t_dec)
        logging.info("  predictor:   %.3f s", self.t_pred)
        logging.info("  decode:      %.3f s", self.t_tok)
        total = self.t_pre + self.t_enc + self.t_cif + self.t_dec + self.t_pred + self.t_tok
        logging.info("  total:       %.3f s", total)
        if audio_dur_s > 0:
            logging.info("  RTF:         %.4f", total / audio_dur_s)


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def read_audio(path, target_sr=16000):
    """Read WAV file -> float32 mono numpy array at target_sr."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except ImportError:
        import wave
        with wave.open(path, 'rb') as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            audio = (np.frombuffer(wf.readframes(n), dtype=np.int16)
                     .astype(np.float32) / 32768.0)

    if sr != target_sr:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * target_sr / sr))

    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SeACoParaformer ASR on BM1684X")
    parser.add_argument('--model_dir', default='../models/BM1684X',
                        help='Directory with bmodel files + config')
    parser.add_argument('--input', required=True, help='Input WAV file (16kHz mono)')
    parser.add_argument('--dev_id', type=int, default=0, help='TPU device id')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    logging.info("Loading SeACoParaformer from %s", args.model_dir)
    t0 = time.time()
    model = SeacoParaformer(args.model_dir, args.dev_id)
    logging.info("Loaded in %.1f s", time.time() - t0)

    audio = read_audio(args.input)
    audio_dur = len(audio) / 16000.0

    t0 = time.time()
    result = model.infer(audio)
    wall = time.time() - t0

    logging.info("============================================================")
    logging.info("TEXT: %s", result["text"])
    logging.info("============================================================")

    if result.get("sentence_info"):
        for si in result["sentence_info"]:
            logging.info("  [%7d][%7d]  %s", si["start"], si["end"], si["text"])

    model.print_timing(audio_dur)

    # Save result
    os.makedirs("./results", exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.input))[0]
    result_json = {
        "audio_file": os.path.abspath(args.input),
        "duration_s": audio_dur,
        "text": result["text"],
        "tokens": result["tokens"],
        "sentence_info": result.get("sentence_info", []),
        "wall_time_s": wall,
    }
    out_path = os.path.join("results", f"{basename}_asr.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=4, ensure_ascii=False)
    logging.info("Result saved -> %s", out_path)


if __name__ == '__main__':
    main()