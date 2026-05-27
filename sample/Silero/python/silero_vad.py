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
Silero VAD (Voice Activity Detection) inference on BM1684X TPU using SAIL API.

Usage:
    python3 silero_vad.py --bmodel ../models/BM1684X/silero_vad_bm1684x_f16.bmodel \\
                          --input test.wav [--threshold 0.5]
"""

import time
import os
import json
import argparse
import logging
import numpy as np
import sophon.sail as sail

logging.basicConfig(level=logging.INFO)


class SileroVAD:
    """Silero VAD inference on BM1684X TPU using SAIL Engine."""

    SAMPLE_RATE = 16000
    NUM_SAMPLES = 512       # samples per frame
    CONTEXT_SIZE = 64       # context samples prepended to each frame
    STATE_DIM = 128         # LSTM hidden/cell state dimension

    def __init__(self, bmodel_path, dev_id=0):
        self.net = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(bmodel_path))

        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)
        self.handle = self.net.get_handle()

        # Cache shapes & dtypes
        self.input_shapes = {
            n: self.net.get_input_shape(self.graph_name, n)
            for n in self.input_names
        }
        self.output_shapes = {
            n: self.net.get_output_shape(self.graph_name, n)
            for n in self.output_names
        }
        logging.info("Inputs : {}".format(
            [(n, self.input_shapes[n]) for n in self.input_names]))
        logging.info("Outputs: {}".format(
            [(n, self.output_shapes[n]) for n in self.output_names]))

        # Identify the input/output tensor names by shape
        self._input_by_shape = self._index_by_shape(self.input_shapes)
        self._output_by_shape = self._index_by_shape(self.output_shapes)

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    @staticmethod
    def _index_by_shape(shapes):
        """Build a shape-keyed lookup. For 128-dim tensors, track order."""
        idx = {}
        seen_128 = 0
        for name, shape in shapes.items():
            key = tuple(shape)
            if key == (1, 128):
                idx[('1,128', seen_128)] = name
                seen_128 += 1
            else:
                idx[str(list(shape))] = name
        return idx

    def _forward(self, x, h, c):
        """Run one frame on TPU and return (prob, h_new, c_new) as numpy arrays."""
        # Build input dict using proper tensor names
        inputs = {}
        inputs[self._input_by_shape['[1, 576]']] = np.ascontiguousarray(x)
        inputs[self._input_by_shape[('1,128', 0)]] = np.ascontiguousarray(h)
        inputs[self._input_by_shape[('1,128', 1)]] = np.ascontiguousarray(c)

        t0 = time.time()
        outputs = self.net.process(self.graph_name, inputs)
        self.inference_time += time.time() - t0

        # Extract outputs by shape
        prob = outputs[self._output_by_shape['[1, 1]']]
        h_new = outputs[self._output_by_shape[('1,128', 0)]]
        c_new = outputs[self._output_by_shape[('1,128', 1)]]
        return prob, h_new, c_new

    def process_audio(self, audio, threshold=0.5,
                      min_speech_duration_ms=250,
                      min_silence_duration_ms=100,
                      speech_pad_ms=30):
        """
        Run VAD on audio and return speech timestamps.

        Args:
            audio: 1D numpy float32 array at 16kHz
            threshold: speech probability threshold (0.0~1.0)
            min_speech_duration_ms: discard speech segments shorter than this
            min_silence_duration_ms: wait for this much silence to end a segment
            speech_pad_ms: pad segment boundaries

        Returns:
            list of dicts: [{'start': sample, 'end': sample}, ...]
        """
        audio_len = len(audio)
        window = self.NUM_SAMPLES
        context = np.zeros(self.CONTEXT_SIZE, dtype=np.float32)
        h = np.zeros((1, self.STATE_DIM), dtype=np.float32)
        c = np.zeros((1, self.STATE_DIM), dtype=np.float32)

        speech_probs = []
        tp0 = time.time()

        for pos in range(0, audio_len, window):
            chunk = audio[pos: pos + window]
            if len(chunk) < window:
                chunk = np.pad(chunk, (0, window - len(chunk)))

            x = np.concatenate([context, chunk]).astype(np.float32).reshape(1, -1)
            prob, h, c = self._forward(x, h, c)
            speech_probs.append(float(prob.item()))
            context = x[0, -self.CONTEXT_SIZE:]

        self.preprocess_time += time.time() - tp0

        # --- Post-process: convert probabilities to timestamps ---
        tp0 = time.time()

        sr = self.SAMPLE_RATE
        min_speech_samples = sr * min_speech_duration_ms / 1000
        speech_pad_samples = sr * speech_pad_ms / 1000
        min_silence_samples = sr * min_silence_duration_ms / 1000
        neg_threshold = max(threshold - 0.15, 0.01)

        triggered = False
        speeches = []
        current_speech = {}
        temp_end = 0

        for i, prob in enumerate(speech_probs):
            cur_sample = window * i

            if prob >= threshold and temp_end:
                temp_end = 0

            if prob >= threshold and not triggered:
                triggered = True
                current_speech['start'] = cur_sample
                continue

            if prob < neg_threshold and triggered:
                if not temp_end:
                    temp_end = cur_sample
                if cur_sample - temp_end < min_silence_samples:
                    continue
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                temp_end = 0
                triggered = False

        if current_speech:
            current_speech['end'] = len(speech_probs) * window
            if (current_speech['end'] - current_speech.get('start', 0)) > min_speech_samples:
                speeches.append(current_speech)

        # Padding
        for i, seg in enumerate(speeches):
            seg['start'] = int(max(0, seg['start'] - speech_pad_samples))
            if i < len(speeches) - 1:
                gap = speeches[i + 1]['start'] - seg['end']
                if gap < 2 * speech_pad_samples:
                    seg['end'] += int(gap // 2)
                    speeches[i + 1]['start'] -= int(gap // 2)
                else:
                    seg['end'] = int(min(audio_len, seg['end'] + speech_pad_samples))
                    speeches[i + 1]['start'] = int(
                        max(0, speeches[i + 1]['start'] - speech_pad_samples))
            else:
                seg['end'] = int(min(audio_len, seg['end'] + speech_pad_samples))

        self.postprocess_time += time.time() - tp0
        return speeches, speech_probs


def read_audio(path, target_sr=16000):
    """Read WAV file, return float32 mono numpy array at target_sr."""
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
            audio = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0

    if sr != target_sr:
        try:
            import scipy.signal
            audio = scipy.signal.resample(audio, int(len(audio) * target_sr / sr))
        except ImportError:
            raise ImportError("Need scipy to resample. Install: pip install scipy soundfile")

    return audio.astype(np.float32)


def save_segments(audio, speeches, sample_rate, output_dir, basename):
    """Save detected speech segments as individual WAV files.

    Args:
        audio: 1D numpy float32 array at sample_rate.
        speeches: list of dicts [{'start': sample, 'end': sample}, ...].
        sample_rate: audio sample rate (Hz).
        output_dir: directory to write output files.
        basename: base filename prefix for output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for i, seg in enumerate(speeches):
        seg_audio = audio[seg['start']:seg['end']]
        out_path = os.path.join(output_dir,
                                "{}_seg{:02d}_{:.2f}s_{:.2f}s.wav".format(
                                    basename, i,
                                    seg['start'] / sample_rate,
                                    seg['end'] / sample_rate))
        try:
            import soundfile as sf
            sf.write(out_path, seg_audio, sample_rate)
        except ImportError:
            import wave
            import struct
            with wave.open(out_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(
                    (seg_audio * 32767.0).astype('<i2').tobytes())
        saved.append(out_path)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Silero VAD on BM1684X TPU")
    parser.add_argument('--bmodel', default='../models/BM1684X/silero_vad_bm1684x_f16.bmodel',
                        help='path to bmodel')
    parser.add_argument('--input', required=True, help='input WAV file')
    parser.add_argument('--dev_id', type=int, default=0, help='TPU device id')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='speech probability threshold (0.0~1.0)')
    parser.add_argument('--min_speech_duration_ms', type=int, default=250)
    parser.add_argument('--min_silence_duration_ms', type=int, default=100)
    parser.add_argument('--speech_pad_ms', type=int, default=30)
    parser.add_argument('--save_segments', action='store_true',
                        help='save detected speech segments as separate WAV files')
    args = parser.parse_args()

    # Check input
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    # Load model
    vad = SileroVAD(args.bmodel, args.dev_id)

    # Read audio
    t0 = time.time()
    audio = read_audio(args.input)
    duration = len(audio) / 16000.0
    logging.info("Loaded {} :: {:.2f}s (read: {:.1f}ms)".format(
        os.path.basename(args.input), duration, (time.time() - t0) * 1000))

    # Run VAD
    speeches, probs = vad.process_audio(
        audio,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
    )

    num_frames = len(probs)
    logging.info("Frames: {}, speech segments: {}".format(num_frames, len(speeches)))
    for i, seg in enumerate(speeches):
        s = seg['start'] / 16000.0
        e = seg['end'] / 16000.0
        logging.info("  seg {}: {:7.2f}s → {:7.2f}s ({:.2f}s)".format(i, s, e, e - s))

    # Save result
    os.makedirs("./results", exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.input))[0]
    result = {
        "audio_file": args.input,
        "duration_s": duration,
        "threshold": args.threshold,
        "num_frames": num_frames,
        "segments": [
            {"start_s": seg['start'] / 16000.0,
             "end_s": seg['end'] / 16000.0,
             "duration_s": (seg['end'] - seg['start']) / 16000.0}
            for seg in speeches
        ],
    }
    result_path = os.path.join("results", f"{basename}_vad.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    logging.info("Result saved to {}".format(result_path))

    # Save speech segments as audio files
    if args.save_segments:
        seg_dir = "results/segments"
        saved = save_segments(audio, speeches, 16000, seg_dir, basename)
        logging.info("Saved {} speech segments to {}/".format(len(saved), seg_dir))
        for p in saved:
            logging.info("  {}".format(p))

    # Timing
    logging.info("------------------ Inference Time Info ----------------------")
    n = max(num_frames, 1)
    logging.info("frames: {}".format(num_frames))
    logging.info("preprocess    (ms/frame): {:.3f}".format(vad.preprocess_time / n * 1000))
    logging.info("inference     (ms/frame): {:.3f}".format(vad.inference_time / n * 1000))
    logging.info("postprocess   (ms/frame): {:.3f}".format(vad.postprocess_time / n * 1000))
    logging.info("real_time_factor: {:.4f}".format(
        vad.inference_time / duration if duration > 0 else 0))


if __name__ == '__main__':
    main()