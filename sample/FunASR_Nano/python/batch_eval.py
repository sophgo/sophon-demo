#!/usr/bin/env python3
# ==============================================================================
# FunASR Nano — Batch WER evaluation on aishell_S0764
#
# Mirrors the WeNet/Whisper precision-testing approach:
#   1. Run full-TPU inference on every wav in the test set.
#   2. Write "<utt_id> <hypothesis>" lines to result.txt.
#   3. Compute CER with tools/eval_aishell.py --char=1 against ground_truth.txt.
#
# Usage:
#   python3 batch_eval.py --dataset ../datasets/aishell_S0764 \
#       --encoder ../models/BM1684X/funasr_encoder_f16_1b.bmodel \
#       --adapter ../models/BM1684X/funasr_adapter_f16_1b.bmodel \
#       --llm ../models/BM1684X/qwen3_0.6b_llm_w4bf16_seq512_bm1684x.bmodel \
#       --config config/ --dev_id 0 --max_new_tokens 64
#
#   # then compute CER:
#   python3 ../tools/eval_aishell.py --char=1 --v=0 \
#       ../datasets/aishell_S0764/ground_truth.txt result.txt | grep Overall
# ==============================================================================

import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from funasr_nano_infer import FunASRNanoInfer


def load_ground_truth(gt_path):
    """ground_truth.txt: '<utt_id> <chinese text>' per line."""
    items = []
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            items.append((utt, text))
    return items


def main():
    p = argparse.ArgumentParser(description='FunASR Nano batch WER eval on aishell')
    p.add_argument('--dataset', required=True, help='aishell_S0764 dir')
    p.add_argument('--encoder', default='models/BM1684X/funasr_encoder_f16_1b.bmodel')
    p.add_argument('--adapter', default='models/BM1684X/funasr_adapter_f16_1b.bmodel')
    p.add_argument('--llm', default='models/BM1684X/qwen3_0.6b_llm_w4bf16_seq512_bm1684x.bmodel')
    p.add_argument('--config', default='config/')
    p.add_argument('--dev_id', type=int, default=0)
    p.add_argument('--max_new_tokens', type=int, default=64)
    p.add_argument('--output', default='result.txt')
    args = p.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    def absf(x):
        return x if os.path.isabs(x) else os.path.join(root, x)

    engine = FunASRNanoInfer.create(
        absf(args.encoder), absf(args.adapter), absf(args.llm), absf(args.config), args.dev_id)

    gt_path = os.path.join(args.dataset, 'ground_truth.txt')
    items = load_ground_truth(gt_path)
    print(f"[eval] {len(items)} utterances from {gt_path}")

    results = []
    total_dur = 0.0
    total_time = 0.0
    t_start = time.time()

    for i, (utt, ref) in enumerate(items):
        wav_path = os.path.join(args.dataset, utt + '.wav')
        if not os.path.exists(wav_path):
            print(f"[skip] {utt}: wav not found")
            results.append((utt, ""))
            continue
        try:
            res = engine.infer(wav_path, args.max_new_tokens)
            hyp = res['text'].strip()
            total_dur += res['duration']
            total_time += res['timings']['total']
            rtf = res['rtf']
        except Exception as e:
            print(f"[err] {utt}: {e}")
            hyp = ""
            rtf = 0.0
        results.append((utt, hyp))
        elapsed = time.time() - t_start
        print(f"[{i+1}/{len(items)}] {utt}  RTF={rtf:.3f}  ({elapsed:.0f}s elapsed)")
        print(f"    ref: {ref}")
        print(f"    hyp: {hyp}")

    # write result.txt
    with open(args.output, 'w', encoding='utf-8') as f:
        for utt, hyp in results:
            f.write(f"{utt} {hyp}\n")
    print(f"\n[eval] wrote {args.output}")

    avg_rtf = total_time / 1000.0 / total_dur if total_dur > 0 else 0
    print(f"[eval] {len(items)} utts | audio={total_dur:.1f}s | infer={total_time/1000.0:.1f}s | avg RTF={avg_rtf:.3f}")
    print(f"[eval] now run:  python3 ../tools/eval_aishell.py --char=1 --v=0 "
          f"{gt_path} {args.output} | grep Overall")


if __name__ == '__main__':
    main()
