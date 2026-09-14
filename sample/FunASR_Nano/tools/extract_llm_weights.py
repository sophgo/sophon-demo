#!/usr/bin/env python3
# ==============================================================================
# Extract a standalone Qwen3-0.6B HuggingFace model from Fun-ASR-Nano-2512.
#
# Fun-ASR-Nano's model.pt is a flat state_dict whose LLM weights live under the
# `llm.` prefix (standard Qwen3ForCausalLM). This script strips that prefix and
# writes a HF model directory that `llm_convert.py` can compile into a w4bf16
# bmodel (embedding / block_* / block_cache_* / lm_head graphs).
#
# Usage:
#   python3 extract_llm_weights.py \
#       --model_dir ~/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512 \
#       --out_dir  tools/qwen3_0.6b_llm
#
# NOTE: the output dir name must NOT contain both "qwen" and "asr" substrings,
#       otherwise llm_convert.py mis-triggers a `qwen_asr` import.
# ==============================================================================

import argparse, os, shutil
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default=None,
                    help='Fun-ASR-Nano-2512 model dir (containing model.pt and Qwen3-0.6B/)')
    ap.add_argument('--out_dir', default='tools/qwen3_0.6b_llm',
                    help='output standalone Qwen3-0.6B HF model dir')
    args = ap.parse_args()

    if args.model_dir is None:
        # try modelscope cache first
        candidates = [
            os.path.expanduser('~/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512'),
            os.path.expanduser('~/.cache/huggingface/hub/models/FunAudioLLM/Fun-ASR-Nano-2512'),
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, 'model.pt')):
                args.model_dir = c
                break
        if args.model_dir is None:
            raise SystemExit("model.pt not found; pass --model_dir <Fun-ASR-Nano-2512 dir>")

    pt_path = os.path.join(args.model_dir, 'model.pt')
    qwen_dir = os.path.join(args.model_dir, 'Qwen3-0.6B')
    print(f"[extract] loading {pt_path}")
    sd = torch.load(pt_path, map_location='cpu')

    # LLM weights are prefixed with 'llm.' (standard Qwen3ForCausalLM keys)
    llm_sd = {}
    for k, v in sd.items():
        if k.startswith('llm.'):
            llm_sd[k[len('llm.'):]] = v
    n_llm = len(llm_sd)
    print(f"[extract] {n_llm} LLM keys (llm.* prefix stripped)")
    if n_llm == 0:
        raise SystemExit("no 'llm.' keys found in model.pt")

    os.makedirs(args.out_dir, exist_ok=True)
    out_pt = os.path.join(args.out_dir, 'pytorch_model.bin')
    torch.save(llm_sd, out_pt)
    print(f"[extract] wrote {out_pt} ({os.path.getsize(out_pt)/1024/1024:.0f}MB)")

    # copy config + tokenizer from the bundled Qwen3-0.6B subdir
    if not os.path.isdir(qwen_dir):
        raise SystemExit(f"missing {qwen_dir}; cannot copy tokenizer/config")
    for f in os.listdir(qwen_dir):
        src = os.path.join(qwen_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out_dir, f))
            print(f"[extract] copied {f}")
    print(f"[extract] done. Compile with:\n  llm_convert -m {args.out_dir} "
          f"-c bm1684x --quantize w4bf16 --num_core 1 --max_input_length 256 -s 512")


if __name__ == '__main__':
    main()
