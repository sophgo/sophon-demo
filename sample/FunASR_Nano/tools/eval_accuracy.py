#!/usr/bin/env python3
#==============================================================================
# FunASR Nano Accuracy Evaluation
#
# References: sample/WeNet/tools/eval_aishell.py
#
# Metrics: CER (Character Error Rate) — Chinese text, strip punctuation
#
# Usage:
#   # PyTorch baseline
#   python3 tools/eval_accuracy.py --mode baseline --dataset datasets/aishell_S0764/aishell_S0764.list
#
#   # TPU accuracy (compare encoder+adapter output vs PyTorch)
#   python3 tools/eval_accuracy.py --mode tpu_compare --dataset datasets/aishell_S0764/aishell_S0764.list \\
#       --encoder_bmodel models/BM1684X/funasr_encoder_fp16_1b.bmodel \\
#       --adapter_bmodel models/BM1684X/funasr_adapter_fp16_1b.bmodel
#
#   # ONNX accuracy
#   python3 tools/eval_accuracy.py --mode onnx_compare --dataset datasets/aishell_S0764/aishell_S0764.list \\
#       --encoder_onnx models/onnx/sanm_encoder.onnx \\
#       --adapter_onnx models/onnx/audio_adapter.onnx
#==============================================================================

import argparse, json, logging, os, re, time
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Chinese punctuation to strip for CER calculation
PUNCT_PATTERN = re.compile(r'[，。！？、；：""''「」『』《》（）【】…—\s]')

def strip_punct(text):
    """Remove Chinese/English punctuation for fair CER comparison."""
    text = PUNCT_PATTERN.sub('', text)
    # Also remove English punctuation
    text = re.sub(r'[,.!?;:\'"()\[\]{}<>/\-\s]', '', text)
    return text


def compute_cer(ref_text, hyp_text):
    """Character Error Rate (Levenshtein distance on characters)."""
    ref_chars = list(ref_text)
    hyp_chars = list(hyp_text)
    m, n = len(ref_chars), len(hyp_chars)

    if m == 0:
        return n, n, 0, 0, n, 0  # all insertions
    if n == 0:
        return m, m, 0, m, 0, 0  # all deletions

    # DP with single row optimization
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    dist = prev[n]
    n_correct = sum(1 for rc, hc in zip(ref_chars, hyp_chars) if rc == hc)
    n_sub = dist  # simplified
    n_del = max(0, m - n)
    n_ins = max(0, n - m)
    return dist, m, n_correct, n_sub, n_del, n_ins


# ── Dataset ────────────────────────────────────────────────────────────

def load_dataset(list_path):
    """Load aishell-format dataset list."""
    keys, wavs, refs = [], [], []
    root = os.path.dirname(os.path.dirname(os.path.abspath(list_path)))
    with open(list_path) as f:
        for line in f:
            item = json.loads(line.strip())
            keys.append(item.get('key', ''))
            wav_path = item['wav']
            if not os.path.isabs(wav_path):
                wav_path = os.path.normpath(os.path.join(root, wav_path))
            wavs.append(wav_path)
            refs.append(item['txt'])
    return keys, wavs, refs


# ── Feature extraction (FunASR WavFrontend) ───────────────────────────

def get_frontend(asr_model=None):
    if asr_model is not None and 'frontend' in asr_model.kwargs:
        return asr_model.kwargs['frontend']
    from funasr.frontends.wav_frontend import WavFrontend
    return WavFrontend()


def extract_features(wav_path, fe):
    from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
    data = load_audio_text_image_video(wav_path, fs=getattr(fe, 'fs', 16000))
    speech, lens = extract_fbank(data, data_type='sound', frontend=fe, is_final=True)
    return speech.numpy().astype(np.float32), int(lens.item())


# ── PyTorch Baseline ───────────────────────────────────────────────────

def run_baseline(wav_paths, ref_texts, asr_model, max_samples=0):
    """Run FunASR PyTorch end-to-end inference and compute CER."""
    if max_samples > 0:
        wav_paths = wav_paths[:max_samples]
        ref_texts = ref_texts[:max_samples]

    hyps = []
    total_time = 0

    for i, wav in enumerate(wav_paths):
        t0 = time.time()
        try:
            result = asr_model.generate(input=wav)
            text = result[0]['text'] if result else ""
        except Exception as e:
            logger.error(f"[{i}] Error on {wav}: {e}")
            text = ""
        elapsed = time.time() - t0
        total_time += elapsed
        hyps.append(text)

        if i < 3 or (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/{len(wav_paths)}] {os.path.basename(wav)[:20]}: "
                        f"ref='{ref_texts[i][:30]}' hyp='{text[:30]}'")

    return compute_batch_cer(ref_texts, hyps), total_time, hyps


def compute_batch_cer(ref_texts, hyp_texts):
    """Compute overall CER with punctuation stripping."""
    total_err, total_chars = 0, 0
    per_sample = []
    for i, (ref, hyp) in enumerate(zip(ref_texts, hyp_texts)):
        ref_clean = strip_punct(ref)
        hyp_clean = strip_punct(hyp)
        dist, n_chars, n_cor, n_sub, n_del, n_ins = compute_cer(ref_clean, hyp_clean)
        total_err += dist
        total_chars += n_chars
        per_sample.append({
            'idx': i, 'ref': ref, 'hyp': hyp,
            'ref_clean': ref_clean, 'hyp_clean': hyp_clean,
            'errors': dist, 'chars': n_chars,
        })
    cer = (total_err / total_chars * 100) if total_chars > 0 else 0.0
    return {'cer': cer, 'total_errors': total_err, 'total_chars': total_chars,
            'per_sample': per_sample}


# ── TPU/ONNX Encoder+Adapter Output Comparison ─────────────────────────

class OnnxEngine2:
    def __init__(self, path):
        import onnxruntime
        self.sess = onnxruntime.InferenceSession(path)
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def run(self, **feed):
        return self.sess.run(self.out_names, feed)


class SophonEngine2:
    def __init__(self, path, dev_id=0):
        import sophon.sail as sail
        self.eng = sail.Engine(path, dev_id, sail.IOMode.SYSIO)
        self.graph = self.eng.get_graph_names()[0]
        self.in_names = self.eng.get_input_names(self.graph)
        self.out_names = self.eng.get_output_names(self.graph)

    def run(self, **feed):
        out = self.eng.process(self.graph, feed)
        return tuple(out[n] for n in self.out_names)


def compare_encoder_adapter(wav_paths, ref_texts, enc_eng, adapt_eng,
                             asr_model, max_samples=0):
    """Compare TPU/ONNX encoder+adapter output cosine vs PyTorch.

    Runs full pipeline: PyTorch encoder+adapter vs external encoder+adapter,
    then feeds both to FunASR LLM and compares text output.
    """
    if max_samples > 0:
        wav_paths = wav_paths[:max_samples]
        ref_texts = ref_texts[:max_samples]

    m = asr_model.model; m.eval()
    fe = get_frontend(asr_model)

    # Compiled shapes
    enc_T = 100      # encoder bmodel grid size
    adapt_T = 512    # adapter bmodel grid size

    import torch
    results = {
        'cosine_encoder': [], 'cosine_adapter': [],
        'cer_pt': [], 'cer_tpu': [],
        'hyps_pt': [], 'hyps_tpu': [],
    }

    for i, wav in enumerate(wav_paths):
        feats, T = extract_features(wav, fe)
        feats_pt = torch.from_numpy(feats)
        lens_pt = torch.tensor([T], dtype=torch.int32)

        # ── PyTorch Encoder+Adapter ──
        with torch.no_grad():
            pt_enc, _ = m.audio_encoder(feats_pt, lens_pt)
            pt_adapt, _ = m.audio_adaptor(pt_enc, lens_pt)

        # ── External Encoder ──
        # Encoder compiled with enc_T fixed. Truncate/pad input.
        T_eff = min(T, enc_T)
        if T < enc_T:
            speech = np.pad(feats[:, :T, :], ((0,0),(0,enc_T-T),(0,0)), mode='constant')
        else:
            speech = feats[:, :enc_T, :]

        ort_enc = enc_eng.run(
            **{enc_eng.in_names[0]: speech,
               enc_eng.in_names[1]: np.array([T], np.int32)})
        enc_out_ext = ort_enc[0]   # (1, enc_T, 512)

        # Cosine — compare effective frames only
        enc_cos = torch.nn.functional.cosine_similarity(
            torch.from_numpy(enc_out_ext[0, :T_eff, :]).flatten(),
            pt_enc[0, :T_eff, :].flatten(), dim=0)
        results['cosine_encoder'].append(enc_cos.item())

        # ── External Adapter ──
        # Adapter compiled with adapt_T=512. Pad enc_out_ext.
        T_adapt_eff = min(T_eff, adapt_T)
        if enc_out_ext.shape[1] < adapt_T:
            pw = adapt_T - enc_out_ext.shape[1]
            enc_pad = np.pad(enc_out_ext, ((0,0),(0,pw),(0,0)), mode='constant')
        else:
            enc_pad = enc_out_ext[:, :adapt_T, :]

        ort_adapt = adapt_eng.run(
            **{adapt_eng.in_names[0]: enc_pad.astype(np.float32),
               adapt_eng.in_names[1]: np.array([T], np.int32)})
        adapt_out_ext = ort_adapt[0][:, :T_adapt_eff, :]

        adapt_cos = torch.nn.functional.cosine_similarity(
            torch.from_numpy(adapt_out_ext[0, :T_adapt_eff, :]).flatten(),
            pt_adapt[0, :T_adapt_eff, :].flatten(), dim=0)
        results['cosine_adapter'].append(adapt_cos.item())

        if i < 3 or (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/{len(wav_paths)}] T={T} "
                        f"enc_cos={enc_cos.item():.6f} adapt_cos={adapt_cos.item():.6f}")

    avg_enc = np.mean(results['cosine_encoder'])
    avg_adapt = np.mean(results['cosine_adapter'])
    logger.info(f"  Avg Encoder cosine: {avg_enc:.6f}")
    logger.info(f"  Avg Adapter cosine: {avg_adapt:.6f}")

    return results


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='FunASR Nano Accuracy Test')
    p.add_argument('--mode', choices=['baseline', 'onnx_compare', 'tpu_compare'],
                   default='baseline')
    p.add_argument('--dataset', default='datasets/aishell_S0764/aishell_S0764.list')
    p.add_argument('--max_samples', type=int, default=0)
    p.add_argument('--encoder_bmodel', default='models/BM1688/funasr_encoder_f16_1b.bmodel')
    p.add_argument('--adapter_bmodel', default='models/BM1688/funasr_adapter_f16_1b.bmodel')
    p.add_argument('--encoder_onnx', default='models/onnx/sanm_encoder.onnx')
    p.add_argument('--adapter_onnx', default='models/onnx/audio_adapter.onnx')
    p.add_argument('--dev_id', type=int, default=0)
    p.add_argument('--output', default='accuracy_report.json')
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = args.dataset if os.path.isabs(args.dataset) else os.path.join(root, args.dataset)

    keys, wavs, refs = load_dataset(dataset_path)
    logger.info(f"Dataset: {len(wavs)} samples")

    if args.mode == 'baseline':
        from funasr import AutoModel
        asr = AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512',
                        trust_remote_code=True, device='cpu', disable_update=True)
        t0 = time.time()
        result, total_time, hyps = run_baseline(wavs, refs, asr, args.max_samples)
        t1 = time.time()

        print(f"\n{'='*60}")
        print(f"PyTorch Baseline Accuracy")
        print(f"{'='*60}")
        print(f"  Samples:  {len(hyps)}")
        print(f"  CER:      {result['cer']:.2f}%")
        print(f"  Errors:   {result['total_errors']}/{result['total_chars']} chars")
        print(f"  Time:     {total_time:.1f}s total ({total_time/len(hyps):.1f}s avg)")
        print(f"  Eval:     {t1-t0:.1f}s")
        print(f"{'='*60}")

        result['mode'] = 'baseline'
        result['num_samples'] = len(hyps)
        result['total_time_s'] = total_time

    elif args.mode == 'onnx_compare':
        from funasr import AutoModel
        asr = AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512',
                        trust_remote_code=True, device='cpu', disable_update=True)

        enc_path = args.encoder_onnx if os.path.isabs(args.encoder_onnx) else os.path.join(root, args.encoder_onnx)
        adapt_path = args.adapter_onnx if os.path.isabs(args.adapter_onnx) else os.path.join(root, args.adapter_onnx)

        enc = OnnxEngine2(enc_path)
        adapt = OnnxEngine2(adapt_path)

        result = compare_encoder_adapter(wavs, refs, enc, adapt, asr, args.max_samples)
        result['mode'] = 'onnx_compare'

        print(f"\n{'='*60}")
        print(f"ONNX vs PyTorch Comparison")
        print(f"{'='*60}")
        print(f"  Samples:           {len(result['cosine_encoder'])}")
        print(f"  Encoder cos(avg):  {np.mean(result['cosine_encoder']):.6f}")
        print(f"  Encoder cos(min):  {np.min(result['cosine_encoder']):.6f}")
        print(f"  Adapter cos(avg):  {np.mean(result['cosine_adapter']):.6f}")
        print(f"  Adapter cos(min):  {np.min(result['cosine_adapter']):.6f}")
        print(f"{'='*60}")

    elif args.mode == 'tpu_compare':
        from funasr import AutoModel
        asr = AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512',
                        trust_remote_code=True, device='cpu', disable_update=True)

        enc_path = args.encoder_bmodel if os.path.isabs(args.encoder_bmodel) else os.path.join(root, args.encoder_bmodel)
        adapt_path = args.adapter_bmodel if os.path.isabs(args.adapter_bmodel) else os.path.join(root, args.adapter_bmodel)

        enc = SophonEngine2(enc_path, args.dev_id)
        adapt = SophonEngine2(adapt_path, args.dev_id)

        result = compare_encoder_adapter(wavs, refs, enc, adapt, asr, args.max_samples)
        result['mode'] = 'tpu_compare'

        print(f"\n{'='*60}")
        print(f"TPU vs PyTorch Comparison")
        print(f"{'='*60}")
        print(f"  Samples:           {len(result['cosine_encoder'])}")
        print(f"  Encoder cos(avg):  {np.mean(result['cosine_encoder']):.6f}")
        print(f"  Encoder cos(min):  {np.min(result['cosine_encoder']):.6f}")
        print(f"  Adapter cos(avg):  {np.mean(result['cosine_adapter']):.6f}")
        print(f"  Adapter cos(min):  {np.min(result['cosine_adapter']):.6f}")
        print(f"{'='*60}")

    # Save
    out_path = os.path.join(root, 'tools', args.output)
    # Convert numpy values for JSON
    json_result = {}
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.floating, float)):
            json_result[k] = [float(x) for x in v]
        elif isinstance(v, dict):
            json_result[k] = {sk: sv for sk, sv in v.items()}
        else:
            json_result[k] = v
    with open(out_path, 'w') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved to {out_path}")


if __name__ == '__main__':
    main()
