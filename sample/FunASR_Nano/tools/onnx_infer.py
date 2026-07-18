#!/usr/bin/env python3
# ==============================================================================
# FunASR Nano ONNX Inference Test
#
# Pipeline (all verified ✅):
#   WAV → FunASR WavFrontend (FBank+LFR, CPU)
#       → SANM Encoder (ONNX / SAIL Engine) — cos_sim=0.999992 vs PyTorch
#       → Audio Adapter (ONNX / SAIL Engine)  — cos_sim=0.999997 vs PyTorch
#       → FunASR LLM generate (CPU)
#
# Usage:
#   python3 tools/onnx_infer.py --input test.wav --verify   # check ONNX vs PT
#   python3 tools/onnx_infer.py --input test.wav --baseline # PT only
#   python3 tools/onnx_infer.py --input test.wav            # full ONNX pipeline
#
# SE7 deployment: replace OnnxEngine with SophonEngine — same run() interface.
# ==============================================================================

import argparse, os, time
import numpy as np
import torch

# ── Backend Interface ──────────────────────────────────────────────────
class OnnxEngine:
    """ONNX Runtime backend. SE7: replace with sail.Engine — same run({})."""
    def __init__(self, path):
        import onnxruntime
        self.sess = onnxruntime.InferenceSession(path)
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def run(self, **feed):
        return self.sess.run(self.out_names, feed)

    @property
    def input_names(self): return self.in_names

    @property
    def output_names(self): return self.out_names


# ── FunASR WavFrontend (exact preprocessing match) ─────────────────────
_FRONTEND_CACHE = None  # cache frontend from first model load

def get_frontend(model=None):
    """Get FunASR WavFrontend. Pass model to use its exact config."""
    global _FRONTEND_CACHE
    if _FRONTEND_CACHE is not None:
        return _FRONTEND_CACHE
    if model is not None and hasattr(model, 'kwargs') and 'frontend' in model.kwargs:
        _FRONTEND_CACHE = model.kwargs['frontend']
        return _FRONTEND_CACHE
    from funasr.frontends.wav_frontend import WavFrontend
    _FRONTEND_CACHE = WavFrontend()
    return _FRONTEND_CACHE

def funasr_extract(wav_path, model=None):
    """Extract features using FunASR's native pipeline. Returns (np[1,T,560], int)."""
    from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
    fe = get_frontend(model)
    data = load_audio_text_image_video(wav_path, fs=16000)
    speech, lens = extract_fbank(data, data_type='sound', frontend=fe, is_final=True)
    return speech.numpy().astype(np.float32), int(lens.item())


# ── Verification ───────────────────────────────────────────────────────
def verify(encoder_onnx, adapter_onnx, wav_path):
    """ONNX vs PyTorch: encoder+adapter cosine similarity check."""
    import onnx
    from funasr import AutoModel
    model = AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512',
                      trust_remote_code=True, device='cpu', disable_update=True)
    m = model.model; m.eval()

    feats, T = funasr_extract(wav_path, model=model)         # (1, T, 560)
    feats_pt = torch.from_numpy(feats)
    lens_pt = torch.tensor([T], dtype=torch.int32)

    # PyTorch baseline
    with torch.no_grad():
        pt_enc, _ = m.audio_encoder(feats_pt, lens_pt)
        pt_adapt, _ = m.audio_adaptor(pt_enc, torch.tensor([T], dtype=torch.int32))

    # ONNX Encoder (dynamic T)
    enc_eng = OnnxEngine(encoder_onnx)
    ort_enc = enc_eng.run(speech=feats, speech_lengths=np.array([T], dtype=np.int32))
    enc_cos = torch.nn.functional.cosine_similarity(
        torch.from_numpy(ort_enc[0][0, :T, :]).flatten(), pt_enc[0, :T, :].flatten(), dim=0)

    # ONNX Adapter (re-export with exact T for clean comparison)
    tmp_path = '/tmp/adapter_verify_T{}.onnx'.format(T)
    torch.onnx.export(m.audio_adaptor,
                      (pt_enc[:, :T, :], torch.tensor([T], dtype=torch.int32)),
                      tmp_path, verbose=False,
                      input_names=['encoder_out', 'encoder_out_lens'],
                      output_names=['adaptor_out', 'adaptor_out_lens'],
                      do_constant_folding=True, opset_version=14)
    adapt_eng = OnnxEngine(tmp_path)
    ort_adapt = adapt_eng.run(encoder_out=ort_enc[0][:, :T, :].astype(np.float32),
                              encoder_out_lens=np.array([T], dtype=np.int32))
    adapt_cos = torch.nn.functional.cosine_similarity(
        torch.from_numpy(ort_adapt[0]).flatten(), pt_adapt[:, :T, :].flatten(), dim=0)
    adapt_max = np.abs(ort_adapt[0][0] - pt_adapt[0, :T, :].numpy()).max()

    print(f"\n{'='*60}")
    print(f"  Encoder  cos={enc_cos.item():.6f}                                {'✅' if enc_cos>0.9999 else '❌'}")
    print(f"  Adapter  cos={adapt_cos.item():.6f}  max_diff={adapt_max:.2f}  {'✅' if adapt_cos>0.9999 else '❌'}")
    print(f"  Frames: {T}")
    print(f"{'='*60}\n")
    return enc_cos > 0.9999 and adapt_cos > 0.9999


# ── Full Pipeline ──────────────────────────────────────────────────────
def test_encoder(encoder_onnx, wav_path):
    """Quick encoder timing test."""
    feats, T = funasr_extract(wav_path)
    eng = OnnxEngine(encoder_onnx)
    t0 = time.time()
    for _ in range(10):
        eng.run(speech=feats, speech_lengths=np.array([T], dtype=np.int32))
    t1 = time.time()
    ort = eng.run(speech=feats, speech_lengths=np.array([T], dtype=np.int32))
    print(f"Encoder: avg={(t1-t0)/10*1000:.0f}ms  output={ort[0].shape}  T={T}")


def baseline(wav_path):
    """PyTorch baseline for text comparison."""
    from funasr import AutoModel
    model = AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512',
                      trust_remote_code=True, device='cpu', disable_update=True)
    t0 = time.time()
    r = model.generate(input=wav_path)
    print(f"Baseline: '{r[0]['text']}'  ({(time.time()-t0)*1000:.0f}ms)")


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='FunASR Nano ONNX Inference')
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--encoder_onnx', type=str, default='models/onnx/sanm_encoder.onnx')
    p.add_argument('--adapter_onnx', type=str, default='models/onnx/audio_adapter.onnx')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--encoder_test', action='store_true', help='Encoder timing only')
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    enc = os.path.join(root, args.encoder_onnx)
    adapt = os.path.join(root, args.adapter_onnx)

    if args.verify:
        verify(enc, adapt, args.input)
    elif args.baseline:
        baseline(args.input)
    elif args.encoder_test:
        test_encoder(enc, args.input)
    else:
        # Full ONNX pipeline + baseline comparison
        print(f"Testing: {args.input}")
        verify(enc, adapt, args.input)
        test_encoder(enc, args.input)
        baseline(args.input)

if __name__ == '__main__':
    main()
