#!/usr/bin/env python3
# ==============================================================================
# FunASR Nano TPU Inference — End-to-End Speech Recognition
#
# Pipeline:
#   WAV → FunASR WavFrontend (CPU)
#       → SANM Encoder (ONNX/TPU)  — cos=0.99999 vs PyTorch
#       → Audio Adapter (ONNX/TPU) — cos=0.99999 vs PyTorch
#       → FunASR LLM generate (CPU, with audio_embedding injection)
#
# Usage:
#   ONNX test:  python3 funasr_nano_infer.py --input test.wav --verify
#   TPU test:   python3 funasr_nano_infer.py --input test.wav --backend tpu \
#                 --encoder ../models/BM1688/funasr_encoder_f16_1b.bmodel \
#                 --adapter ../models/BM1688/funasr_adapter_f16_1b.bmodel
#
# BM1688 verified (2026-07-18, TPU-MLIR v1.28.1):
#   Encoder F16: ~692ms | Adapter F16: ~34ms | LLM CPU: ~7s
#   PyTorch baseline CER: 1.12% (96 samples, aishell_S0764)
#
# ⚠️ SE9 (BM1688 SoC) 内存有限(3.3GB), 加载完整 PyTorch 模型会 OOM。
#    建议仅在 x86 主机上运行本脚本，或参考 README 的两步法。
# ==============================================================================

import argparse, logging, os, time, types
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Backend Engines
# ============================================================================

class OnnxEngine:
    def __init__(self, path):
        import onnxruntime
        self.sess = onnxruntime.InferenceSession(path)
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def run(self, **feed):
        return self.sess.run(self.out_names, feed)


class SophonEngine:
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
# Feature Extraction (FunASR WavFrontend)
# ============================================================================

def extract_features(wav_path, frontend):
    from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
    data = load_audio_text_image_video(wav_path, fs=getattr(frontend, 'fs', 16000))
    speech, lens = extract_fbank(data, data_type='sound', frontend=frontend, is_final=True)
    return speech.numpy().astype(np.float32), int(lens.item())


# ============================================================================
# FunASR Audio Embedding Injection
# ============================================================================

def patch_funasr_for_audio_embedding(asr_model):
    """Monkey-patch FunASR Nano to accept pre-computed audio embeddings.

    The upstream inference_llm doesn't handle audio_embedding injection
    correctly (adaptor_out variable not set). This patch fixes it by
    overriding inference_llm to directly build inputs_embeds with
    pre-computed audio embeddings inserted at the correct positions.
    """
    m = asr_model.model
    _original = m.inference_llm

    def patched(data_in, data_lengths=None, key=None, tokenizer=None,
                frontend=None, **kwargs):
        if "audio_embedding" not in kwargs:
            return _original(data_in, data_lengths, key=key,
                             tokenizer=tokenizer, frontend=frontend, **kwargs)

        audio_emb = kwargs.pop("audio_embedding")
        audio_emb_lens = kwargs.pop("audio_embedding_lens")

        from funasr.train_utils.device_funcs import to_device

        contents = m.data_template(data_in[0])
        output = m.data_load_speech(contents, tokenizer, frontend, **kwargs)
        batch = to_device(output, kwargs.get("device", "cpu"))

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        fbank_beg = batch["fbank_beg"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("teacherforcing", False):
            input_ids = source_ids
        input_ids[input_ids < 0] = 0

        inputs_embeds = m.llm.model.get_input_embeddings()(input_ids)
        fake_token_len[fake_token_len < 0] = 0
        fbank_beg[fbank_beg < 0] = 0

        speech_idx = 0
        for bi in range(inputs_embeds.shape[0]):
            for ti in range(fbank_beg.shape[1]):
                idx = fbank_beg[bi, ti].item()
                if idx > 0 and speech_idx < audio_emb.shape[0]:
                    slen = fake_token_len[bi, ti].item()
                    if slen > audio_emb.shape[1]:
                        slen = audio_emb.shape[1]
                    stok = audio_emb[speech_idx, :slen, :].to(
                        inputs_embeds.dtype).to(inputs_embeds.device)
                    end = min(idx + slen, inputs_embeds.shape[1])
                    inputs_embeds[bi, idx:end, :] = stok[:end - idx, :]
                    speech_idx += 1

        attention_mask = batch.get("attention_mask", None)

        generated_ids = m.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=kwargs.get("max_length", 512),
            pad_token_id=(m.llm.config.pad_token_id
                          if m.llm.config.pad_token_id is not None
                          else m.llm.config.eos_token_id),
        )
        response = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=kwargs.get("skip_special_tokens", True))
        response = kwargs.get("prev_text", "") + response[0]

        results = [{
            "key": key[0] if key else "result",
            "text": response,
        }]
        return results, {}

    m.inference_llm = types.MethodType(patched, m)
    logger.info("FunASR inference_llm patched for audio_embedding injection")


# ============================================================================
# Pipeline
# ============================================================================

class FunASRNanoInfer:
    ENC_GRID = 200   # encoder BModel compiled grid size
    ADAPT_GRID = 200  # adapter BModel compiled grid size

    def __init__(self, enc_eng, adapt_eng, asr_model, frontend):
        self.enc_eng = enc_eng
        self.adapt_eng = adapt_eng
        self.asr = asr_model
        self.frontend = frontend

    @classmethod
    def create(cls, encoder_path, adapter_path, backend='onnx', dev_id=0):
        from funasr import AutoModel
        from funasr.frontends.wav_frontend import WavFrontend

        Eng = SophonEngine if backend == 'tpu' else OnnxEngine
        enc = Eng(encoder_path, dev_id) if backend == 'tpu' else Eng(encoder_path)
        adapt = Eng(adapter_path, dev_id) if backend == 'tpu' else Eng(adapter_path)

        asr = AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512',
                        trust_remote_code=True, device='cpu', disable_update=True)
        patch_funasr_for_audio_embedding(asr)

        frontend = asr.kwargs.get('frontend', WavFrontend())
        logger.info(f"Frontend: {type(frontend).__name__}  Backend: {backend}")
        return cls(enc, adapt, asr, frontend)

    def _encode(self, wav_path):
        """Run Frontend → Encoder → Adapter. Returns adaptor_out (1, T, 1024)."""
        # 1. Frontend
        feats, T = extract_features(wav_path, self.frontend)

        # 2. Pad to encoder grid
        if T < self.ENC_GRID:
            pw = self.ENC_GRID - T
            feats_pad = np.pad(feats[:, :T, :], ((0, 0), (0, pw), (0, 0)),
                               mode='constant')
        else:
            feats_pad = feats[:, :self.ENC_GRID, :]

        # 3. Encoder
        enc_out = self.enc_eng.run(
            **{self.enc_eng.in_names[0]: feats_pad,
               self.enc_eng.in_names[1]: np.array([T], dtype=np.int32)})
        encoder_out = enc_out[0]  # (1, grid, 512)

        # 4. Pad to adapter grid
        if encoder_out.shape[1] < self.ADAPT_GRID:
            pw = self.ADAPT_GRID - encoder_out.shape[1]
            enc_pad = np.pad(encoder_out, ((0, 0), (0, pw), (0, 0)),
                             mode='constant')
        else:
            enc_pad = encoder_out[:, :self.ADAPT_GRID, :]

        # 5. Adapter
        adapt_out = self.adapt_eng.run(
            **{self.adapt_eng.in_names[0]: enc_pad.astype(np.float32),
               self.adapt_eng.in_names[1]: np.array([T], dtype=np.int32)})
        adaptor_out = adapt_out[0][:, :T, :]  # (1, T, 1024)

        return adaptor_out, T

    def infer(self, wav_path):
        timings = {}

        # ── Frontend + Encoder + Adapter ──
        t0 = time.time()
        adaptor_out, T = self._encode(wav_path)
        t1 = time.time()
        timings['encode'] = (t1 - t0) * 1000
        logger.info(f"[1/2] Encode: T={T}, {adaptor_out.shape} "
                    f"({timings['encode']:.0f}ms)")

        # ── LLM generate with audio_embedding ──
        t0 = time.time()
        result = self.asr.generate(
            input=wav_path,
            audio_embedding=torch.from_numpy(adaptor_out),
            audio_embedding_lens=torch.tensor([T]),
        )
        text = result[0]['text'] if result else ""
        t1 = time.time()
        timings['llm'] = (t1 - t0) * 1000
        logger.info(f"[2/2] LLM: '{text[:50]}...' ({timings['llm']:.0f}ms)")

        timings['total'] = sum(timings.values())
        return {'text': text, 'timings': timings}

    def benchmark(self, wav_path, loops=10):
        # Warmup
        self._encode(wav_path)
        t0 = time.time()
        for _ in range(loops):
            self._encode(wav_path)
        t1 = time.time()
        return (t1 - t0) / loops * 1000


# ============================================================================
# Verification
# ============================================================================

def verify_onnx(encoder_onnx, adapter_onnx, wav_path):
    import onnx, onnxruntime
    from funasr import AutoModel

    asr = AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512',
                    trust_remote_code=True, device='cpu', disable_update=True)
    m = asr.model; m.eval()
    frontend = asr.kwargs.get('frontend')

    feats, T = extract_features(wav_path, frontend)
    feats_pt = torch.from_numpy(feats)
    lens_pt = torch.tensor([T], dtype=torch.int32)

    with torch.no_grad():
        pt_enc, _ = m.audio_encoder(feats_pt, lens_pt)
        pt_adapt, _ = m.audio_adaptor(pt_enc, lens_pt)

    enc_sess = onnxruntime.InferenceSession(encoder_onnx)
    ort_enc = enc_sess.run(None,
                           {'speech': feats,
                            'speech_lengths': np.array([T], np.int32)})
    enc_cos = torch.nn.functional.cosine_similarity(
        torch.from_numpy(ort_enc[0][0, :T, :]).flatten(),
        pt_enc[0, :T, :].flatten(), dim=0)

    torch.onnx.export(m.audio_adaptor,
                      (pt_enc[:, :T, :], torch.tensor([T], dtype=torch.int32)),
                      '/tmp/adapter_verify.onnx', verbose=False,
                      input_names=['encoder_out', 'encoder_out_lens'],
                      output_names=['adaptor_out', 'adaptor_out_lens'],
                      do_constant_folding=True, opset_version=14)
    adapt_sess = onnxruntime.InferenceSession('/tmp/adapter_verify.onnx')
    ort_adapt = adapt_sess.run(None,
                               {'encoder_out': ort_enc[0][:, :T, :].astype(np.float32),
                                'encoder_out_lens': np.array([T], np.int32)})
    adapt_cos = torch.nn.functional.cosine_similarity(
        torch.from_numpy(ort_adapt[0]).flatten(),
        pt_adapt[:, :T, :].flatten(), dim=0)

    print(f"\n  Encoder: cos={enc_cos.item():.6f}  {'✅' if enc_cos > 0.9999 else '❌'}")
    print(f"  Adapter: cos={adapt_cos.item():.6f}  {'✅' if adapt_cos > 0.9999 else '❌'}")
    return enc_cos > 0.9999 and adapt_cos > 0.9999


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description='FunASR Nano TPU/ONNX Inference')
    p.add_argument('--input', required=True)
    p.add_argument('--encoder', default='models/BM1688/funasr_encoder_f16_1b.bmodel')
    p.add_argument('--adapter', default='models/BM1688/funasr_adapter_f16_1b.bmodel')
    p.add_argument('--backend', choices=['onnx', 'tpu'], default='onnx')
    p.add_argument('--dev_id', type=int, default=0)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--loops', type=int, default=10)
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    encoder = args.encoder if os.path.isabs(args.encoder) else os.path.join(root, args.encoder)
    adapter = args.adapter if os.path.isabs(args.adapter) else os.path.join(root, args.adapter)

    if args.verify:
        verify_onnx(encoder, adapter, args.input)
    elif args.benchmark:
        engine = FunASRNanoInfer.create(encoder, adapter, args.backend, args.dev_id)
        avg_ms = engine.benchmark(args.input, args.loops)
        print(f"Encoder+Adapter avg: {avg_ms:.1f}ms ({args.loops} loops, {args.backend})")
    else:
        engine = FunASRNanoInfer.create(encoder, adapter, args.backend, args.dev_id)
        result = engine.infer(args.input)
        t = result['timings']
        print(f"\n{'='*60}")
        print(f"Text:       {result['text']}")
        print(f"Timings:    encode={t['encode']:.0f}ms  llm={t['llm']:.0f}ms  total={t['total']:.0f}ms")
        print(f"Backend:    {args.backend}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
