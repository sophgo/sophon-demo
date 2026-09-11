#!/usr/bin/env python3
"""
Export SeACoParaformer (speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404)
into 3 ONNX parts matching the sample's 3-bmodel contract:

  encoder.onnx   (speech (B,T,560) f32, speech_lengths (B,) i32)
                  -> enc_out (B,T,512), hidden (B,T+1,512), alphas (B,T+1), token_num (B,)
  decoder.onnx   (enc_out (B,T,512) f32, enc_len (B,) i32,
                  pre_acoustic_embeds (B,N,512) f32, pre_token_length (B,) i32)
                  -> logits (B,N,vocab), hidden (B,N,512)
  predictor.onnx (enc_out (B,T,512) f32, enc_len (B,) i32)
                  -> us_alphas (B,3T) [masked, unnormalized], token_num (B,)

Follows the export_contextual part-split pattern from the FunASR-bmodel repo.
The CIF integrate-and-fire loops (cif_export / cif_wo_hidden) are NOT traced:
- encoder part stops at tail_process_fn (hidden/alphas/token_num),
- predictor part is the CifPredictorV3 upsample head WITHOUT final normalization
  (normalization by pre_token_length happens on CPU in the sample, same as the
  FunASR-bmodel bmodel flow).

Usage (HOST, needs funasr + torch):
    python3 tools/export_onnx.py --model_dir <modelscope model dir> --output_dir ../models/onnx
"""

import argparse
import os

import numpy as np
import torch

from funasr import AutoModel
from funasr.register import tables
from funasr.utils.torch_function import sequence_mask

MAX_SEQ_LEN = 3000   # only a ctor arg; sequence_mask.forward derives size from lengths
FEATS_DIM = 560


class EncoderPart(torch.nn.Module):
    """encoder + CifPredictorV3 head up to tail_process_fn (no CIF loop)."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.predictor = model.predictor
        self.make_pad_mask = sequence_mask(MAX_SEQ_LEN, flip=False)

    def forward(self, speech, speech_lengths):
        enc, enc_len = self.encoder(speech=speech, speech_lengths=speech_lengths)
        p = self.predictor
        mask = self.make_pad_mask(enc_len)                # (B, T)

        context = enc.transpose(1, 2)
        queries = p.pad(context)
        output = torch.relu(p.cif_conv1d(queries))
        output = output.transpose(1, 2)
        output = p.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * p.smooth_factor - p.noise_threshold)
        m3 = mask[:, None, :].transpose(-1, -2).float()   # (B, T, 1)
        alphas = (alphas * m3).squeeze(-1)                # (B, T)
        m2 = m3.squeeze(-1)                               # (B, T)
        hidden, alphas, token_num = p.tail_process_fn(enc, alphas, mask=m2)
        return enc, hidden, alphas, token_num


class DecoderPart(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, enc, enc_len, pre_acoustic_embeds, pre_token_length):
        logits, hidden, _ = self.decoder(
            enc, enc_len, pre_acoustic_embeds, pre_token_length,
            return_hidden=True, return_both=True,
        )
        return logits, hidden


class PredictorPart(torch.nn.Module):
    """CifPredictorV3 upsample head (ConvTranspose x3 + BLSTM) without normalization."""

    def __init__(self, model):
        super().__init__()
        self.predictor = model.predictor
        self.make_pad_mask = sequence_mask(MAX_SEQ_LEN, flip=False)

    def forward(self, enc, enc_len):
        p = self.predictor
        context = enc.transpose(1, 2)
        output2 = p.upsample_cnn(context)                 # (B, 512, 3T)
        output2 = output2.transpose(1, 2)                 # (B, 3T, 512)
        output2, (_, _) = p.blstm(output2)
        alphas2 = torch.sigmoid(p.cif_output2(output2))   # (B, 3T, 1)
        alphas2 = torch.nn.functional.relu(
            alphas2 * p.smooth_factor2 - p.noise_threshold2)

        # NOTE: single-stream inference has no padding (mask is all ones),
        # so the pad-mask (Range op, breaks static bm1684x2 compile) is
        # dropped and token_num is simply the sum over all frames.
        alphas2 = alphas2.squeeze(-1)                      # (B, 3T)
        token_num = alphas2.sum(-1)
        return alphas2, token_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                        help="dir with model.pt/config.yaml (modelscope layout)")
    parser.add_argument("--output_dir", default="../models/onnx")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading AutoModel from", args.model_dir)
    model = AutoModel(model=args.model_dir, device="cpu", disable_update=True)

    # Wrap submodules into export (onnx-friendly) classes, same as
    # funasr seaco export_rebuild_model does.
    m = model.model
    encoder_class = tables.encoder_classes.get("SANMEncoderExport")
    m.encoder = encoder_class(m.encoder, onnx=True)

    predictor_class = tables.predictor_classes.get("CifPredictorV3Export")
    m.predictor = predictor_class(m.predictor, onnx=True)

    decoder_class = tables.decoder_classes.get("ParaformerSANMDecoderExport")
    m.decoder = decoder_class(m.decoder, onnx=True)

    # ------------------------------------------------------------------
    # encoder.onnx
    # ------------------------------------------------------------------
    enc_part = EncoderPart(m).eval()
    speech = torch.randn(2, 30, FEATS_DIM, dtype=torch.float32)
    speech_lengths = torch.tensor([15, 30], dtype=torch.int32)
    with torch.no_grad():
        ref = enc_part(speech, speech_lengths)
    print("encoder ref shapes:", [tuple(r.shape) for r in ref])
    assert tuple(ref[0].shape) == (2, 30, 512)
    assert tuple(ref[1].shape) == (2, 31, 512), "hidden must be (B, T+1, D)"
    assert tuple(ref[2].shape) == (2, 31), "alphas must be (B, T+1)"
    assert tuple(ref[3].shape) == (2,)

    torch.onnx.export(
        enc_part,
        (speech, speech_lengths),
        os.path.join(args.output_dir, "encoder.onnx"),
        input_names=["speech", "speech_lengths"],
        output_names=["enc_out", "hidden", "alphas", "token_num"],
        dynamic_axes={
            "speech": {0: "batch_size", 1: "feats_length"},
            "speech_lengths": {0: "batch_size"},
            "enc_out": {0: "batch_size", 1: "feats_length"},
            "hidden": {0: "batch_size", 1: "feats_length_plus"},
            "alphas": {0: "batch_size", 1: "feats_length_plus"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print("exported encoder.onnx")

    # ------------------------------------------------------------------
    # decoder.onnx
    # ------------------------------------------------------------------
    dec_part = DecoderPart(m).eval()
    enc = torch.randn(2, 30, 512, dtype=torch.float32)
    enc_len = torch.tensor([15, 30], dtype=torch.int32)
    pre_acoustic_embeds = torch.randn(2, 9, 512, dtype=torch.float32)
    pre_token_length = torch.tensor([6, 9], dtype=torch.int32)
    with torch.no_grad():
        ref = dec_part(enc, enc_len, pre_acoustic_embeds, pre_token_length)
    print("decoder ref shapes:", [tuple(r.shape) for r in ref])

    torch.onnx.export(
        dec_part,
        (enc, enc_len, pre_acoustic_embeds, pre_token_length),
        os.path.join(args.output_dir, "decoder.onnx"),
        input_names=["enc", "enc_len", "pre_acoustic_embeds", "pre_token_length"],
        output_names=["logits", "hidden"],
        dynamic_axes={
            "enc": {0: "batch_size", 1: "feats_length"},
            "enc_len": {0: "batch_size"},
            "pre_acoustic_embeds": {0: "batch_size", 1: "unknown_length"},
            "pre_token_length": {0: "batch_size"},
            "logits": {0: "batch_size", 1: "logits_length"},
            "hidden": {0: "batch_size", 1: "logits_length"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print("exported decoder.onnx")

    # ------------------------------------------------------------------
    # predictor.onnx
    # ------------------------------------------------------------------
    pred_part = PredictorPart(m).eval()
    with torch.no_grad():
        ref = pred_part(enc, enc_len)
    print("predictor ref shapes:", [tuple(r.shape) for r in ref])
    assert tuple(ref[0].shape) == (2, 90), "us_alphas must be (B, 3T)"

    torch.onnx.export(
        pred_part,
        (enc, enc_len),
        os.path.join(args.output_dir, "predictor.onnx"),
        input_names=["enc", "enc_len"],
        output_names=["us_alphas", "token_num"],
        dynamic_axes={
            "enc": {0: "batch_size", 1: "feats_length"},
            "enc_len": {0: "batch_size"},
            "us_alphas": {0: "batch_size", 1: "alphas_length"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print("exported predictor.onnx")


if __name__ == "__main__":
    main()
