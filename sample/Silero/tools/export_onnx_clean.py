#!/usr/bin/env python3
"""
Export Silero VAD to ONNX with clean graph (no control flow ops).
Reconstructs the model in pure PyTorch to eliminate If/Size/Shape/Equal etc.
that come from the TorchScript LSTM state handling.

Target: BM1684X TPU via TPU-MLIR toolchain.
"""

import torch
import torch.nn as nn
import argparse
import os
import numpy as np


class SileroSTFT(nn.Module):
    """Custom STFT using Conv1d with learnable basis (ReflectionPad1d + Conv1d).
    pad=32 on each side (total 64) matches the JIT ReflectionPad1d config."""

    def __init__(self):
        super().__init__()
        self.padding = nn.ReflectionPad1d((0, 64))
        self.n_fft = 256
        self.hop_length = 128
        self.cutoff = self.n_fft // 2 + 1  # 129

    def forward(self, x):
        # x: [B, T]
        x = self.padding(x).unsqueeze(1)  # [B, 1, T+128]
        forward_transform = self.conv_real(x)  # [B, 258, T']
        real_part = forward_transform[:, :self.cutoff, :]   # [B, 129, T']
        imag_part = forward_transform[:, self.cutoff:, :]   # [B, 129, T']
        magnitude = torch.sqrt(real_part**2 + imag_part**2)  # [B, 129, T']
        return magnitude


class SileroEncoder(nn.Module):
    """4 Conv1d blocks with ReLU"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(129, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(64, 128, 3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        return x


class SileroDecoder(nn.Module):
    """LSTMCell (manual gate logic, ONNX-friendly) + final Conv1d → Sigmoid"""

    def __init__(self):
        super().__init__()
        self.input_size = 128
        self.hidden_size = 128
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(128, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h, c):
        # x: [B, 128], h: [B, 128], c: [B, 128]
        # Manual LSTMCell forward: avoids unsupported unsafe_chunk in ONNX export.
        # gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh  (shape: [B, 512])
        gates = (
            torch.mm(x, self.w_ih.t()) + self.b_ih +
            torch.mm(h, self.w_hh.t()) + self.b_hh
        )
        i, f, g, o = gates.chunk(4, dim=1)  # each [B, 128]
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_raw = o * torch.tanh(c_new)

        h_post = self.relu(self.dropout(h_raw))
        x_out = h_post.unsqueeze(-1)          # [B, 128, 1]
        x_out = self.conv(x_out)             # [B, 1, 1]
        x_out = self.sigmoid(x_out)          # [B, 1, 1]
        return x_out, h_raw, c_new


class SileroVADCore(nn.Module):
    """Clean Silero VAD model for ONNX export"""
    def __init__(self, jit_model_path=None):
        super().__init__()
        self.stft = SileroSTFT()
        self.encoder = SileroEncoder()
        self.decoder = SileroDecoder()

        if jit_model_path is not None:
            self._load_weights(jit_model_path)

    def _load_weights(self, jit_path):
        wrapper = torch.jit.load(jit_path, map_location='cpu')
        core = wrapper._model

        # STFT conv weight: forward_basis_buffer is [258, 1, 256]
        # which matches Conv1d(out_ch, in_ch, kernel) format
        self.stft.conv_real = nn.Conv1d(1, 258, 256, stride=128, padding=0, bias=False)
        self.stft.conv_real.weight.data.copy_(core.stft.forward_basis_buffer.data)

        # Encoder weights
        state_dict = core.state_dict()
        self.encoder.conv1.weight.data.copy_(state_dict['encoder.0.reparam_conv.weight'])
        self.encoder.conv1.bias.data.copy_(state_dict['encoder.0.reparam_conv.bias'])
        self.encoder.conv2.weight.data.copy_(state_dict['encoder.1.reparam_conv.weight'])
        self.encoder.conv2.bias.data.copy_(state_dict['encoder.1.reparam_conv.bias'])
        self.encoder.conv3.weight.data.copy_(state_dict['encoder.2.reparam_conv.weight'])
        self.encoder.conv3.bias.data.copy_(state_dict['encoder.2.reparam_conv.bias'])
        self.encoder.conv4.weight.data.copy_(state_dict['encoder.3.reparam_conv.weight'])
        self.encoder.conv4.bias.data.copy_(state_dict['encoder.3.reparam_conv.bias'])

        # Decoder weights
        self.decoder.w_ih = nn.Parameter(state_dict['decoder.rnn.weight_ih'])
        self.decoder.w_hh = nn.Parameter(state_dict['decoder.rnn.weight_hh'])
        self.decoder.b_ih = nn.Parameter(state_dict['decoder.rnn.bias_ih'])
        self.decoder.b_hh = nn.Parameter(state_dict['decoder.rnn.bias_hh'])
        self.decoder.conv.weight.data.copy_(state_dict['decoder.decoder.2.weight'])
        self.decoder.conv.bias.data.copy_(state_dict['decoder.decoder.2.bias'])

    def forward(self, x, h, c):
        # x: [B, 576]  (64 context + 512 audio samples @ 16kHz)
        # h, c: [B, 128]  LSTM hidden/cell states, initialized to 0 by host
        x = self.stft(x)         # [B, 129, 4]
        x = self.encoder(x)      # [B, 128, 1]
        x = x.reshape(-1, 128)   # [B, 128]  (T'=1 fixed, avoids dynamic squeeze)
        out, h_new, c_new = self.decoder(x, h, c)  # out: [B, 1, 1]
        out = out.reshape(-1, 1) # [B, 1]  (avoids dynamic squeeze/mean)
        return out, h_new, c_new


def export_model(jit_path, output_path, opset_version=16):
    model = SileroVADCore(jit_model_path=jit_path)
    model.eval()

    batch_size = 1
    context_size = 64
    num_samples = 512
    state_dim = 128

    dummy_x = torch.randn(batch_size, context_size + num_samples)
    dummy_h = torch.zeros(batch_size, state_dim)
    dummy_c = torch.zeros(batch_size, state_dim)

    print(f"Input  x : {dummy_x.shape}")
    print(f"       h : {dummy_h.shape}")
    print(f"       c : {dummy_c.shape}")

    # Verify with PyTorch
    with torch.no_grad():
        out, h_new, c_new = model(dummy_x, dummy_h, dummy_c)
    print(f"Output out : {out.shape}")
    print(f"       h   : {h_new.shape}")
    print(f"       c   : {c_new.shape}")

    # Also verify against JIT model
    wrapper = torch.jit.load(jit_path, map_location='cpu')
    core = wrapper._model
    state = torch.stack([dummy_h, dummy_c], dim=0)  # [2, 1, 128]
    with torch.no_grad():
        jit_out, jit_new_state = core(dummy_x, state)
    jit_h_new = jit_new_state[0]
    jit_c_new = jit_new_state[1]

    print(f"\nMax diff (out):    {(out - jit_out).abs().max():.6e}")
    print(f"Max diff (h):      {(h_new - jit_h_new).abs().max():.6e}")
    print(f"Max diff (c):      {(c_new - jit_c_new).abs().max():.6e}")

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_x, dummy_h, dummy_c),
        output_path,
        input_names=["x", "h", "c"],
        output_names=["out", "h_new", "c_new"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"\nExported ONNX model to: {output_path}")

    # Check
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check: PASSED")

    ops = sorted(set(n.op_type for n in onnx_model.graph.node))
    print(f"ONNX ops used ({len(ops)}): {ops}")

    # Check for unsupported ops by TPU-MLIR
    problematic = {'Size', 'If', 'Equal', 'Not', 'Shape', 'Gather'}
    found = problematic & set(ops)
    if found:
        print(f"WARNING: still contains potentially unsupported ops: {found}")
    else:
        print("No problematic control-flow ops found - good for TPU-MLIR!")

    # ONNX Runtime verification
    try:
        import onnxruntime
        sess = onnxruntime.InferenceSession(output_path)
        ort_inputs = {"x": dummy_x.numpy(), "h": dummy_h.numpy(), "c": dummy_c.numpy()}
        ort_outs = sess.run(None, ort_inputs)
        print(f"ONNX Runtime out: {ort_outs[0].shape}, h: {ort_outs[1].shape}, c: {ort_outs[2].shape}")
        print(f"Max diff (out) PT vs ORT: {abs(out.numpy() - ort_outs[0]).max():.6e}")
    except ImportError:
        print("onnxruntime not installed, skipping runtime verification")


def main():
    parser = argparse.ArgumentParser(description="Export clean Silero VAD ONNX")
    parser.add_argument("--jit", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                             "upstream/src/silero_vad/data/silero_vad.jit"))
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__), "silero_vad_core_clean.onnx"))
    parser.add_argument("--opset", type=int, default=16)
    args = parser.parse_args()
    export_model(args.jit, args.output, args.opset)


if __name__ == "__main__":
    main()