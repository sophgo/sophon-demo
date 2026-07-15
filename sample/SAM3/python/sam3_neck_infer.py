#!/usr/bin/env python3
"""
SAM3 Neck (FPN) TPU Inference
==============================
Runs the Sam3DualViTDetNeck FPN on TPU using SAIL.

Input:  ViT output [1, 1024, H, W] float32 (grid=36 for 504x504, grid=72 for 1008x1008)
Output: 4 multi-scale feature maps (shapes depend on input grid):
  - feat_s4:  [1, 256, 4H, 4W]  (scale 4.0)
  - feat_s2:  [1, 256, 2H, 2W]  (scale 2.0)
  - feat_s1:  [1, 256, H, W]    (scale 1.0)
  - feat_s05: [1, 256, H/2, W/2] (scale 0.5)

Position encodings are computed in numpy (sinusoidal) to save TPU compute.
"""

import sophon.sail as sail
import numpy as np
import time
import logging

logger = logging.getLogger("SAM3Neck")


class PositionEmbeddingSine:
    """
    Sinusoidal position encoding matching Sam3DualViTDetNeck's PositionEmbeddingSine.

    Generates position embeddings of shape [B, num_pos_feats, H, W] for each
    feature map size. These are used by the transformer encoder as spatial references.
    """

    def __init__(self, num_pos_feats: int = 256, temperature: int = 10000,
                 normalize: bool = True, scale: float = None):
        self.num_pos_feats = num_pos_feats // 2  # half for x, half for y
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * np.pi
        self._cache = {}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Feature map [B, C, H, W] (only H, W used)
        Returns:
            pos_enc: [B, num_pos_feats, H, W]
        """
        H, W = x.shape[2], x.shape[3]
        cache_key = (H, W)

        if cache_key in self._cache:
            pos = self._cache[cache_key]
            return np.repeat(pos[np.newaxis], x.shape[0], axis=0)

        # Generate grid
        y_embed = np.arange(1, H + 1, dtype=np.float32).reshape(1, -1, 1)
        y_embed = np.tile(y_embed, (1, 1, W))

        x_embed = np.arange(1, W + 1, dtype=np.float32).reshape(1, 1, -1)
        x_embed = np.tile(x_embed, (1, H, 1))

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Frequency bands
        dim_t = np.arange(self.num_pos_feats, dtype=np.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Position encoding
        pos_x = x_embed[:, :, :, np.newaxis] / dim_t[np.newaxis, np.newaxis, np.newaxis, :]
        pos_y = y_embed[:, :, :, np.newaxis] / dim_t[np.newaxis, np.newaxis, np.newaxis, :]

        # sin/cos interleaving
        pos_x = np.stack([np.sin(pos_x[:, :, :, 0::2]), np.cos(pos_x[:, :, :, 1::2])], axis=4)
        pos_x = pos_x.reshape(pos_x.shape[0], pos_x.shape[1], pos_x.shape[2], -1)

        pos_y = np.stack([np.sin(pos_y[:, :, :, 0::2]), np.cos(pos_y[:, :, :, 1::2])], axis=4)
        pos_y = pos_y.reshape(pos_y.shape[0], pos_y.shape[1], pos_y.shape[2], -1)

        # Concatenate and permute to NCHW
        pos = np.concatenate([pos_y, pos_x], axis=3)
        pos = pos.transpose(0, 3, 1, 2)

        self._cache[cache_key] = pos[0]
        return np.repeat(pos, x.shape[0], axis=0)


class SAM3Neck:
    """
    SAM3 ViT Neck (FPN) TPU inference.

    Takes ViT backbone output and produces multi-scale feature maps + position encodings.
    """

    def __init__(self, model_dir: str, precision: str = "f32", dev_id: int = 0):
        """
        Args:
            model_dir: Path to directory containing neck/*.bmodel files
            precision: "f32" or "f16"
            dev_id: TPU device ID
        """
        self.dev_id = dev_id
        self.precision = precision

        import os
        bmodel_path = f"{model_dir}/neck/sam3_neck_{precision}_1b.bmodel"
        if not os.path.exists(bmodel_path):
            raise FileNotFoundError(f"Neck BModel not found: {bmodel_path}")

        self.bmodel_path = bmodel_path
        size_mb = os.path.getsize(bmodel_path) / (1024 * 1024)
        logger.info(f"Neck: {size_mb:.0f}MB at {bmodel_path}")

        # Load engine once (small model, 35MB/21MB)
        self.engine = None
        self.graph_name = None
        self.input_name = None
        self.output_names = None

        # Position encoding (compute in numpy)
        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=256)

        # Timing
        self.tpu_time = 0

    def _ensure_engine(self):
        """Lazy-load engine."""
        if self.engine is None:
            self.engine = sail.Engine(self.bmodel_path, self.dev_id, sail.IOMode.SYSIO)
            self.graph_name = self.engine.get_graph_names()[0]
            self.input_name = self.engine.get_input_names(self.graph_name)[0]
            self.output_names = self.engine.get_output_names(self.graph_name)
            logger.info(f"Neck engine loaded: input={self.input_name}, outputs={self.output_names}")

    def __call__(self, vit_features: np.ndarray):
        """
        Run FPN neck on ViT output.

        Args:
            vit_features: [1, 1024, G, G] float32 from ViT backbone

        Returns:
            fpn_feats: list of 4 feature maps [feat_s4, feat_s2, feat_s1, feat_s05]
            fpn_pos: list of 4 position encodings
        """
        self._ensure_engine()

        t0 = time.time()
        output = self.engine.process(self.graph_name, {self.input_name: vit_features})
        self.tpu_time = time.time() - t0

        # Extract output tensors in order (shapes are G*4, G*2, G*1, G*0.5)
        feat_s4 = output[self.output_names[0]]   # [1, 256, G*4, G*4]
        feat_s2 = output[self.output_names[1]]   # [1, 256, G*2, G*2]
        feat_s1 = output[self.output_names[2]]   # [1, 256, G,   G  ]
        feat_s05 = output[self.output_names[3]]  # [1, 256, G/2, G/2]

        fpn_feats = [feat_s4, feat_s2, feat_s1, feat_s05]

        # Compute position encodings in numpy
        fpn_pos = [self.pos_encoder(f).astype(np.float32) for f in fpn_feats]

        return fpn_feats, fpn_pos

    def get_timing(self):
        return {"tpu_ms": self.tpu_time * 1000}

    def close(self):
        if self.engine is not None:
            del self.engine
            self.engine = None


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Neck TPU Inference Test")
    parser.add_argument("--model_dir", type=str, default="../models/BM1684X_504",
                        help="Path to compiled bmodel directory")
    parser.add_argument("--precision", type=str, default="f16",
                        choices=["f32", "f16"], help="Model precision")
    parser.add_argument("--dev_id", type=int, default=0, help="TPU device ID")
    parser.add_argument("--grid", type=int, default=36,
                        help="Feature grid size (36 for 504, 72 for 1008)")
    parser.add_argument("--repeat", type=int, default=10, help="Number of inference runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    neck = SAM3Neck(args.model_dir, args.precision, args.dev_id)

    # Create dummy ViT output
    dummy_vit_out = np.random.randn(1, 1024, args.grid, args.grid).astype(np.float32) * 0.1

    # Warmup
    logger.info("Warming up...")
    for _ in range(3):
        neck(dummy_vit_out)

    # Benchmark
    logger.info(f"Running {args.repeat} iterations...")
    times = []
    for _ in range(args.repeat):
        t0 = time.time()
        feats, poses = neck(dummy_vit_out)
        times.append(time.time() - t0)

    avg_time = np.mean(times) * 1000
    logger.info(f"Results:")
    logger.info(f"  Average time: {avg_time:.1f} ms")
    logger.info(f"  TPU time: {neck.get_timing()['tpu_ms']:.1f} ms")
    logger.info(f"  Feature shapes:")
    for i, (f, p) in enumerate(zip(feats, poses)):
        logger.info(f"    Scale[{i}]: feat={list(f.shape)}, pos={list(p.shape)}")

    neck.close()
