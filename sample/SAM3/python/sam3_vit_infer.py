# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
"""
SAM3 ViT Image Encoder Inference (Split Model)
Chains 5 TPU bmodel parts to run the full ViT backbone.

Pipeline (504x504):
  Image → Part 0 (preproc) → Part 1 (blocks 0-7) → Part 2 (blocks 8-15)
       → Part 3 (blocks 16-23) → Part 4 (blocks 24-31) → Features [1,1024,36,36]
"""

import sophon.sail as sail
import numpy as np
import time
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SAM3ViT")


class SAM3ViTEncoder:
    """
    SAM3 ViT backbone encoder using 5 split TPU bmodels (504x504 resolution).

    Model split:
      Part 0: patch_embed + pos_embed + ln_pre  (input: [1,3,504,504])
      Part 1: ViT blocks 0-7   (input/output: [1,36,36,1024])
      Part 2: ViT blocks 8-15  (input/output: [1,36,36,1024])
      Part 3: ViT blocks 16-23 (input/output: [1,36,36,1024])
      Part 4: ViT blocks 24-31 (input/output: [1,36,36,1024])

    Output is reshaped to [1, 1024, 36, 36] to match original SAM3 format.
    """

    def __init__(self, model_dir: str = None, precision: str = "f16", dev_id: int = 0,
                 resolution: int = 504, engines: list = None,
                 graph_names: list = None, input_names: list = None,
                 output_names: list = None, streaming: bool = False):
        """
        Args:
            model_dir: Path to directory containing vit/*.bmodel files
            precision: "f32", "f16", "w8f16", or "int8"
            dev_id: TPU device ID
            resolution: Input image resolution (504 or 1008)
            engines: Pre-loaded list of sail.Engine (pipeline integration)
            graph_names: Pre-loaded graph names
            input_names: Pre-loaded input names
            output_names: Pre-loaded output names
            streaming: SoC 流式加载（part1-4 load→run→free，峰值显存 = max 单 part）。
                int8_1008 落 SoC 3GB 需开此选项；PCIe 显存充裕时关掉更快（全常驻）。
        """
        self.dev_id = dev_id
        self.precision = precision
        self.model_dir = model_dir
        self.resolution = resolution
        self.grid = resolution // 14  # patch_size=14
        self.streaming = streaming

        if engines is not None:
            # Use externally-provided engines (pipeline integration)
            self.engines = engines
            self.graph_names = graph_names
            self.input_names = input_names
            self.output_names = output_names
            self._owns_engines = False
            self.streaming = False  # 外部引擎已全常驻，流式无意义
        else:
            # BModel 路径：int8 时 part0（patch_embed+preproc）无 int8 变体，回退 f16
            # （part0 仅 17MB，f16/int8 差 6MB，对 SoC 3GB 预算可忽略；用已验证 f16 基线）
            def _bmodel_path(part):
                p = "f16" if (precision == "int8" and part == 0) else precision
                return os.path.join(model_dir, "vit",
                                    f"sam3_vit_part{part}_{p}_1b.bmodel")
            self._bmodel_path = _bmodel_path

            self.engines = []
            self.graph_names = []
            self.input_names = []
            self.output_names = []
            if streaming:
                # 流式：只常驻 part0（小），part1-4 在 __call__ 里 load→run→free
                load_parts = [0]
            else:
                # 全常驻（SAM2 模式，PCIe 显存充裕）
                load_parts = range(5)
            for part in load_parts:
                bmodel = _bmodel_path(part)
                if not os.path.exists(bmodel):
                    raise FileNotFoundError(f"BModel not found: {bmodel}")
                size_mb = os.path.getsize(bmodel) / (1024 * 1024)
                logger.info(f"Part {part}: {size_mb:.0f}MB at {bmodel}")
                engine = sail.Engine(bmodel, dev_id, sail.IOMode.SYSIO)
                gname = engine.get_graph_names()[0]
                self.engines.append(engine)
                self.graph_names.append(gname)
                self.input_names.append(engine.get_input_names(gname)[0])
                self.output_names.append(engine.get_output_names(gname)[0])
            self._owns_engines = True

        # SAM3 preprocessing: normalize to [-1, 1]
        # Formula: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
        self.input_size = (resolution, resolution)

        # Timing stats
        self.tpu_times = []
        self.preprocess_time = 0

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for SAM3 ViT.

        Args:
            image: BGR image (H, W, 3) in uint8 [0, 255]

        Returns:
            Normalized tensor [1, 3, H, W] float32 in range [-1, 1]
        """
        import cv2
        t0 = time.time()

        # BGR → RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to target resolution
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Normalize: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0

        # HWC → NCHW
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :, :, :]

        self.preprocess_time = time.time() - t0
        return img

    def __call__(self, image: np.ndarray, return_intermediate: bool = False):
        """
        Run ViT encoder on input image.

        Args:
            image: BGR image (H, W, 3) uint8, or preprocessed tensor [1,3,H,W] float32
            return_intermediate: If True, also return intermediate outputs from each part

        Returns:
            features: [1, 1024, grid, grid] float32
            (optionally list of intermediate outputs)
        """
        expected_shape = (1, 3, self.resolution, self.resolution)
        # Check if already preprocessed
        if image.dtype == np.float32 and image.shape == expected_shape:
            x = image.astype(np.float32).copy()
        else:
            x = self.preprocess(image)

        self.tpu_times = []
        intermediates = [] if return_intermediate else None

        if self.streaming:
            # part0 常驻
            t0 = time.time()
            output = self.engines[0].process(
                self.graph_names[0], {self.input_names[0]: x})
            self.tpu_times.append(time.time() - t0)
            x = output[self.output_names[0]]
            if return_intermediate:
                intermediates.append(x.copy())
            # part1-4 流式：load → run → free（SoC 3GB fit，峰值 = max 单 part 1.35GB）
            for part in range(1, 5):
                bmodel = self._bmodel_path(part)
                t0 = time.time()
                engine = sail.Engine(bmodel, self.dev_id, sail.IOMode.SYSIO)
                gname = engine.get_graph_names()[0]
                iname = engine.get_input_names(gname)[0]
                oname = engine.get_output_names(gname)[0]
                output = engine.process(gname, {iname: x})
                self.tpu_times.append(time.time() - t0)
                x = output[oname]
                if return_intermediate:
                    intermediates.append(x.copy())
                del engine
        else:
            # 全常驻：链式跑 5 个预加载引擎（SAM2 模式，PCIe 显存充裕）
            for part in range(5):
                t0 = time.time()
                output = self.engines[part].process(
                    self.graph_names[part],
                    {self.input_names[part]: x})
                self.tpu_times.append(time.time() - t0)
                x = output[self.output_names[part]]
                if return_intermediate:
                    intermediates.append(x.copy())

        # Reshape from [1, grid, grid, 1024] to [1, 1024, grid, grid]
        g = self.grid
        x = x.reshape(1, g, g, 1024).transpose(0, 3, 1, 2)

        self.features = x

        if return_intermediate:
            return x, intermediates
        return x

    def get_timing(self):
        """Return timing breakdown."""
        return {
            "preprocess_ms": self.preprocess_time * 1000,
            "tpu_times_ms": [t * 1000 for t in self.tpu_times],
            "total_tpu_ms": sum(self.tpu_times) * 1000,
        }

    def print_timing(self):
        timing = self.get_timing()
        logger.info(f"Timing breakdown:")
        logger.info(f"  Preprocess:      {timing['preprocess_ms']:.1f}ms")
        for i, tt in enumerate(timing['tpu_times_ms']):
            logger.info(f"  Part {i}: tpu={tt:.1f}ms")
        logger.info(f"  Total TPU:       {timing['total_tpu_ms']:.1f}ms")

    def close(self):
        """Free all TPU engines (only if self-owned)."""
        if self._owns_engines:
            for engine in self.engines:
                del engine
            self.engines.clear()


# ============================================================
# Test / Demo
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 ViT Encoder Inference")
    parser.add_argument("--model_dir", type=str, default="../models/BM1684X_504",
                       help="Directory containing vit/*.bmodel files")
    parser.add_argument("--precision", type=str, default="f16",
                       choices=["f32", "f16", "w8f16", "int8"],
                       help="Model precision (int8 = 1008 SoC 量化版)")
    parser.add_argument("--dev_id", type=int, default=0,
                       help="TPU device ID")
    parser.add_argument("--resolution", type=int, default=504,
                       help="Input resolution")
    parser.add_argument("--streaming", action="store_true",
                       help="SoC 流式加载 part1-4（load→run→free，峰值显存=max 单 part）。"
                            "int8_1008 落 SoC 3GB 需开此选项")
    parser.add_argument("--image", type=str, default=None,
                       help="Input image path (optional, uses random if not set)")
    parser.add_argument("--warmup", type=int, default=3,
                       help="Warmup iterations")
    parser.add_argument("--loops", type=int, default=10,
                       help="Benchmark iterations")
    args = parser.parse_args()

    logger.info(f"Loading SAM3 ViT encoder ({args.precision}, {args.resolution}x{args.resolution})...")
    encoder = SAM3ViTEncoder(args.model_dir, args.precision, args.dev_id,
                             resolution=args.resolution, streaming=args.streaming)

    # Prepare input
    if args.image and os.path.exists(args.image):
        import cv2
        image = cv2.imread(args.image)
        logger.info(f"Loaded image: {image.shape}")
    else:
        # Generate random test image
        np.random.seed(42)
        image = np.random.randint(0, 256, (args.resolution, args.resolution, 3), dtype=np.uint8)
        logger.info(f"Using random test image [{args.resolution}x{args.resolution}] uint8")

    # Warmup
    logger.info(f"Warming up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        _ = encoder(image)

    # Benchmark
    logger.info(f"Benchmarking ({args.loops} runs)...")
    times = []
    for _ in range(args.loops):
        t0 = time.time()
        features = encoder(image)
        times.append(time.time() - t0)

    avg_time = np.mean(times) * 1000
    logger.info(f"Features shape: {features.shape}, norm: {np.linalg.norm(features):.2f}")
    logger.info(f"Average total time: {avg_time:.1f}ms")
    logger.info(f"Throughput: {1000/avg_time:.2f} FPS")
    encoder.print_timing()
