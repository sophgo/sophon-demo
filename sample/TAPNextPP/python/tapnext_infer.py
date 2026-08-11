#!/usr/bin/env python3
# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
"""TAPNext++ point-tracking inference on Sophon BM1688 via SAIL.

TAPNext++ is a recurrent point tracker: a ViT backbone mixed across time by
RG-LRU (gated linear recurrent unit) layers.  The model is split into two
graphs that the host chains in a loop:

  init graph  (frame_0, query_points)  -> tracks, vis, 24 cache tensors
  step graph  (frame_k, step, query_points, *caches) -> tracks, vis, 24 caches

The 24 cache tensors (12 blocks x (rg_lru_state, conv1d_state)) carry the
recurrent state and are fed back each step.  This script loads both bmodels,
runs the init/step loop, and collects per-frame point trajectories.

Preprocessing is host-side (the bmodel was compiled without --mean/--scale):
  BGR -> RGB, bilinear resize to 256x256, x / 127.5 - 1.0  ->  [-1, 1] float32

Usage:
  # track a single point (y=128, x=128) born on frame 0 in a video
  python tapnext_infer.py --video input.mp4 --query 128,128 \
      --init_bmodel ../models/BM1688/tapnext_init_int8_1b.bmodel \
      --step_bmodel ../models/BM1688/tapnext_step_int8_1b.bmodel

  # track points from a query file (JSON: [[t,y,x], ...])
  python tapnext_infer.py --video input.mp4 --query_file queries.json \
      --init_bmodel ... --step_bmodel ...
"""
import os
import sys
import gc
import time
import json
import argparse
import logging

import cv2
import numpy as np
import sophon.sail as sail

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TAPNextPP")

MODEL_SIZE = 256          # input resolution HxW
NUM_BLOCKS = 12           # ViT blocks, each contributes (rg_lru, conv1d)


def _log_mem(tag):
    """Log CPU RSS (VmRSS) and TPU memory (bm-smi) at a checkpoint.

    Used to pinpoint whether OOM (exit 137) is CPU-side or TPU-side on SE9.
    Falls back silently if /proc or bm-smi is unavailable (e.g. x86 host).
    """
    # CPU RSS
    rss_mb = -1
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                rss_mb = int(line.split()[1]) // 1024  # kB -> MB
                break
    except Exception:
        pass
    # TPU memory via bm-smi
    tpu_info = "(unavailable)"
    try:
        import subprocess as _sp
        out = _sp.run(["bm-smi"], capture_output=True, text=True, timeout=5)
        for line in out.stdout.splitlines():
            if "Memory" in line and "Free" in line:
                tpu_info = line.strip()
                break
    except Exception:
        pass
    logger.info("[mem] %-24s CPU RSS=%d MB  TPU: %s", tag, rss_mb, tpu_info)


class TAPNextPP:
    """Two-graph recurrent tracker.

    init_engine : (frame, query_points) -> tracks, vis, 24 caches
    step_engine : (frame, step, query_points, *24 caches) -> tracks, vis, 24 caches

    The SE9 SoC has only ~850 MB system RAM, too small to hold both bmodels
    (363 MB + 309 MB) at once.  Engines are therefore loaded and freed
    sequentially inside ``track()``: init engine runs frame 0 and is released
    before the step engine is loaded for the recurrence loop.
    """

    def __init__(self, init_bmodel, step_bmodel, dev_id=0):
        self.dev_id = dev_id
        self.init_bmodel = init_bmodel
        self.step_bmodel = step_bmodel

        # cache input/output name mapping: init outputs "new_rg_lru_0" ->
        # step inputs "rg_lru_0".  Build the ordered list of 24 base names.
        self.cache_out_bases = [f"new_rg_lru_{i}" for i in range(NUM_BLOCKS)] + \
                               [f"new_conv1d_{i}" for i in range(NUM_BLOCKS)]
        self.cache_in_names = [f"rg_lru_{i}" for i in range(NUM_BLOCKS)] + \
                              [f"conv1d_{i}" for i in range(NUM_BLOCKS)]

        self.pre_time = 0.0
        self.init_time = 0.0
        self.step_time = 0.0
        self.post_time = 0.0

    @staticmethod
    def _build_out_map(engine, graph):
        """Map base ONNX output names to actual bmodel output names.

        TPU-MLIR renames outputs to ``<onnx_name>_<op>_f32``.  We match by
        prefix so the rest of the code can use the clean ONNX names.
        """
        actual = engine.get_output_names(graph)
        out_map = {}
        for bname in ["tracks", "vis_logits"] + \
                     [f"new_rg_lru_{i}" for i in range(NUM_BLOCKS)] + \
                     [f"new_conv1d_{i}" for i in range(NUM_BLOCKS)]:
            matches = [a for a in actual if a.startswith(bname)]
            if not matches:
                raise KeyError(f"bmodel output for '{bname}' not found in {actual}")
            out_map[bname] = matches[0]
        return out_map

    # ------------------------------------------------------------------ #
    #  SAIL helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _run(engine, graph, arrays):
        """Run one graph with SYSIO (numpy in -> numpy out).

        ``engine.process`` with SYSIO takes ``{name: np.ndarray}`` and returns
        ``{name: np.ndarray}`` directly — no sail.Tensor needed.  Inputs must
        be contiguous float32; we avoid copies when they already are (the 24
        recurrent caches are ~144 MB — copying them on every step would double
        peak memory and OOM the SE9's 850 MB RAM).
        """
        feed = {}
        for name, arr in arrays.items():
            arr = np.ascontiguousarray(arr)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            feed[name] = arr
        return engine.process(graph, feed)

    def _load_engine(self, bmodel_path):
        """Load a SAIL engine + graph + output name map."""
        engine = sail.Engine(bmodel_path, self.dev_id, sail.IOMode.SYSIO)
        graph = engine.get_graph_names()[0]
        out_map = self._build_out_map(engine, graph)
        return engine, graph, out_map

    def _init_subprocess(self, frame0, query_points):
        """Run the init graph in a child process to guarantee memory reclaim.

        Saves frame0 + query to a temp npz, spawns ``python tapnext_infer.py
        --init_only`` which loads the init bmodel, runs frame 0, and writes
        tracks[0], vis[0], and the 24 caches to a result npz.  When the child
        exits the OS reclaims all SAIL coefficient buffers (~363 MB on SE9),
        which ``del engine + gc.collect()`` does not return in-process.

        Returns ``(tracks0, vis0, cache_handle)`` where ``cache_handle`` is a
        *lazy* ``np.load`` NpzFile.  The 24 caches (~144 MB) are NOT read yet
        — the caller must extract them AFTER the step engine is loaded to avoid
        holding caches + step bmodel in RAM simultaneously on the 851 MB SE9.
        """
        tmp_in = "/tmp/_tapnext_init_in.npz"
        tmp_out = "/tmp/_tapnext_init_out.npz"
        np.savez(tmp_in, frame0=frame0, query_points=query_points)

        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--init_only",
            "--init_bmodel", self.init_bmodel,
            "--dev_id", str(self.dev_id),
            "--_init_input", tmp_in,
            "--_init_output", tmp_out,
        ]
        env = dict(os.environ)
        env.setdefault("LD_LIBRARY_PATH",
                       "/opt/sophon/libsophon-current/lib")
        import subprocess
        subprocess.run(cmd, check=True, env=env)

        os.remove(tmp_in)
        # lazy load: reading tracks0/vis0 does NOT materialize the 144 MB of
        # caches — they stay mmap'd until cache_handle["cache_N"] is accessed.
        cache_handle = np.load(tmp_out)
        tracks0 = cache_handle["tracks0"]   # [Q, 2]  (tiny, copied)
        vis0 = cache_handle["vis0"]         # [Q]     (tiny, copied)
        return tracks0, vis0, cache_handle, tmp_out

    # ------------------------------------------------------------------ #
    #  Preprocessing
    # ------------------------------------------------------------------ #
    def preprocess(self, frame_bgr):
        """BGR uint8 image -> [1,3,256,256] float32 in [-1, 1]."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
        chw = resized.transpose(2, 0, 1).astype(np.float32)   # [3,H,W]
        normalized = chw / 127.5 - 1.0                          # [-1, 1]
        return normalized[np.newaxis]                           # [1,3,H,W]

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    def init_track(self, frame, query_points):
        """Run init graph on frame 0.

        Args:
            frame: [1,3,256,256] float32 preprocessed
            query_points: [1,Q,3] float32 (t, y, x) in model pixels
        Returns:
            tracks: [1,1,Q,2], vis: [1,1,Q,1], caches: list of 24 np arrays
        """
        arrays = {"frame": frame, "query_points": query_points}
        outs = self._run(self._init_engine, self._init_graph, arrays)
        m = self._init_out_map
        tracks = outs[m["tracks"]]
        vis = outs[m["vis_logits"]]
        caches = [outs[m[b]] for b in self.cache_out_bases]
        return tracks, vis, caches

    def step_track(self, frame, step, query_points, caches):
        """Run step graph on frame k.

        Args:
            frame: [1,3,256,256] float32 preprocessed
            step: int frame counter (1-based)
            query_points: [1,Q,3] float32
            caches: list of 24 np arrays from previous step
        Returns:
            tracks, vis, new_caches
        """
        arrays = {
            "frame": frame,
            "step": np.array([step], dtype=np.float32),
            "query_points": query_points,
        }
        for name, arr in zip(self.cache_in_names, caches):
            arrays[name] = arr
        outs = self._run(self._step_engine, self._step_graph, arrays)
        m = self._step_out_map
        tracks = outs[m["tracks"]]
        vis = outs[m["vis_logits"]]
        new_caches = [outs[m[b]] for b in self.cache_out_bases]
        return tracks, vis, new_caches

    # ------------------------------------------------------------------ #
    #  Full rollout
    # ------------------------------------------------------------------ #
    def track(self, frames, query_points):
        """Track query points across a sequence of BGR frames.

        Args:
            frames: list of BGR uint8 numpy arrays (HxWx3)
            query_points: [1,Q,3] float32 (t, y, x) in model pixels
        Returns:
            all_tracks: [T, Q, 2] float32 — per-frame (y, x) in model pixels
            all_vis: [T, Q] float32 — visibility logits (sigmoid > 0.5 = visible)
        """
        n_frames = len(frames)
        all_tracks = []
        all_vis = []

        # --- frame 0: init graph (run in a subprocess) ---
        # The SE9 has only ~850 MB system RAM.  The init bmodel (~363 MB) and
        # step bmodel (~310 MB) cannot be loaded simultaneously, and ``del
        # engine + gc.collect()`` does NOT return the SAIL C++ coefficient
        # buffers to the OS (the malloc arena holds them).  Running init in a
        # child process guarantees all init memory is reclaimed when the child
        # exits, leaving enough headroom to load the step engine.
        t0 = time.time()
        frame0 = self.preprocess(frames[0])
        self.pre_time += time.time() - t0

        logger.info("Running init graph (subprocess): %s", self.init_bmodel)
        t0 = time.time()
        tracks0, vis0, cache_handle, tmp_out = \
            self._init_subprocess(frame0, query_points)
        self.init_time += time.time() - t0

        t0 = time.time()
        all_tracks.append(tracks0)      # [Q, 2]
        all_vis.append(vis0)            # [Q]
        self.post_time += time.time() - t0

        # --- frames 1..N: step graph ---
        # Load the step engine BEFORE materializing the 144 MB of caches from
        # cache_handle, so peak RAM is max(step_bmodel, caches) not their sum.
        logger.info("Loading step bmodel: %s", self.step_bmodel)
        _log_mem("before step load")
        self._step_engine, self._step_graph, self._step_out_map = \
            self._load_engine(self.step_bmodel)
        _log_mem("after step load")

        # now safe to load the caches
        caches = [cache_handle[f"cache_{i}"]
                  for i in range(len(self.cache_out_bases))]
        del cache_handle
        os.remove(tmp_out)
        _log_mem("after cache materialize")

        for k in range(1, n_frames):
            t0 = time.time()
            frame_k = self.preprocess(frames[k])
            self.pre_time += time.time() - t0

            t0 = time.time()
            tracks, vis, caches = self.step_track(
                frame_k, k, query_points, caches)
            self.step_time += time.time() - t0

            t0 = time.time()
            all_tracks.append(tracks[0, 0])
            all_vis.append(vis[0, 0, :, 0])
            self.post_time += time.time() - t0

            if k % 10 == 0:
                logger.info("  step %d/%d", k, n_frames - 1)

        # free step engine
        del self._step_engine
        gc.collect()

        return np.array(all_tracks), np.array(all_vis)

    def reset_timer(self):
        self.pre_time = self.init_time = self.step_time = self.post_time = 0.0


# ---------------------------------------------------------------------- #
#  I/O helpers
# ---------------------------------------------------------------------- #
def load_video(path, max_frames=0):
    """Read video frames as BGR uint8 list."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def load_image_dir(path, max_frames=0):
    """Read sorted image frames from a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(f for f in os.listdir(path)
                   if os.path.splitext(f)[1].lower() in exts)
    frames = []
    for f in files:
        img = cv2.imread(os.path.join(path, f))
        if img is not None:
            frames.append(img)
            if max_frames and len(frames) >= max_frames:
                break
    return frames


def parse_queries(query_str, query_file, model_size):
    """Parse query points into [1, Q, 3] float32 (t, y, x in model pixels).

    --query "y1,x1;y2,x2"  (t=0 assumed, coordinates in original image pixels
                            scaled to model_size by --img-size)
    --query_file queries.json  [[t,y,x,...], ...] in model pixels directly
    """
    if query_file:
        pts = json.loads(open(query_file).read())
        qp = np.array(pts, dtype=np.float32)
        if qp.ndim == 1:
            qp = qp[np.newaxis]
        return qp[np.newaxis]          # [1, Q, >=3]  (first 3 cols used)

    # parse "y1,x1;y2,x2"
    points = []
    for part in query_str.split(";"):
        y, x = [float(v) for v in part.split(",")]
        points.append([0.0, y, x])
    qp = np.array(points, dtype=np.float32)
    return qp[np.newaxis]              # [1, Q, 3]


def save_results(tracks, vis, out_path, model_size):
    """Save trajectories as JSON + visualization-ready npz."""
    np.savez(out_path + ".npz", tracks=tracks, vis=vis, model_size=model_size)
    # JSON: list of per-frame {y, x, visible}
    results = []
    for t in range(len(tracks)):
        frame_res = []
        for q in range(tracks.shape[1]):
            # tracks layout is [y, x]: the model does cat([tracks_x, tracks_y])
            # but tracks_x/tracks_y are misnamed — the grid_sample H/W swap in
            # embed_queries makes tracks_x the row (y) and tracks_y the col (x).
            frame_res.append({
                "y": float(tracks[t, q, 0]),
                "x": float(tracks[t, q, 1]),
                "visible": bool(1.0 / (1.0 + np.exp(-vis[t, q])) > 0.5),
            })
        results.append(frame_res)
    with open(out_path + ".json", "w") as f:
        json.dump(results, f, indent=2)


def visualize(frames, tracks, vis, model_size, out_path, fps=30):
    """Draw tracked points on frames and write a video."""
    h, w = frames[0].shape[:2]
    sy, sx = h / model_size, w / model_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
              (0, 255, 255), (255, 0, 255)]
    for t, frame in enumerate(frames):
        vis_frame = frame.copy()
        for q in range(tracks.shape[1]):
            color = colors[q % len(colors)]
            # draw trajectory up to frame t (tracks layout is [y, x])
            for tt in range(max(0, t - 20), t + 1):
                yy, xx = tracks[tt, q, 0] * sy, tracks[tt, q, 1] * sx
                if 0 <= yy < h and 0 <= xx < w:
                    cv2.circle(vis_frame, (int(xx), int(yy)), 2, color, -1)
            # current point
            yy, xx = tracks[t, q, 0] * sy, tracks[t, q, 1] * sx
            visible = 1.0 / (1.0 + np.exp(-vis[t, q])) > 0.5
            if visible and 0 <= yy < h and 0 <= xx < w:
                cv2.circle(vis_frame, (int(xx), int(yy)), 5, color, 2)
        writer.write(vis_frame)
    writer.release()


# ---------------------------------------------------------------------- #
#  Main
# ---------------------------------------------------------------------- #
def main(args):
    # --- init-only subprocess mode ---
    # Invoked by _init_subprocess(): load init bmodel, run frame 0, save caches.
    if args.init_only:
        data = np.load(args._init_input)
        frame0 = data["frame0"]
        query_points = data["query_points"]
        del data

        tracker = TAPNextPP(args.init_bmodel, args.init_bmodel, args.dev_id)
        logger.info("Loading init bmodel: %s", args.init_bmodel)
        _log_mem("before init load")
        tracker._init_engine, tracker._init_graph, tracker._init_out_map = \
            tracker._load_engine(args.init_bmodel)
        _log_mem("after init load")
        tracks, vis, caches = tracker.init_track(frame0, query_points)
        _log_mem("after init inference")

        save_dict = {
            "tracks0": tracks[0, 0],   # [Q, 2]
            "vis0": vis[0, 0, :, 0],   # [Q]
        }
        for i, c in enumerate(caches):
            save_dict[f"cache_{i}"] = c
        np.savez(args._init_output, **save_dict)
        logger.info("init done, caches saved to %s", args._init_output)
        return

    # --- load frames ---
    if args.input is None:
        raise ValueError("--input is required (video file or image directory)")
    if os.path.isdir(args.input):
        frames = load_image_dir(args.input, args.max_frames)
    else:
        frames = load_video(args.input, args.max_frames)
    logger.info("Loaded %d frames from %s", len(frames), args.input)
    if not frames:
        raise RuntimeError("no frames loaded")

    # --- scale query coordinates from original image to model pixels ---
    h, w = frames[0].shape[:2]
    query_points = parse_queries(args.query, args.query_file, MODEL_SIZE)
    if not args.query_file:
        # scale y, x from image pixels to model pixels
        query_points[0, :, 1] *= MODEL_SIZE / h
        query_points[0, :, 2] *= MODEL_SIZE / w
    logger.info("Query points (model pixels): %s", query_points)

    # --- run tracking ---
    tracker = TAPNextPP(args.init_bmodel, args.step_bmodel, args.dev_id)
    t_start = time.time()
    tracks, vis = tracker.track(frames, query_points)
    total = time.time() - t_start
    n = len(frames)

    logger.info("Tracked %d points across %d frames in %.2f s", tracks.shape[1], n, total)
    logger.info("------------------ Timing (per frame) ----------------------")
    logger.info("preprocess : %.2f ms", tracker.pre_time / n * 1000)
    logger.info("init       : %.2f ms", tracker.init_time * 1000)
    logger.info("step       : %.2f ms", tracker.step_time / max(n - 1, 1) * 1000)
    logger.info("postprocess: %.2f ms", tracker.post_time / n * 1000)
    logger.info("total      : %.2f ms/frame", total / n * 1000)

    # --- save results ---
    os.makedirs(args.output_dir, exist_ok=True)
    out_base = os.path.join(args.output_dir, "tracks")
    save_results(tracks, vis, out_base, MODEL_SIZE)
    logger.info("Results saved to %s.{npz,json}", out_base)

    if args.visualize:
        vis_path = os.path.join(args.output_dir, "tracked.mp4")
        visualize(frames, tracks, vis, MODEL_SIZE, vis_path)
        logger.info("Visualization saved to %s", vis_path)


def argsparser():
    p = argparse.ArgumentParser(prog=__file__, description="TAPNext++ inference on Sophon BM1688")
    p.add_argument("--input", type=str, default=None,
                   help="video file or image directory")
    p.add_argument("--init_bmodel", type=str,
                   default="../models/BM1688/tapnext_init_fp16_1b.bmodel",
                   help="init graph bmodel")
    p.add_argument("--step_bmodel", type=str,
                   default="../models/BM1688/tapnext_step_fp16_1b.bmodel",
                   help="step graph bmodel")
    p.add_argument("--dev_id", type=int, default=0, help="TPU device id")
    p.add_argument("--query", type=str, default="128,128",
                   help='query points "y1,x1;y2,x2" in image pixels (t=0)')
    p.add_argument("--query_file", type=str, default=None,
                   help="JSON file with query points [[t,y,x],...] in model pixels")
    p.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--output_dir", type=str, default="./results", help="output directory")
    p.add_argument("--visualize", action="store_true", help="write tracked video")
    # internal: init-only subprocess mode (see _init_subprocess)
    p.add_argument("--init_only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_init_input", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_init_output", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


if __name__ == "__main__":
    main(argsparser())
    print("all done.")
