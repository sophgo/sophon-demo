#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""
KITTI stereo matching evaluation script (fixed version).

Supported metrics:
  KITTI 2012 standard:
    - 3-noc:  Outlier ratio in non-occluded regions (|error| > 3 px)
    - 3-all:  Outlier ratio in all regions         (|error| > 3 px)
    - EPE-noc: End-Point Error in non-occluded regions (mean |error|)
    - EPE-all: End-Point Error in all regions         (mean |error|)

  KITTI 2015 standard:
    - D1-bg:  Outlier ratio in non-occluded regions (|error| > 3 px AND > 5%)
    - D1-all: Outlier ratio in all regions          (|error| > 3 px AND > 5%)

Fixes over the original eval.py:
  Bug 1: KITTI ground-truth is a 3-channel 16-bit PNG where
         Channel 0 (R) = disparity (uint16, disparity = value / 256.0)
         Channel 2 (B) = validity mask (255 = valid, 0 = invalid)
         The original script loaded it with cv2.imread default (8-bit BGR),
         which both truncates 16-bit precision AND swaps channel order.
  Bug 2: Inference saves float32 disparity maps. cv2.imwrite converts them
         to uint16 PNG internally (value * 256). The original script loaded
         them with cv2.imread default (8-bit), losing both precision and scale.
"""

import numpy as np
import argparse
import os
import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def outlier_3px(disp_pred, disp_gt, valid_mask):
    """KITTI 2012 outlier: |error| > 3 px (single threshold).

    Returns:
        float - ratio of outlier pixels among valid pixels
    """
    error = np.abs(disp_gt - disp_pred)
    outlier_mask = (error > 3.0) & valid_mask
    num_valid = np.sum(valid_mask)
    if num_valid == 0:
        return 0.0
    return float(np.sum(outlier_mask)) / float(num_valid)


def outlier_d1(disp_pred, disp_gt, valid_mask):
    """KITTI 2015 D1 outlier: |error| > 3 px AND > 5% (dual threshold).

    Returns:
        float - ratio of outlier pixels among valid pixels
    """
    error = np.abs(disp_gt - disp_pred)
    outlier_mask = (error > 3.0) & (error / (np.abs(disp_gt) + 1e-8) > 0.05) & valid_mask
    num_valid = np.sum(valid_mask)
    if num_valid == 0:
        return 0.0
    return float(np.sum(outlier_mask)) / float(num_valid)


def epe(disp_pred, disp_gt, valid_mask):
    """End-Point Error: mean absolute disparity error over valid pixels.

    Returns:
        float - mean |disp_gt - disp_pred| in pixels
    """
    num_valid = np.sum(valid_mask)
    if num_valid == 0:
        return 0.0
    error = np.abs(disp_gt - disp_pred)
    return float(np.sum(error * valid_mask)) / float(num_valid)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_kitti_gt(gt_path):
    """Load a KITTI 2012 ground-truth disparity PNG.

    File format (3-channel, 16-bit PNG):
        Channel 0 (R) - disparity, stored as uint16; real_disp = value / 256.0
        Channel 1 (G) - unused (zero)
        Channel 2 (B) - validity flag: 255 = valid pixel, 0 = invalid

    Note: cv2.imread returns BGR order for 3-channel images, so after loading:
        img[:,:,0] -> original channel 2 (B) -> validity mask
        img[:,:,1] -> original channel 1 (G) -> unused
        img[:,:,2] -> original channel 0 (R) -> disparity values

    Returns:
        disp_gt:    (H, W) float32 - ground truth disparity
        valid_mask: (H, W) bool     - True where the pixel has valid GT
    """
    gt_img = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if gt_img is None:
        raise FileNotFoundError(f"Cannot read GT file: {gt_path}")

    if len(gt_img.shape) == 3 and gt_img.shape[2] >= 3:
        # 3-channel KITTI GT: after cv2 BGR conversion
        #   channel 2 = original R = disparity
        #   channel 0 = original B = validity mask
        disp_gt = gt_img[:, :, 2].astype(np.float32) / 256.0
        valid_mask = gt_img[:, :, 0] > 0
    elif len(gt_img.shape) == 2:
        # Fallback: single-channel disparity (non-KITTI format)
        disp_gt = gt_img.astype(np.float32) / 256.0
        valid_mask = disp_gt > 0
    else:
        raise ValueError(f"Unexpected GT shape: {gt_img.shape} for {gt_path}")

    return disp_gt, valid_mask


def load_prediction(pred_path):
    """Load a prediction disparity map saved by the inference script.

    The inference script saves float32 disparity via cv2.imwrite.
    OpenCV may save float32 as either 8-bit or 16-bit PNG depending on the
    build/version, so we auto-detect the bit depth:
      - uint8  -> values ARE the disparity (truncated to integer), use as-is
      - uint16 -> values = disparity * 256, divide by 256.0 to recover

    Returns:
        disp_pred: (H, W) float32 - predicted disparity
    """
    pred_img = cv2.imread(pred_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if pred_img is None:
        raise FileNotFoundError(f"Cannot read prediction file: {pred_path}")

    # Squeeze to single channel (H, W) if needed
    if len(pred_img.shape) == 3:
        pred_img = pred_img[:, :, 0]

    if pred_img.dtype == np.uint16:
        # 16-bit PNG: values = disparity * 256
        disp_pred = pred_img.astype(np.float32) / 256.0
    else:
        # 8-bit PNG (or other): values are raw disparity (truncated to integer)
        disp_pred = pred_img.astype(np.float32)

    return disp_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def argsparser():
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="KITTI stereo matching evaluation (supports 3-all & D1-all)"
    )
    parser.add_argument(
        '--results_path', type=str,
        default='../python/results/images',
        help='directory of predicted disparity images from inference script'
    )
    parser.add_argument(
        '--gt_path', type=str,
        default='../datasets/KITTI12/training',
        help='KITTI training directory (should contain disp_occ/ and disp_noc/)'
    )
    parser.add_argument(
        '--MAX_DISP', type=int, default=192,
        help='maximum disparity value (pixels with GT > this are excluded)'
    )
    return parser.parse_args()


def main():
    args = argsparser()

    disp_occ_dir = os.path.join(args.gt_path, 'disp_occ')
    disp_noc_dir = os.path.join(args.gt_path, 'disp_noc')
    has_occ = os.path.isdir(disp_occ_dir)
    has_noc = os.path.isdir(disp_noc_dir)

    if not has_occ and not has_noc:
        raise FileNotFoundError(
            f"Neither disp_occ/ nor disp_noc/ found under {args.gt_path}"
        )
    if not os.path.isdir(args.results_path):
        raise FileNotFoundError(f"Results directory not found: {args.results_path}")

    # Collect prediction files
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    pred_files = sorted([
        f for f in os.listdir(args.results_path)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if not pred_files:
        logger.warning("No prediction images found in %s", args.results_path)
        return

    logger.info("Evaluating %d predictions from: %s", len(pred_files), args.results_path)
    if has_occ:
        logger.info("GT disp_occ: %s", disp_occ_dir)
    if has_noc:
        logger.info("GT disp_noc: %s", disp_noc_dir)
    logger.info("MAX_DISP: %d", args.MAX_DISP)
    logger.info("-" * 60)

    # Accumulators for per-image metrics
    results = {
        '3-all': [], 'EPE-all': [], 'D1-all': [],   # from disp_occ
        '3-noc': [], 'EPE-noc': [], 'D1-bg': [],     # from disp_noc
    }
    skipped = 0
    first_image_logged = True

    for filename in pred_files:
        pred_path = os.path.join(args.results_path, filename)

        # --- Load prediction ---
        try:
            disp_pred = load_prediction(pred_path)
        except FileNotFoundError as e:
            logger.warning(str(e))
            skipped += 1
            continue

        # --- Evaluate against disp_occ (3-all / EPE-all / D1-all) ---
        if has_occ:
            gt_occ_path = os.path.join(disp_occ_dir, filename)
            if not os.path.exists(gt_occ_path):
                logger.warning("disp_occ GT not found, skipping: %s", gt_occ_path)
                skipped += 1
                continue
            try:
                disp_gt_occ, mask_occ = load_kitti_gt(gt_occ_path)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(str(e))
                skipped += 1
                continue

            if disp_pred.shape != disp_gt_occ.shape:
                logger.warning(
                    "Shape mismatch: pred %s vs GT %s for %s, skipping",
                    disp_pred.shape, disp_gt_occ.shape, filename
                )
                skipped += 1
                continue

            eval_mask_occ = mask_occ & (disp_gt_occ > 0) & (disp_gt_occ < args.MAX_DISP)

            # Debug: detailed diagnostics for the first successfully loaded image
            if not first_image_logged:
                logger.info("[DEBUG] First image: %s", filename)
                logger.info("[DEBUG]   pred file : %s", pred_path)

                # Check raw file bit depth
                raw = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    logger.info("[DEBUG]   pred raw dtype: %s, shape: %s",
                                raw.dtype, raw.shape)
                    if len(raw.shape) == 3:
                        raw_flat = raw[:, :, 0]
                    else:
                        raw_flat = raw
                    logger.info("[DEBUG]   pred raw range: [%s, %s], mean=%.2f",
                                raw_flat.min(), raw_flat.max(), raw_flat.mean())

                logger.info("[DEBUG]   pred  range: [%.2f, %.2f], mean=%.2f",
                            disp_pred.min(), disp_pred.max(), disp_pred.mean())
                logger.info("[DEBUG]   GT    range: [%.2f, %.2f], mean=%.2f",
                            disp_gt_occ.min(), disp_gt_occ.max(), disp_gt_occ.mean())
                logger.info("[DEBUG]   eval mask: %d valid pixels", np.sum(eval_mask_occ))

                # Show values at valid pixels only
                valid_pred = disp_pred[eval_mask_occ]
                valid_gt = disp_gt_occ[eval_mask_occ]
                if len(valid_gt) > 0:
                    logger.info("[DEBUG]   pred (valid only): mean=%.2f, median=%.2f",
                                np.mean(valid_pred), np.median(valid_pred))
                    logger.info("[DEBUG]   GT   (valid only): mean=%.2f, median=%.2f",
                                np.mean(valid_gt), np.median(valid_gt))
                    err = np.abs(valid_pred - valid_gt)
                    logger.info("[DEBUG]   |error| (valid only): mean=%.2f, median=%.2f",
                                np.mean(err), np.median(err))

                    # Show a few example pixel values side by side
                    step = max(1, len(valid_gt) // 5)
                    indices = list(range(0, len(valid_gt), step))[:5]
                    logger.info("[DEBUG]   Sample pixels (pred vs GT):")
                    for idx in indices:
                        logger.info("[DEBUG]     pred=%.2f  GT=%.2f  diff=%.2f",
                                    valid_pred[idx], valid_gt[idx],
                                    valid_pred[idx] - valid_gt[idx])

                logger.info("-" * 60)
                first_image_logged = True

            results['3-all'].append(outlier_3px(disp_pred, disp_gt_occ, eval_mask_occ))
            results['EPE-all'].append(epe(disp_pred, disp_gt_occ, eval_mask_occ))
            results['D1-all'].append(outlier_d1(disp_pred, disp_gt_occ, eval_mask_occ))

        # --- Evaluate against disp_noc (3-noc / EPE-noc / D1-bg) ---
        if has_noc:
            gt_noc_path = os.path.join(disp_noc_dir, filename)
            if not os.path.exists(gt_noc_path):
                # disp_noc missing for this image is not fatal if disp_occ exists
                if not has_occ:
                    skipped += 1
                continue
            try:
                disp_gt_noc, mask_noc = load_kitti_gt(gt_noc_path)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(str(e))
                if not has_occ:
                    skipped += 1
                continue

            if disp_pred.shape != disp_gt_noc.shape:
                continue

            eval_mask_noc = mask_noc & (disp_gt_noc > 0) & (disp_gt_noc < args.MAX_DISP)
            results['3-noc'].append(outlier_3px(disp_pred, disp_gt_noc, eval_mask_noc))
            results['EPE-noc'].append(epe(disp_pred, disp_gt_noc, eval_mask_noc))
            results['D1-bg'].append(outlier_d1(disp_pred, disp_gt_noc, eval_mask_noc))

    # --- Print results ---
    logger.info("-" * 60)
    n_eval = len(results['3-all']) if has_occ else len(results['3-noc'])
    logger.info("Evaluated images : %d", n_eval)
    logger.info("Skipped images   : %d", skipped)
    logger.info("")

    if has_occ:
        avg_3_all = np.mean(results['3-all'])
        avg_epe_all = np.mean(results['EPE-all'])
        avg_d1_all = np.mean(results['D1-all'])
        logger.info("=== KITTI 2012 Metrics (disp_occ, all regions) ===")
        logger.info("  3-all   : %.4f (%.2f%%)", avg_3_all, avg_3_all * 100)
        logger.info("  EPE-all : %.4f px", avg_epe_all)
        logger.info("")
        logger.info("=== KITTI 2015 Metrics (disp_occ, for reference) ===")
        logger.info("  D1-all  : %.4f (%.2f%%)", avg_d1_all, avg_d1_all * 100)
        logger.info("")

    if has_noc:
        avg_3_noc = np.mean(results['3-noc'])
        avg_epe_noc = np.mean(results['EPE-noc'])
        avg_d1_bg = np.mean(results['D1-bg'])
        logger.info("=== KITTI 2012 Metrics (disp_noc, non-occluded) ===")
        logger.info("  3-noc   : %.4f (%.2f%%)", avg_3_noc, avg_3_noc * 100)
        logger.info("  EPE-noc : %.4f px", avg_epe_noc)
        logger.info("")
        logger.info("=== KITTI 2015 Metrics (disp_noc, for reference) ===")
        logger.info("  D1-bg   : %.4f (%.2f%%)", avg_d1_bg, avg_d1_bg * 100)

    if not n_eval:
        logger.warning("No images were evaluated.")


if __name__ == "__main__":
    main()