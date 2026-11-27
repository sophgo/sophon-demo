"""
Self-contained NumPy post-processing for MoGe-2.

Reproduces the parts of ``MoGeModel.infer()`` that are *not* inside the ONNX/bmodel
graph: recovering the focal length and Z-shift from the affine point map, then
building metric depth, normalized intrinsics and the reprojected point map, plus
the depth/normal colorization used for visualization.

Only NumPy is strictly required. SciPy is used for the 1-D Levenberg-Marquardt fit
when available (this is what the official repo does via ``scipy.optimize.least_squares``);
otherwise a pure-NumPy golden-section fallback is used, which agrees to ~1e-6 with
SciPy on the recovered shift/focal.
"""

from typing import Optional, Tuple

import numpy as np

try:
    from scipy.optimize import least_squares
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is absent on the target boards
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# view-plane UV
# --------------------------------------------------------------------------- #
def normalized_view_plane_uv(width: int, height: int, aspect_ratio: Optional[float] = None,
                             dtype=np.float32) -> np.ndarray:
    """UV with left-top corner (-width/diag, -height/diag) and right-bottom (+width/diag, +height/diag)."""
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5
    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing='xy')
    return np.stack([u, v], axis=-1)


# --------------------------------------------------------------------------- #
# focal / shift recovery (1-D fit on the shift, closed-form focal)
# --------------------------------------------------------------------------- #
def _residual_focal_shift(uv, xy, z, shift):
    """Residual ``focal(shift) * xy/(z+shift) - uv`` with the closed-form focal at that shift."""
    shift = float(shift)
    zs = z + shift
    if np.any(zs <= 1e-6):
        return np.full((uv.size,), 1e3, dtype=np.float64)
    xy_proj = xy / zs[:, None]
    num = (xy_proj * uv).sum()
    den = (xy_proj * xy_proj).sum()
    if den <= 1e-12:
        return np.full((uv.size,), 1e3, dtype=np.float64)
    focal = num / den
    return ((focal * xy_proj - uv).ravel()).astype(np.float64)


def _residual_shift(uv, xy, z, focal, shift):
    """Residual ``focal * xy/(z+shift) - uv`` for a fixed focal."""
    shift = float(shift)
    zs = z + shift
    if np.any(zs <= 1e-6):
        return np.full((uv.size,), 1e3, dtype=np.float64)
    xy_proj = xy / zs[:, None]
    return ((focal * xy_proj - uv).ravel()).astype(np.float64)


def _gss_min(f, a, b, tol=1e-6, max_iter=200):
    """Golden-section search of a 1-D unimodal function on [a, b]."""
    gr = (5.0 ** 0.5 - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
        if abs(b - a) < tol * (abs(a) + abs(b) + 1e-6):
            break
    return (a + b) / 2.0


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray) -> Tuple[float, float]:
    """Solve ``min |focal * xy/(z+shift) - uv|`` w.r.t. shift and focal. Returns (shift, focal)."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    xy = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)[:, :2]
    z = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)[:, 2]

    if _HAS_SCIPY:
        sol = least_squares(lambda s: _residual_focal_shift(uv, xy, z, s), x0=np.array([0.0]),
                            ftol=1e-3, method='lm')
        shift = float(np.asarray(sol['x']).squeeze().astype(np.float32))
    else:
        cost = lambda s: float(_residual_focal_shift(uv, xy, z, s) @ _residual_focal_shift(uv, xy, z, s))
        lo = -z.min() + max(1e-3, 0.05 * abs(z.min()))
        hi = max(4.0, 4.0 * abs(z).max())
        if cost(hi) < cost((lo + hi) / 2.0):
            while cost(hi) < cost(hi - (hi - lo) / 2.0) and hi < 1e4:
                hi *= 2.0
        shift = _gss_min(cost, lo, hi)

    zs = z + shift
    xy_proj = xy / zs[:, None]
    focal = float(np.asarray((xy_proj * uv).sum() / (xy_proj * xy_proj).sum(), dtype=np.float32))
    return shift, focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float) -> float:
    """Solve ``min |focal * xy/(z+shift) - uv|`` w.r.t. shift only."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    xy = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)[:, :2]
    z = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)[:, 2]

    if _HAS_SCIPY:
        sol = least_squares(lambda s: _residual_shift(uv, xy, z, focal, s), x0=np.array([0.0]),
                            ftol=1e-3, method='lm')
        return float(np.asarray(sol['x']).squeeze().astype(np.float32))
    cost = lambda s: float(_residual_shift(uv, xy, z, focal, s) @ _residual_shift(uv, xy, z, focal, s))
    lo = -z.min() + max(1e-3, 0.05 * abs(z.min()))
    hi = max(4.0, 4.0 * abs(z).max())
    return _gss_min(cost, lo, hi)


def _nearest_resize(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize ``arr`` (channels first, last two dims H, W) to ``size``=(H, W).

    Uses the same indexing as ``torch.nn.functional.interpolate(mode='nearest')``
    (src = floor(dst * src_size / dst_size), clamped), which is what the official
    ``recover_focal_shift`` downsample does.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    th, tw = size
    ys = np.minimum((np.arange(th) * h // th).astype(np.int32), h - 1)
    xs = np.minimum((np.arange(tw) * w // tw).astype(np.int32), w - 1)
    return arr[..., ys[:, None], xs[None, :]]


def recover_focal_shift(points: np.ndarray, mask: Optional[np.ndarray] = None, focal: Optional[float] = None,
                        downsample_size: Tuple[int, int] = (64, 64)) -> Tuple[float, float]:
    """Recover focal (relative to half diagonal) and Z-shift from a (..., H, W, 3) point map.

    Returns (focal, shift). Matches ``moge.utils.geometry_torch.recover_focal_shift``.
    """
    points = np.asarray(points, dtype=np.float32)
    height, width = points.shape[-3], points.shape[-2]
    uv = normalized_view_plane_uv(width, height)

    points_lr = _nearest_resize(points.transpose(2, 0, 1), downsample_size).transpose(1, 2, 0)
    uv_lr = _nearest_resize(uv.transpose(2, 0, 1), downsample_size).transpose(1, 2, 0)

    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32) > 0.5
        mask_lr = _nearest_resize(mask, downsample_size)
        points_lr = points_lr[mask_lr]
        uv_lr = uv_lr[mask_lr]

    if points_lr.reshape(-1, 3).shape[0] < 2:
        return 1.0, 0.0

    if focal is None:
        shift, focal = solve_optimal_focal_shift(uv_lr, points_lr)
        return focal, shift
    shift = solve_optimal_shift(uv_lr, points_lr, focal)
    return focal, shift


# --------------------------------------------------------------------------- #
# intrinsics / fov
# --------------------------------------------------------------------------- #
def intrinsics_from_focal_center(fx, fy, cx, cy) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def fov_from_intrinsics(intrinsics: np.ndarray):
    return 2 * np.arctan(0.5 / intrinsics[..., 0, 0]), 2 * np.arctan(0.5 / intrinsics[..., 1, 1])


# --------------------------------------------------------------------------- #
# full end-to-end post-processing
# --------------------------------------------------------------------------- #
def end_to_end(points: np.ndarray, mask: np.ndarray, metric_scale: np.ndarray,
               force_projection: bool = True) -> dict:
    """Convert raw bmodel/ONNX outputs into metrically-scaled depth, intrinsics and points.

    Args:
        points: (H, W, 3) affine point map, already remapped by the network.
        mask:   (H, W) sigmoid mask (pixel validity in [0, 1]).
        metric_scale: float or array (already exp'd).
        force_projection: reproject the point map using the depth + intrinsics.

    Returns:
        dict with 'depth'/'points'/'intrinsics' (float32 arrays), 'fov_x_deg'/'fov_y_deg'
        and 'focal'/'shift' (floats).
    """
    points = np.asarray(points, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=np.float32)
    metric_scale = float(np.asarray(metric_scale).ravel()[0])

    H, W = points.shape[-3], points.shape[-2]
    aspect = W / H
    mask_binary = mask > 0.5

    focal, shift = recover_focal_shift(points, mask_binary)

    fx = focal / 2 * (1 + aspect ** 2) ** 0.5 / aspect
    fy = focal / 2 * (1 + aspect ** 2) ** 0.5
    intrinsics = intrinsics_from_focal_center(fx, fy, 0.5, 0.5)

    points[..., 2] += shift
    mask_binary &= points[..., 2] > 0
    depth = points[..., 2].copy()

    if force_projection:
        # pixel-center UV (matches utils3d `uv_map` -> `unproject_cv` in the official repo)
        uu, vv = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H, indexing='xy')
        pts = np.stack([(uu - 0.5) / fx * depth, (vv - 0.5) / fy * depth, depth], axis=-1)
    else:
        pts = points

    depth *= metric_scale
    pts *= metric_scale

    fov_x, fov_y = fov_from_intrinsics(intrinsics)
    return {
        'depth': depth,
        'points': pts,
        'intrinsics': intrinsics,
        'mask': mask_binary,
        'fov_x_deg': float(np.rad2deg(fov_x)),
        'fov_y_deg': float(np.rad2deg(fov_y)),
        'focal': float(np.asarray(focal, dtype=np.float32)),
        'shift': float(np.asarray(shift, dtype=np.float32)),
    }


# --------------------------------------------------------------------------- #
# visualization (NumPy only, no matplotlib on the target board)
# --------------------------------------------------------------------------- #
def _hsv_to_rgb(h, s, v):
    """Vectorized HSV -> RGB. h/s/v in [0, 1], returns (..., 3) uint8 in [0, 255]."""
    h = np.asarray(h) * 6.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    imod = i % 6
    r = np.choose(imod, [v, q, p, p, t, v])
    g = np.choose(imod, [t, v, v, q, p, p])
    b = np.choose(imod, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def colorize_depth(depth: np.ndarray, mask: Optional[np.ndarray] = None, max_quantile: float = 0.98,
                   min_quantile: float = 0.02) -> np.ndarray:
    """Colorize a metric depth map (H, W) -> (H, W, 3) uint8, using disparity for contrast."""
    depth = np.asarray(depth, dtype=np.float32).copy()
    if mask is None:
        valid = depth > 0
    else:
        valid = np.asarray(mask, dtype=bool) & (depth > 0)
    disp = np.where(valid, 1.0 / np.where(depth > 0, depth, np.nan), np.nan)
    lo, hi = np.nanquantile(disp, min_quantile), np.nanquantile(disp, max_quantile)
    if hi <= lo:
        hi = lo + 1e-6
    t = np.clip((disp - lo) / (hi - lo), 0.0, 1.0)
    # blue (240 deg) -> red (0 deg); t=0 far (small disparity) -> t=1 near
    h = (1.0 - t) * (240.0 / 360.0)
    colored = _hsv_to_rgb(h, np.ones_like(t), np.where(valid, 1.0, 0.0))
    colored = (colored * 255.0).astype(np.uint8)
    return colored


def colorize_normal(normal: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Colorize a normal map (H, W, 3) -> (H, W, 3) uint8 (same convention as moge.utils.vis)."""
    normal = np.asarray(normal, dtype=np.float32).copy()
    if mask is not None:
        normal = np.where(np.asarray(mask, dtype=bool)[..., None], normal, 0.0)
    normal = normal * np.array([0.5, -0.5, -0.5], dtype=np.float32) + 0.5
    return (normal.clip(0, 1) * 255).astype(np.uint8)