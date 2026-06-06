from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_pairs(ir_dir: str | Path, vi_dir: str | Path) -> list[str]:
    ir = Path(ir_dir)
    vi = Path(vi_dir)
    ir_names = sorted(p.name for p in ir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    vi_names = sorted(p.name for p in vi.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    missing_vi = sorted(set(ir_names) - set(vi_names))
    missing_ir = sorted(set(vi_names) - set(ir_names))
    if missing_vi or missing_ir:
        raise ValueError(f"Pair mismatch. missing_vi={missing_vi[:5]} missing_ir={missing_ir[:5]}")
    return ir_names


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).copy()


def to_float01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    if arr.max(initial=0) > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def gray_float(img: Image.Image) -> np.ndarray:
    if img.mode not in ("L", "RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "L":
        return to_float01(np.asarray(img))
    rgb = to_float01(np.asarray(img.convert("RGB")))
    return np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], 0.0, 1.0)


def rgb_float(img: Image.Image) -> np.ndarray:
    return to_float01(np.asarray(img.convert("RGB")))


def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def percentile_stretch(x: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo + 1e-8:
        return np.clip(x, 0.0, 1.0)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def gradient_mag(x: np.ndarray) -> np.ndarray:
    sx = ndimage.sobel(x, axis=1, mode="reflect")
    sy = ndimage.sobel(x, axis=0, mode="reflect")
    return np.sqrt(sx * sx + sy * sy).astype(np.float32)


def local_std(x: np.ndarray, size: int = 7) -> np.ndarray:
    mean = ndimage.uniform_filter(x, size=size, mode="reflect")
    mean2 = ndimage.uniform_filter(x * x, size=size, mode="reflect")
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0)).astype(np.float32)


def robust_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    s = percentile_stretch(source, 0.3, 99.7)
    r_mean = float(np.mean(reference))
    r_std = float(np.std(reference)) + 1e-6
    s_mean = float(np.mean(s))
    s_std = float(np.std(s)) + 1e-6
    out = (s - s_mean) / s_std * r_std + r_mean
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def recombine_y_with_vi_color(vi_img: Image.Image, fused_y: np.ndarray, saturation: float = 1.0) -> Image.Image:
    rgb = rgb_float(vi_img)
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    cb = (rgb[..., 2] - y) * 0.564
    cr = (rgb[..., 0] - y) * 0.713
    cb *= saturation
    cr *= saturation
    r = fused_y + cr / 0.713
    b = fused_y + cb / 0.564
    g = (fused_y - 0.299 * r - 0.114 * b) / 0.587
    out = np.stack([r, g, b], axis=-1)
    return Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")


def save_image_like(img: Image.Image, out_path: str | Path) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    suffix = out_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        img.convert("RGB").save(out_path, quality=100, subsampling=0, optimize=False)
    else:
        img.save(out_path)


def entropy_u8(gray: np.ndarray) -> float:
    g = np.clip(gray * 255.0 + 0.5, 0, 255).astype(np.uint8)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist / max(1.0, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 256) -> float:
    a8 = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8).ravel()
    b8 = np.clip(b * 255.0 + 0.5, 0, 255).astype(np.uint8).ravel()
    hist2d, _, _ = np.histogram2d(a8, b8, bins=bins, range=[[0, 255], [0, 255]])
    pxy = hist2d / max(1.0, hist2d.sum())
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return float((pxy[nz] * np.log2(pxy[nz] / (px_py[nz] + 1e-12))).sum())


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64).ravel()
    bb = b.astype(np.float64).ravel()
    aa -= aa.mean()
    bb -= bb.mean()
    den = math.sqrt(float((aa * aa).sum() * (bb * bb).sum())) + 1e-12
    return float((aa * bb).sum() / den)


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = ndimage.gaussian_filter(a, 1.5)
    mu_b = ndimage.gaussian_filter(b, 1.5)
    sig_a = ndimage.gaussian_filter(a * a, 1.5) - mu_a * mu_a
    sig_b = ndimage.gaussian_filter(b * b, 1.5) - mu_b * mu_b
    sig_ab = ndimage.gaussian_filter(a * b, 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (sig_a + sig_b + c2)
    return float(np.mean(num / (den + 1e-12)))


def iter_image_files(path: str | Path) -> Iterable[Path]:
    p = Path(path)
    return sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS)
