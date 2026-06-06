from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from fusion_common import (
    ensure_dir,
    gradient_mag,
    gray_float,
    list_pairs,
    load_image,
    local_std,
    normalize01,
    percentile_stretch,
    recombine_y_with_vi_color,
    robust_match,
    save_image_like,
)


PRESETS: dict[str, dict[str, float | str]] = {
    "balanced": {
        "base_ir": 0.44,
        "saliency_gain": 0.34,
        "thermal_power": 0.85,
        "detail_strength": 0.82,
        "detail_ir_bias": 1.10,
        "contrast_low": 0.4,
        "contrast_high": 99.6,
        "gamma": 0.96,
        "local_contrast": 0.12,
        "sharpen": 0.12,
        "color_saturation": 0.92,
        "color_mode": "rgb",
    },
    "structure": {
        "base_ir": 0.38,
        "saliency_gain": 0.26,
        "thermal_power": 1.05,
        "detail_strength": 0.62,
        "detail_ir_bias": 0.95,
        "contrast_low": 0.8,
        "contrast_high": 99.2,
        "gamma": 1.00,
        "local_contrast": 0.04,
        "sharpen": 0.05,
        "color_saturation": 0.85,
        "color_mode": "rgb",
    },
    "detail": {
        "base_ir": 0.46,
        "saliency_gain": 0.40,
        "thermal_power": 0.78,
        "detail_strength": 1.04,
        "detail_ir_bias": 1.28,
        "contrast_low": 0.25,
        "contrast_high": 99.75,
        "gamma": 0.92,
        "local_contrast": 0.18,
        "sharpen": 0.20,
        "color_saturation": 0.96,
        "color_mode": "rgb",
    },
    "entropy": {
        "base_ir": 0.48,
        "saliency_gain": 0.32,
        "thermal_power": 0.90,
        "detail_strength": 0.90,
        "detail_ir_bias": 1.08,
        "contrast_low": 0.15,
        "contrast_high": 99.85,
        "gamma": 0.88,
        "local_contrast": 0.22,
        "sharpen": 0.12,
        "color_saturation": 1.00,
        "color_mode": "rgb",
    },
    "thermal": {
        "base_ir": 0.55,
        "saliency_gain": 0.42,
        "thermal_power": 0.70,
        "detail_strength": 0.78,
        "detail_ir_bias": 1.30,
        "contrast_low": 0.4,
        "contrast_high": 99.6,
        "gamma": 0.95,
        "local_contrast": 0.10,
        "sharpen": 0.10,
        "color_saturation": 0.82,
        "color_mode": "rgb",
    },
    "gray_rank": {
        "base_ir": 0.48,
        "saliency_gain": 0.38,
        "thermal_power": 0.82,
        "detail_strength": 0.95,
        "detail_ir_bias": 1.20,
        "contrast_low": 0.25,
        "contrast_high": 99.75,
        "gamma": 0.92,
        "local_contrast": 0.16,
        "sharpen": 0.16,
        "color_saturation": 0.0,
        "color_mode": "gray",
    },
}


def load_params(preset: str, params_json: str | None = None) -> dict[str, float | str]:
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset {preset}. Choices: {sorted(PRESETS)}")
    params = dict(PRESETS[preset])
    if params_json:
        override = json.loads(Path(params_json).read_text(encoding="utf-8"))
        params.update(override)
    return params


def fuse_pair(ir_img: Image.Image, vi_img: Image.Image, params: dict[str, float | str]) -> Image.Image:
    ir0 = gray_float(ir_img)
    vi_y0 = gray_float(vi_img)

    ir = robust_match(ir0, vi_y0)
    vi_y = percentile_stretch(vi_y0, 0.3, 99.7)

    ir_low = ndimage.gaussian_filter(ir, sigma=3.0, mode="reflect")
    vi_low = ndimage.gaussian_filter(vi_y, sigma=3.0, mode="reflect")
    ir_base = ndimage.gaussian_filter(ir, sigma=9.0, mode="reflect")

    g_ir = normalize01(gradient_mag(ir))
    g_vi = normalize01(gradient_mag(vi_y))
    sal_dark_bright = normalize01(np.abs(ir - ir_base))
    sal_std = normalize01(local_std(ir, 9))
    saliency = normalize01(0.48 * sal_dark_bright + 0.34 * g_ir + 0.18 * sal_std)
    saliency = ndimage.gaussian_filter(saliency, sigma=1.2, mode="reflect")
    saliency = np.power(np.clip(saliency, 0, 1), float(params["thermal_power"]))

    base_ir = float(params["base_ir"])
    saliency_gain = float(params["saliency_gain"])
    w_low = base_ir + saliency_gain * saliency + 0.10 * (g_ir - g_vi)
    w_low = ndimage.gaussian_filter(np.clip(w_low, 0.05, 0.94), sigma=2.0, mode="reflect")
    low = w_low * ir_low + (1.0 - w_low) * vi_low

    ir_detail = ir - ir_low
    vi_detail = vi_y - vi_low
    detail_ir_bias = float(params["detail_ir_bias"])
    w_detail = (detail_ir_bias * g_ir + 1e-4) / (detail_ir_bias * g_ir + g_vi + 2e-4)
    w_detail = ndimage.gaussian_filter(w_detail, sigma=0.8, mode="reflect")
    detail = w_detail * ir_detail + (1.0 - w_detail) * vi_detail

    fused = low + float(params["detail_strength"]) * detail
    local_contrast = float(params["local_contrast"])
    if local_contrast:
        fused = fused + local_contrast * (fused - ndimage.gaussian_filter(fused, sigma=6.0, mode="reflect"))
    sharpen = float(params["sharpen"])
    if sharpen:
        fused = fused + sharpen * (fused - ndimage.gaussian_filter(fused, sigma=1.1, mode="reflect"))

    fused = percentile_stretch(fused, float(params["contrast_low"]), float(params["contrast_high"]))
    gamma = float(params["gamma"])
    if abs(gamma - 1.0) > 1e-4:
        fused = np.power(np.clip(fused, 0, 1), gamma)
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)

    if str(params.get("color_mode", "rgb")) == "gray" or vi_img.mode == "L":
        return Image.fromarray(np.clip(fused * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")
    return recombine_y_with_vi_color(vi_img, fused, saturation=float(params["color_saturation"]))


def generate_dir(ir_dir: str | Path, vi_dir: str | Path, out_dir: str | Path, params: dict[str, float | str]) -> list[str]:
    out = ensure_dir(out_dir)
    names = list_pairs(ir_dir, vi_dir)
    for name in names:
        ir_img = load_image(Path(ir_dir) / name)
        vi_img = load_image(Path(vi_dir) / name)
        fused = fuse_pair(ir_img, vi_img, params)
        if fused.size != ir_img.size:
            raise RuntimeError(f"Size drift for {name}: {fused.size} != {ir_img.size}")
        save_image_like(fused, out / name)
    (out / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--vi-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--preset", default="balanced", choices=sorted(PRESETS))
    parser.add_argument("--params-json")
    args = parser.parse_args()
    params = load_params(args.preset, args.params_json)
    names = generate_dir(args.ir_dir, args.vi_dir, args.out_dir, params)
    print(f"generated={len(names)} out={args.out_dir}")


if __name__ == "__main__":
    main()
