from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fusion_common import (
    gray_float,
    load_image,
    percentile_stretch,
    recombine_y_with_vi_color,
    save_image_like,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--deep-dir", required=True)
    parser.add_argument("--vi-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--deep-weight", type=float, default=0.25)
    parser.add_argument("--contrast-low", type=float, default=0.4)
    parser.add_argument("--contrast-high", type=float, default=99.6)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--color-mode", choices=["base", "vi", "gray"], default="base")
    parser.add_argument("--saturation", type=float, default=0.9)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    deep_dir = Path(args.deep_dir)
    vi_dir = Path(args.vi_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    for name in names:
        base_img = load_image(base_dir / name)
        deep_img = load_image(deep_dir / name)
        vi_img = load_image(vi_dir / name)
        base_y = gray_float(base_img)
        deep_y = gray_float(deep_img)
        fused = (1.0 - args.deep_weight) * base_y + args.deep_weight * percentile_stretch(deep_y, 0.3, 99.7)
        fused = percentile_stretch(fused, args.contrast_low, args.contrast_high)
        if abs(args.gamma - 1.0) > 1e-4:
            fused = np.power(np.clip(fused, 0, 1), args.gamma)
        fused = np.clip(fused, 0, 1)
        if args.color_mode == "gray":
            out = Image.fromarray(np.clip(fused * 255 + 0.5, 0, 255).astype(np.uint8), mode="L")
        elif args.color_mode == "vi":
            out = recombine_y_with_vi_color(vi_img, fused, saturation=args.saturation)
        else:
            if base_img.mode == "L":
                out = Image.fromarray(np.clip(fused * 255 + 0.5, 0, 255).astype(np.uint8), mode="L")
            else:
                out = recombine_y_with_vi_color(base_img, fused, saturation=args.saturation)
        save_image_like(out, out_dir / name)

    (out_dir / "blend_params.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"blended={len(names)} out={out_dir}")


if __name__ == "__main__":
    main()
