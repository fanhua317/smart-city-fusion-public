from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from fusion_common import (
    corrcoef,
    ensure_dir,
    entropy_u8,
    gradient_mag,
    gray_float,
    list_pairs,
    load_image,
    mutual_information,
    ssim_simple,
)


HIGHER_IS_BETTER = {
    "AG": True,
    "CC": True,
    "EN": True,
    "MI": True,
    "MSE": False,
    "Nabf": False,
    "PSNR": True,
    "Qabf": True,
    "SCD": True,
    "SD": True,
    "SF": True,
    "SSIM": True,
    "VIF": True,
}


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / m))


def spatial_frequency(x: np.ndarray) -> float:
    rf = np.diff(x, axis=0)
    cf = np.diff(x, axis=1)
    return float(math.sqrt(np.mean(rf * rf) + np.mean(cf * cf)))


def average_gradient(x: np.ndarray) -> float:
    gx = np.diff(x, axis=1)
    gy = np.diff(x, axis=0)
    h = min(gx.shape[0], gy.shape[0])
    w = min(gx.shape[1], gy.shape[1])
    return float(np.mean(np.sqrt((gx[:h, :w] ** 2 + gy[:h, :w] ** 2) / 2.0)))


def qabf_like(ir: np.ndarray, vi: np.ndarray, f: np.ndarray) -> float:
    gi = gradient_mag(ir)
    gv = gradient_mag(vi)
    gf = gradient_mag(f)
    qi = np.minimum(gi, gf) / (np.maximum(gi, gf) + 1e-6)
    qv = np.minimum(gv, gf) / (np.maximum(gv, gf) + 1e-6)
    wi = gi / (gi + gv + 1e-6)
    score = wi * qi + (1.0 - wi) * qv
    return float(np.mean(score))


def nabf_like(ir: np.ndarray, vi: np.ndarray, f: np.ndarray) -> float:
    gi = gradient_mag(ir)
    gv = gradient_mag(vi)
    gf = gradient_mag(f)
    source_max = np.maximum(gi, gv)
    artifact = np.maximum(gf - source_max, 0.0)
    return float(np.mean(artifact) / (np.mean(gf) + 1e-6))


def vif_like(src: np.ndarray, f: np.ndarray) -> float:
    vals = []
    for sigma in (1.0, 2.0, 4.0):
        src_l = src - src.mean()
        f_l = f - f.mean()
        src_s = src_l if sigma == 1.0 else src_l[:: int(sigma), :: int(sigma)]
        f_s = f_l if sigma == 1.0 else f_l[:: int(sigma), :: int(sigma)]
        vals.append(max(0.0, corrcoef(src_s, f_s)))
    return float(np.mean(vals))


def metrics_for_pair(ir: np.ndarray, vi: np.ndarray, f: np.ndarray) -> dict[str, float]:
    m_ir = mse(f, ir)
    m_vi = mse(f, vi)
    return {
        "AG": average_gradient(f),
        "CC": 0.5 * (corrcoef(f, ir) + corrcoef(f, vi)),
        "EN": entropy_u8(f),
        "MI": mutual_information(f, ir) + mutual_information(f, vi),
        "MSE": 0.5 * (m_ir + m_vi),
        "Nabf": nabf_like(ir, vi, f),
        "PSNR": 0.5 * (psnr(f, ir) + psnr(f, vi)),
        "Qabf": qabf_like(ir, vi, f),
        "SCD": corrcoef(f - vi, ir) + corrcoef(f - ir, vi),
        "SD": float(np.std(f)),
        "SF": spatial_frequency(f),
        "SSIM": 0.5 * (ssim_simple(f, ir) + ssim_simple(f, vi)),
        "VIF": 0.5 * (vif_like(ir, f) + vif_like(vi, f)),
    }


def evaluate_dir(ir_dir: str | Path, vi_dir: str | Path, fused_dir: str | Path) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    rows: list[dict[str, float | str]] = []
    names = list_pairs(ir_dir, vi_dir)
    for name in names:
        f_path = Path(fused_dir) / name
        if not f_path.exists():
            raise FileNotFoundError(f"Missing fused image {f_path}")
        ir = gray_float(load_image(Path(ir_dir) / name))
        vi = gray_float(load_image(Path(vi_dir) / name))
        f = gray_float(load_image(f_path))
        if ir.shape != f.shape:
            raise ValueError(f"Shape mismatch {name}: ir={ir.shape} fused={f.shape}")
        row: dict[str, float | str] = {"image": name}
        row.update(metrics_for_pair(ir, vi, f))
        rows.append(row)
    summary = {k: float(np.mean([float(r[k]) for r in rows])) for k in HIGHER_IS_BETTER}
    return rows, summary


def write_csv(path: str | Path, rows: list[dict[str, float | str]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames = ["image"] + list(HIGHER_IS_BETTER)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--vi-dir", required=True)
    parser.add_argument("--fused-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    args = parser.parse_args()
    rows, summary = evaluate_dir(args.ir_dir, args.vi_dir, args.fused_dir)
    write_csv(args.out, rows)
    summary_path = Path(args.summary_out) if args.summary_out else Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
