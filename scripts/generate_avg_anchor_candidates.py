from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from fusion_common import (
    ensure_dir,
    gradient_mag,
    gray_float,
    list_pairs,
    load_image,
    normalize01,
    recombine_y_with_vi_color,
    save_image_like,
)
from local_metrics import HIGHER_IS_BETTER, evaluate_dir


def match_std(x: np.ndarray, reference: np.ndarray, target_std: float, amount: float) -> np.ndarray:
    mean = float(reference.mean())
    current = float(x.std()) + 1e-6
    adjusted = (x - float(x.mean())) / current * target_std + mean
    return np.clip((1.0 - amount) * x + amount * adjusted, 0.0, 1.0).astype(np.float32)


def fuse_avg_anchor(name: str, ir_dir: Path, vi_dir: Path, params: dict[str, float | str]) -> object:
    ir_img = load_image(ir_dir / name)
    vi_img = load_image(vi_dir / name)
    ir = gray_float(ir_img)
    vi = gray_float(vi_img)

    low_sigma = float(params["low_sigma"])
    detail_sigma = float(params["detail_sigma"])
    ir_low = ndimage.gaussian_filter(ir, sigma=low_sigma, mode="reflect")
    vi_low = ndimage.gaussian_filter(vi, sigma=low_sigma, mode="reflect")
    vi_detail = vi - ndimage.gaussian_filter(vi, sigma=detail_sigma, mode="reflect")
    ir_detail = ir - ndimage.gaussian_filter(ir, sigma=detail_sigma, mode="reflect")

    g_ir = normalize01(gradient_mag(ir))
    g_vi = normalize01(gradient_mag(vi))
    thermal = normalize01(np.abs(ir - ndimage.gaussian_filter(ir, sigma=10.0, mode="reflect")))
    ir_gate = ndimage.gaussian_filter(
        normalize01(np.maximum(g_ir - float(params["edge_floor"]) * g_vi, 0.0) + float(params["thermal_mix"]) * thermal),
        sigma=1.0,
        mode="reflect",
    )

    low = float(params["ir_low_weight"]) * ir_low + (1.0 - float(params["ir_low_weight"])) * vi_low
    fused = low
    fused += float(params["vi_detail_weight"]) * vi_detail
    fused += float(params["ir_detail_weight"]) * ir_gate * ir_detail
    fused += float(params["avg_residual"]) * (0.5 * ir + 0.5 * vi - (0.5 * ir_low + 0.5 * vi_low))

    if float(params["target_std"]):
        fused = match_std(fused, vi, float(params["target_std"]), float(params["contrast_amount"]))
    if float(params["sharpen"]):
        fused += float(params["sharpen"]) * (fused - ndimage.gaussian_filter(fused, sigma=1.0, mode="reflect"))

    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
    return recombine_y_with_vi_color(vi_img, fused, saturation=float(params["saturation"]))


def candidates() -> list[tuple[str, dict[str, float | str]]]:
    base: dict[str, float | str] = {
        "low_sigma": 6.0,
        "detail_sigma": 1.4,
        "ir_low_weight": 0.50,
        "vi_detail_weight": 1.00,
        "ir_detail_weight": 0.08,
        "avg_residual": 0.08,
        "edge_floor": 0.75,
        "thermal_mix": 0.20,
        "target_std": 0.180,
        "contrast_amount": 0.70,
        "sharpen": 0.0,
        "saturation": 0.95,
        "family": "avg_anchor",
    }
    out: list[tuple[str, dict[str, float | str]]] = []

    def add(name: str, **updates: float | str) -> None:
        params = dict(base)
        params.update(updates)
        out.append((name, params))

    add("avg_anchor_std17", target_std=0.170, contrast_amount=0.55, vi_detail_weight=0.90, ir_detail_weight=0.05)
    add("avg_anchor_std18", target_std=0.180, contrast_amount=0.70, vi_detail_weight=1.00, ir_detail_weight=0.08)
    add("avg_anchor_std19", target_std=0.190, contrast_amount=0.80, vi_detail_weight=1.05, ir_detail_weight=0.10)
    add("avg_anchor_ir60", ir_low_weight=0.60, target_std=0.185, vi_detail_weight=1.00, ir_detail_weight=0.12)
    add("avg_anchor_qabf", ir_low_weight=0.42, target_std=0.180, vi_detail_weight=1.12, ir_detail_weight=0.05, edge_floor=0.90)
    add("avg_anchor_detail", target_std=0.188, vi_detail_weight=1.08, ir_detail_weight=0.18, avg_residual=0.12, sharpen=0.01)
    return out


def add_local_rank_scores(rows: list[dict[str, float | str]]) -> None:
    for metric, higher in HIGHER_IS_BETTER.items():
        order = sorted(range(len(rows)), key=lambda i: float(rows[i][metric]), reverse=higher)
        for rank, idx in enumerate(order, start=1):
            rows[idx][f"R_local_{metric}"] = rank
    for row in rows:
        row["LocalPseudoRank"] = float(np.mean([float(row[f"R_local_{metric}"]) for metric in HIGHER_IS_BETTER]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--out-root", default="results/candidates_round2")
    parser.add_argument("--start-index", type=int, default=20)
    parser.add_argument("--score-out", default="results/avg_anchor_local_scores.csv")
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir)
    vi_dir = Path(args.vi_dir)
    out_root = ensure_dir(args.out_root)
    rows: list[dict[str, float | str]] = []
    for offset, (name, params) in enumerate(candidates()):
        cand_dir = out_root / f"{args.start_index + offset:02d}_{name}"
        ensure_dir(cand_dir)
        print(cand_dir)
        for image_name in list_pairs(ir_dir, vi_dir):
            save_image_like(fuse_avg_anchor(image_name, ir_dir, vi_dir, params), cand_dir / image_name)
        (cand_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        _, summary = evaluate_dir(ir_dir, vi_dir, cand_dir)
        (cand_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        row: dict[str, float | str] = {"candidate": cand_dir.name, "path": str(cand_dir), "family": str(params["family"])}
        row.update(summary)
        rows.append(row)

    add_local_rank_scores(rows)
    rows.sort(key=lambda row: float(row["LocalPseudoRank"]))
    fieldnames = ["candidate", "path", "family", "LocalPseudoRank"] + list(HIGHER_IS_BETTER) + [
        f"R_local_{metric}" for metric in HIGHER_IS_BETTER
    ]
    score_out = Path(args.score_out)
    ensure_dir(score_out.parent)
    with score_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['LocalPseudoRank']:.3f} {row['candidate']}")


if __name__ == "__main__":
    main()
