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
    robust_match,
    save_image_like,
)
from local_metrics import HIGHER_IS_BETTER, evaluate_dir


def fuse_vi_anchor(name: str, ir_dir: Path, vi_dir: Path, params: dict[str, float | str]) -> object:
    ir_img = load_image(ir_dir / name)
    vi_img = load_image(vi_dir / name)
    ir_raw = gray_float(ir_img)
    vi = gray_float(vi_img)
    ir = robust_match(ir_raw, vi)

    low_sigma = float(params["low_sigma"])
    vi_low = ndimage.gaussian_filter(vi, sigma=low_sigma, mode="reflect")
    ir_low = ndimage.gaussian_filter(ir, sigma=low_sigma, mode="reflect")
    ir_detail = ir - ndimage.gaussian_filter(ir, sigma=float(params["detail_sigma"]), mode="reflect")
    vi_detail = vi - ndimage.gaussian_filter(vi, sigma=float(params["detail_sigma"]), mode="reflect")

    g_ir = normalize01(gradient_mag(ir))
    g_vi = normalize01(gradient_mag(vi))
    thermal = normalize01(np.abs(ir - ndimage.gaussian_filter(ir, sigma=10.0, mode="reflect")))
    edge = normalize01(np.maximum(g_ir - float(params["edge_floor"]) * g_vi, 0.0))
    gate = ndimage.gaussian_filter(
        np.clip(float(params["edge_mix"]) * edge + float(params["thermal_mix"]) * thermal, 0.0, 1.0),
        sigma=1.0,
        mode="reflect",
    )

    fused = vi.copy()
    fused += float(params["ir_low_weight"]) * (ir_low - vi_low)
    fused += float(params["avg_low_weight"]) * (0.5 * (ir_low + vi_low) - vi_low)
    fused += float(params["ir_detail_weight"]) * gate * ir_detail
    fused += float(params["vi_detail_boost"]) * (1.0 - gate) * vi_detail

    if float(params["contrast_blend"]):
        mean = float(vi.mean())
        std = float(vi.std()) + 1e-6
        target_std = float(params["target_std"])
        matched = (fused - float(fused.mean())) / (float(fused.std()) + 1e-6) * target_std + mean
        fused = (1.0 - float(params["contrast_blend"])) * fused + float(params["contrast_blend"]) * matched

    if float(params["sharpen"]):
        fused += float(params["sharpen"]) * (fused - ndimage.gaussian_filter(fused, sigma=1.0, mode="reflect")) * (
            0.35 + 0.65 * gate
        )

    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
    return recombine_y_with_vi_color(vi_img, fused, saturation=float(params["saturation"]))


def candidates() -> list[tuple[str, dict[str, float | str]]]:
    base: dict[str, float | str] = {
        "low_sigma": 8.0,
        "detail_sigma": 2.0,
        "ir_low_weight": 0.38,
        "avg_low_weight": 0.0,
        "ir_detail_weight": 0.08,
        "vi_detail_boost": 0.02,
        "edge_floor": 0.80,
        "edge_mix": 0.70,
        "thermal_mix": 0.25,
        "contrast_blend": 0.20,
        "target_std": 0.188,
        "sharpen": 0.0,
        "saturation": 0.95,
        "family": "vi_anchor",
    }
    out: list[tuple[str, dict[str, float | str]]] = []

    def add(name: str, **updates: float | str) -> None:
        params = dict(base)
        params.update(updates)
        out.append((name, params))

    add("vi_anchor_low32", ir_low_weight=0.32, ir_detail_weight=0.06, contrast_blend=0.10, target_std=0.182)
    add("vi_anchor_low42", ir_low_weight=0.42, ir_detail_weight=0.08, contrast_blend=0.18, target_std=0.188)
    add("vi_anchor_low52", ir_low_weight=0.52, ir_detail_weight=0.10, contrast_blend=0.25, target_std=0.194)
    add("vi_anchor_detail14", ir_low_weight=0.42, ir_detail_weight=0.14, vi_detail_boost=0.03, sharpen=0.01, target_std=0.192)
    add("vi_anchor_avg_low", ir_low_weight=0.22, avg_low_weight=0.34, ir_detail_weight=0.08, contrast_blend=0.12)
    add("vi_anchor_safe_qabf", ir_low_weight=0.28, ir_detail_weight=0.04, edge_floor=0.95, thermal_mix=0.15, contrast_blend=0.08)
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
    parser.add_argument("--start-index", type=int, default=14)
    parser.add_argument("--score-out", default="results/vi_anchor_local_scores.csv")
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
            fused = fuse_vi_anchor(image_name, ir_dir, vi_dir, params)
            save_image_like(fused, cand_dir / image_name)
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
