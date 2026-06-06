from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fusion_common import ensure_dir
from fusion_generate import PRESETS, generate_dir
from local_metrics import HIGHER_IS_BETTER, evaluate_dir


def candidate_params() -> list[tuple[str, dict[str, float | str]]]:
    candidates: list[tuple[str, dict[str, float | str]]] = []
    for name, params in PRESETS.items():
        candidates.append((name, dict(params)))

    base = dict(PRESETS["balanced"])
    variants = [
        ("b_ir040_detail090", {"base_ir": 0.40, "saliency_gain": 0.30, "detail_strength": 0.90, "sharpen": 0.10}),
        ("b_ir050_detail090", {"base_ir": 0.50, "saliency_gain": 0.36, "detail_strength": 0.90, "sharpen": 0.12}),
        ("b_ir060_soft", {"base_ir": 0.60, "saliency_gain": 0.28, "detail_strength": 0.68, "local_contrast": 0.06}),
        ("qabf_high", {"base_ir": 0.48, "saliency_gain": 0.44, "detail_strength": 1.12, "detail_ir_bias": 1.45, "sharpen": 0.24}),
        ("nabf_safe", {"base_ir": 0.42, "saliency_gain": 0.22, "detail_strength": 0.55, "sharpen": 0.02, "local_contrast": 0.02}),
        ("mi_color", {"base_ir": 0.46, "saliency_gain": 0.34, "contrast_low": 0.1, "contrast_high": 99.9, "gamma": 0.90, "color_saturation": 1.08}),
        ("ssim_safe_gray", {"base_ir": 0.40, "saliency_gain": 0.24, "detail_strength": 0.58, "gamma": 1.02, "color_mode": "gray"}),
        ("thermal_gray", {"base_ir": 0.58, "saliency_gain": 0.45, "detail_strength": 0.82, "color_mode": "gray"}),
        ("detail_gray", {"base_ir": 0.48, "saliency_gain": 0.42, "detail_strength": 1.10, "sharpen": 0.22, "color_mode": "gray"}),
        ("visible_soft", {"base_ir": 0.32, "saliency_gain": 0.30, "detail_strength": 0.74, "color_saturation": 1.05}),
        ("contrast_push", {"base_ir": 0.48, "saliency_gain": 0.34, "contrast_low": 0.05, "contrast_high": 99.95, "gamma": 0.86, "local_contrast": 0.26}),
        ("low_artifact_color", {"base_ir": 0.44, "saliency_gain": 0.30, "detail_strength": 0.70, "detail_ir_bias": 1.05, "sharpen": 0.04, "contrast_low": 0.6, "contrast_high": 99.4}),
    ]
    for name, updates in variants:
        p = dict(base)
        p.update(updates)
        candidates.append((name, p))
    return candidates


def add_rank_scores(rows: list[dict[str, float | str]]) -> None:
    n = len(rows)
    for metric, higher in HIGHER_IS_BETTER.items():
        order = sorted(range(n), key=lambda i: float(rows[i][metric]), reverse=higher)
        for rank, idx in enumerate(order, start=1):
            rows[idx][f"R_{metric}"] = rank
    for row in rows:
        row["PseudoRank"] = float(np.mean([float(row[f"R_{m}"]) for m in HIGHER_IS_BETTER]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--out-root", default="results/candidates")
    parser.add_argument("--score-out", default="results/candidate_scores.csv")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_root = ensure_dir(args.out_root)
    rows: list[dict[str, float | str]] = []
    cands = candidate_params()
    if args.limit:
        cands = cands[: args.limit]
    for idx, (name, params) in enumerate(cands, start=1):
        cand_dir = out_root / f"{idx:02d}_{name}"
        print(f"[{idx}/{len(cands)}] generate {cand_dir}")
        generate_dir(args.ir_dir, args.vi_dir, cand_dir, params)
        image_rows, summary = evaluate_dir(args.ir_dir, args.vi_dir, cand_dir)
        with (cand_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        row: dict[str, float | str] = {"candidate": cand_dir.name, "path": str(cand_dir)}
        row.update(summary)
        rows.append(row)

    add_rank_scores(rows)
    rows = sorted(rows, key=lambda r: float(r["PseudoRank"]))
    fieldnames = ["candidate", "path", "PseudoRank"] + list(HIGHER_IS_BETTER) + [f"R_{m}" for m in HIGHER_IS_BETTER]
    score_out = Path(args.score_out)
    ensure_dir(score_out.parent)
    with score_out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {score_out}")
    for row in rows[:5]:
        print(f"TOP {row['PseudoRank']:.3f} {row['candidate']}")


if __name__ == "__main__":
    main()
