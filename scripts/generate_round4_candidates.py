from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from fusion_common import (
    corrcoef,
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


def soft_std_lift(x: np.ndarray, reference: np.ndarray, target_std: float, amount: float) -> np.ndarray:
    current = float(np.std(x))
    if current < 1e-6:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    target = (x - float(np.mean(x))) / current * target_std + float(np.mean(reference))
    return np.clip((1.0 - amount) * x + amount * target, 0.0, 1.0).astype(np.float32)


def source_edge_gate(ir: np.ndarray, vi: np.ndarray, edge_floor: float, thermal_weight: float, sigma: float) -> np.ndarray:
    g_ir = normalize01(gradient_mag(ir))
    g_vi = normalize01(gradient_mag(vi))
    thermal = normalize01(np.abs(ir - ndimage.gaussian_filter(ir, sigma=10.0, mode="reflect")))
    edge = normalize01(np.maximum(g_ir - edge_floor * g_vi, 0.0))
    gate = np.clip(edge + thermal_weight * thermal, 0.0, 1.0)
    return ndimage.gaussian_filter(gate, sigma=sigma, mode="reflect").astype(np.float32)


def cap_gradient_to_sources(fused: np.ndarray, low: np.ndarray, ir: np.ndarray, vi: np.ndarray, cap: float, amount: float) -> np.ndarray:
    gf = gradient_mag(fused)
    gmax = np.maximum(gradient_mag(ir), gradient_mag(vi))
    excess = normalize01(np.maximum(gf - cap * gmax, 0.0))
    smooth = ndimage.gaussian_filter(fused, sigma=0.75, mode="reflect")
    pulled = (1.0 - amount * excess) * fused + amount * excess * (0.65 * smooth + 0.35 * low)
    return np.clip(pulled, 0.0, 1.0).astype(np.float32)


def choose_anchor(name: str, dirs: dict[str, Path], params: dict[str, float | str]) -> np.ndarray:
    anchor = str(params.get("anchor", "base33"))
    if anchor == "base33":
        return gray_float(load_image(dirs["base33"] / name))
    if anchor == "r3_10":
        return gray_float(load_image(dirs["r3_10"] / name))
    if anchor == "r3_07":
        return gray_float(load_image(dirs["r3_07"] / name))
    if anchor == "r3_09":
        return gray_float(load_image(dirs["r3_09"] / name))
    raise ValueError(f"Unknown anchor: {anchor}")


def fuse_pair(name: str, dirs: dict[str, Path], params: dict[str, float | str]) -> object:
    ir_img = load_image(dirs["ir"] / name)
    vi_img = load_image(dirs["vi"] / name)
    ir_raw = gray_float(ir_img)
    vi = gray_float(vi_img)
    ir = robust_match(ir_raw, vi)
    base33 = gray_float(load_image(dirs["base33"] / name))
    avg_raw = gray_float(load_image(dirs["avg_raw"] / name))
    anchor = choose_anchor(name, dirs, params)
    family = str(params["family"])

    if family == "micro_edge":
        low = (1.0 - float(params["avg_low_weight"])) * base33 + float(params["avg_low_weight"]) * avg_raw
        detail_sigma = float(params["detail_sigma"])
        vi_detail = vi - ndimage.gaussian_filter(vi, sigma=detail_sigma, mode="reflect")
        ir_detail = ir - ndimage.gaussian_filter(ir, sigma=detail_sigma, mode="reflect")
        gate = source_edge_gate(ir, vi, float(params["edge_floor"]), float(params["thermal_weight"]), float(params["gate_sigma"]))
        fused = low + float(params["vi_detail_weight"]) * vi_detail + float(params["ir_detail_weight"]) * gate * ir_detail
        fused = cap_gradient_to_sources(fused, low, ir, vi, float(params["grad_cap"]), float(params["cap_amount"]))
        fused = soft_std_lift(fused, vi, float(params["target_std"]), float(params["std_amount"]))
    elif family == "blend_r3":
        low = (1.0 - float(params["avg_low_weight"])) * base33 + float(params["avg_low_weight"]) * avg_raw
        fused = (1.0 - float(params["r3_weight"])) * low + float(params["r3_weight"]) * anchor
        fused = cap_gradient_to_sources(fused, low, ir, vi, float(params["grad_cap"]), float(params["cap_amount"]))
        fused = soft_std_lift(fused, vi, float(params["target_std"]), float(params["std_amount"]))
    elif family == "mi_soft":
        low = (1.0 - float(params["avg_weight"])) * base33 + float(params["avg_weight"]) * avg_raw
        source_mix = float(params["vi_weight"]) * vi + float(params["ir_weight"]) * ir
        fused = (1.0 - float(params["source_weight"])) * low + float(params["source_weight"]) * source_mix
        fused = soft_std_lift(fused, vi, float(params["target_std"]), float(params["std_amount"]))
        fused = cap_gradient_to_sources(fused, low, ir, vi, float(params["grad_cap"]), float(params["cap_amount"]))
    elif family == "contrast_repair":
        blurred = ndimage.gaussian_filter(anchor, sigma=float(params["blur_sigma"]), mode="reflect")
        detail = anchor - blurred
        contrast = soft_std_lift(base33, vi, float(params["target_std"]), float(params["std_amount"]))
        fused = (1.0 - float(params["contrast_weight"])) * base33 + float(params["contrast_weight"]) * contrast
        fused = fused + float(params["detail_weight"]) * detail
        fused = cap_gradient_to_sources(fused, base33, ir, vi, float(params["grad_cap"]), float(params["cap_amount"]))
    else:
        raise ValueError(f"Unknown family: {family}")

    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
    return recombine_y_with_vi_color(vi_img, fused, saturation=float(params["saturation"]))


def candidate_params() -> list[tuple[str, dict[str, float | str]]]:
    out: list[tuple[str, dict[str, float | str]]] = []

    def add(name: str, family: str, **updates: float | str) -> None:
        params: dict[str, float | str] = {"family": family, "saturation": 0.98}
        params.update(updates)
        out.append((name, params))

    micro_base = {
        "avg_low_weight": 0.08,
        "detail_sigma": 1.4,
        "edge_floor": 0.95,
        "thermal_weight": 0.10,
        "gate_sigma": 1.2,
        "vi_detail_weight": 0.12,
        "ir_detail_weight": 0.025,
        "target_std": 42 / 255.0,
        "std_amount": 0.25,
        "grad_cap": 1.04,
        "cap_amount": 0.50,
    }
    add("micro_edge_very_safe", "micro_edge", **micro_base)
    p = dict(micro_base)
    p.update({"avg_low_weight": 0.12, "vi_detail_weight": 0.18, "ir_detail_weight": 0.035, "target_std": 43 / 255.0, "std_amount": 0.32})
    add("micro_edge_balanced43", "micro_edge", **p)
    p = dict(micro_base)
    p.update({"edge_floor": 0.88, "thermal_weight": 0.14, "vi_detail_weight": 0.22, "ir_detail_weight": 0.045, "target_std": 44 / 255.0, "std_amount": 0.38})
    add("micro_edge_qabf44", "micro_edge", **p)
    p = dict(micro_base)
    p.update({"detail_sigma": 1.8, "vi_detail_weight": 0.26, "ir_detail_weight": 0.05, "target_std": 44 / 255.0, "std_amount": 0.42, "grad_cap": 1.08})
    add("micro_edge_broad44", "micro_edge", **p)

    blend_base = {
        "avg_low_weight": 0.05,
        "r3_weight": 0.12,
        "target_std": 42 / 255.0,
        "std_amount": 0.25,
        "grad_cap": 1.03,
        "cap_amount": 0.60,
    }
    add("blend10_w12_cap", "blend_r3", anchor="r3_10", **blend_base)
    p = dict(blend_base)
    p.update({"r3_weight": 0.18, "target_std": 43 / 255.0, "std_amount": 0.34, "grad_cap": 1.05})
    add("blend10_w18", "blend_r3", anchor="r3_10", **p)
    p = dict(blend_base)
    p.update({"r3_weight": 0.22, "target_std": 44 / 255.0, "std_amount": 0.40, "grad_cap": 1.06})
    add("blend07_w22", "blend_r3", anchor="r3_07", **p)
    p = dict(blend_base)
    p.update({"r3_weight": 0.25, "target_std": 44 / 255.0, "std_amount": 0.42, "grad_cap": 1.06})
    add("blend09_w25", "blend_r3", anchor="r3_09", **p)

    mi_base = {
        "avg_weight": 0.08,
        "source_weight": 0.08,
        "vi_weight": 0.62,
        "ir_weight": 0.38,
        "target_std": 42 / 255.0,
        "std_amount": 0.25,
        "grad_cap": 1.04,
        "cap_amount": 0.55,
    }
    add("mi_soft_source08", "mi_soft", **mi_base)
    p = dict(mi_base)
    p.update({"avg_weight": 0.12, "source_weight": 0.12, "target_std": 43 / 255.0, "std_amount": 0.35})
    add("mi_soft_source12", "mi_soft", **p)
    p = dict(mi_base)
    p.update({"vi_weight": 0.50, "ir_weight": 0.50, "source_weight": 0.16, "target_std": 44 / 255.0, "std_amount": 0.40})
    add("mi_soft_even16", "mi_soft", **p)

    add(
        "contrast_r3safe_light",
        "contrast_repair",
        anchor="r3_07",
        blur_sigma=1.2,
        target_std=43 / 255.0,
        std_amount=0.30,
        contrast_weight=0.65,
        detail_weight=0.10,
        grad_cap=1.04,
        cap_amount=0.60,
    )
    add(
        "contrast_r3qabf_mid",
        "contrast_repair",
        anchor="r3_09",
        blur_sigma=1.4,
        target_std=44 / 255.0,
        std_amount=0.38,
        contrast_weight=0.70,
        detail_weight=0.16,
        grad_cap=1.06,
        cap_amount=0.58,
    )
    return out


def add_local_rank_scores(rows: list[dict[str, float | str]]) -> None:
    for metric, higher in HIGHER_IS_BETTER.items():
        order = sorted(range(len(rows)), key=lambda i: float(rows[i][metric]), reverse=higher)
        for rank, idx in enumerate(order, start=1):
            rows[idx][f"R_local_{metric}"] = rank
    for row in rows:
        row["LocalPseudoRank"] = float(np.mean([float(row[f"R_local_{metric}"]) for metric in HIGHER_IS_BETTER]))


def add_anchor_correlations(rows: list[dict[str, float | str]], ir_dir: Path, vi_dir: Path, base_dir: Path, cand_dir: Path) -> None:
    vals = []
    names = list_pairs(ir_dir, vi_dir)
    for name in names:
        base = gray_float(load_image(base_dir / name))
        fused = gray_float(load_image(cand_dir / name))
        vals.append(corrcoef(base, fused))
    rows[-1]["Base33Corr"] = float(np.mean(vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--base33-dir", default="results/candidates_round2/33_mix_avg21_45")
    parser.add_argument("--avg-raw-dir", default="results/candidates_round2/12_avg_raw")
    parser.add_argument("--r3-10-dir", default="results/candidates_round3/10_source_edge_avg_low30")
    parser.add_argument("--r3-07-dir", default="results/candidates_round3/07_source_edge_vi030_ir006")
    parser.add_argument("--r3-09-dir", default="results/candidates_round3/09_source_edge_safe_qabf")
    parser.add_argument("--out-root", default="results/candidates_round4")
    parser.add_argument("--score-out", default="results/round4_local_scores.csv")
    args = parser.parse_args()

    dirs = {
        "ir": Path(args.ir_dir),
        "vi": Path(args.vi_dir),
        "base33": Path(args.base33_dir),
        "avg_raw": Path(args.avg_raw_dir),
        "r3_10": Path(args.r3_10_dir),
        "r3_07": Path(args.r3_07_dir),
        "r3_09": Path(args.r3_09_dir),
    }
    for label, path in dirs.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} directory not found: {path}")

    out_root = ensure_dir(args.out_root)
    names = list_pairs(dirs["ir"], dirs["vi"])
    rows: list[dict[str, float | str]] = []
    for idx, (name, params) in enumerate(candidate_params(), start=1):
        cand_dir = ensure_dir(out_root / f"{idx:02d}_{name}")
        print(cand_dir)
        for image_name in names:
            save_image_like(fuse_pair(image_name, dirs, params), cand_dir / image_name)
        (cand_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        _, summary = evaluate_dir(dirs["ir"], dirs["vi"], cand_dir)
        (cand_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        row: dict[str, float | str] = {"candidate": cand_dir.name, "path": str(cand_dir), "family": str(params["family"])}
        row.update(summary)
        rows.append(row)
        add_anchor_correlations(rows, dirs["ir"], dirs["vi"], dirs["base33"], cand_dir)

    add_local_rank_scores(rows)
    rows.sort(key=lambda row: (float(row["LocalPseudoRank"]), -float(row["Base33Corr"])))
    fieldnames = ["candidate", "path", "family", "LocalPseudoRank", "Base33Corr"] + list(HIGHER_IS_BETTER) + [
        f"R_local_{metric}" for metric in HIGHER_IS_BETTER
    ]
    score_out = Path(args.score_out)
    ensure_dir(score_out.parent)
    with score_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{float(row['LocalPseudoRank']):.3f} {row['candidate']} "
            f"corr={float(row['Base33Corr']):.5f} {row['family']}"
        )
    print(f"wrote {score_out}")


if __name__ == "__main__":
    main()
