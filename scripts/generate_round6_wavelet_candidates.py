from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pywt
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


SOURCE_DEFAULTS = {
    "base33": "results/candidates_round2/33_mix_avg21_45",
    "avg12": "results/candidates_round2/12_avg_raw",
    "detail18": "results/candidates/18_low_artifact_color",
    "cross21": "results/candidates/21_crossblend_35_color",
    "vi19": "results/candidates_round2/19_vi_anchor_safe_qabf",
    "vi11": "results/candidates_round2/11_vi_only",
    "r4_02": "results/candidates_round4/02_micro_edge_balanced43",
    "r5_08": "results/candidates_round5/08_transfer18_mid",
}


def soft_std_lift(x: np.ndarray, reference: np.ndarray, target_std: float, amount: float) -> np.ndarray:
    current = float(np.std(x))
    if current < 1e-6:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    target = (x - float(np.mean(x))) / current * target_std + float(np.mean(reference))
    return np.clip((1.0 - amount) * x + amount * target, 0.0, 1.0).astype(np.float32)


def source_gate(ir: np.ndarray, vi: np.ndarray, edge_floor: float, thermal_weight: float, sigma: float) -> np.ndarray:
    g_ir = normalize01(gradient_mag(ir))
    g_vi = normalize01(gradient_mag(vi))
    thermal = normalize01(np.abs(ir - ndimage.gaussian_filter(ir, sigma=10.0, mode="reflect")))
    edge = normalize01(np.maximum(g_ir - edge_floor * g_vi, 0.0))
    gate = np.clip(edge + thermal_weight * thermal, 0.0, 1.0)
    return ndimage.gaussian_filter(gate, sigma=sigma, mode="reflect").astype(np.float32)


def cap_gradient_to_sources(fused: np.ndarray, low: np.ndarray, ir: np.ndarray, vi: np.ndarray, cap: float, amount: float) -> np.ndarray:
    gf = gradient_mag(fused)
    source_max = np.maximum(gradient_mag(ir), gradient_mag(vi))
    excess = normalize01(np.maximum(gf - cap * source_max, 0.0))
    smooth = ndimage.gaussian_filter(fused, sigma=0.75, mode="reflect")
    pulled = (1.0 - amount * excess) * fused + amount * excess * (0.70 * smooth + 0.30 * low)
    return np.clip(pulled, 0.0, 1.0).astype(np.float32)


def gray_source(name: str, key: str, dirs: dict[str, Path]) -> np.ndarray:
    return gray_float(load_image(dirs[key] / name))


def resize_like(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if x.shape == shape:
        return x
    return x[: shape[0], : shape[1]]


def weighted_sum(values: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(float(v) for v in weights.values())
    if total <= 1e-8:
        raise ValueError("weights must sum to a positive number")
    out = None
    for key, weight in weights.items():
        scaled = values[key] * (float(weight) / total)
        out = scaled if out is None else out + scaled
    return out.astype(np.float32)


def maxabs_mix(coeffs: dict[str, np.ndarray], source_keys: list[str], blend: np.ndarray, strength: float) -> np.ndarray:
    stack = np.stack([coeffs[key] for key in source_keys], axis=0)
    idx = np.argmax(np.abs(stack), axis=0)
    picked = np.take_along_axis(stack, idx[None, ...], axis=0)[0]
    return ((1.0 - strength) * blend + strength * picked).astype(np.float32)


def fuse_wavelet(name: str, dirs: dict[str, Path], params: dict[str, Any]) -> object:
    vi_img = load_image(dirs["vi"] / name)
    ir_img = load_image(dirs["ir"] / name)
    vi = gray_float(vi_img)
    ir = robust_match(gray_float(ir_img), vi)
    base33 = gray_source(name, "base33", dirs)
    avg12 = gray_source(name, "avg12", dirs)
    detail18 = gray_source(name, "detail18", dirs)
    cross21 = gray_source(name, "cross21", dirs)
    vi19 = gray_source(name, "vi19", dirs)
    vi11 = gray_source(name, "vi11", dirs)
    r4_02 = gray_source(name, "r4_02", dirs)
    r5_08 = gray_source(name, "r5_08", dirs)
    sources = {
        "vi": vi,
        "ir": ir,
        "base33": base33,
        "avg12": avg12,
        "detail18": detail18,
        "cross21": cross21,
        "vi19": vi19,
        "vi11": vi11,
        "r4_02": r4_02,
        "r5_08": r5_08,
    }
    wavelet = str(params["wavelet"])
    level = int(params["level"])
    mode = str(params.get("mode", "symmetric"))
    coeffs = {key: pywt.wavedec2(val, wavelet=wavelet, level=level, mode=mode) for key, val in sources.items()}

    low = weighted_sum({key: coeffs[key][0] for key in sources}, params["low_weights"])
    fused_coeffs: list[Any] = [low]
    maxabs_keys = list(params.get("maxabs_keys", []))
    maxabs_strength = float(params.get("maxabs_strength", 0.0))
    level_gains = list(params.get("level_gains", [1.0] * level))
    for detail_idx in range(1, level + 1):
        triplet = []
        gain = float(level_gains[min(detail_idx - 1, len(level_gains) - 1)])
        for band_idx in range(3):
            band_values = {key: coeffs[key][detail_idx][band_idx] for key in sources}
            blend = weighted_sum(band_values, params["detail_weights"])
            if maxabs_keys and maxabs_strength:
                blend = maxabs_mix(band_values, maxabs_keys, blend, maxabs_strength)
            triplet.append((gain * blend).astype(np.float32))
        fused_coeffs.append(tuple(triplet))
    fused = pywt.waverec2(fused_coeffs, wavelet=wavelet, mode=mode)
    fused = resize_like(fused.astype(np.float32), vi.shape)

    spatial_low = weighted_sum({key: sources[key] for key in sources}, params["spatial_low_weights"])
    if float(params.get("spatial_low_blend", 0.0)):
        fused = (1.0 - float(params["spatial_low_blend"])) * fused + float(params["spatial_low_blend"]) * spatial_low

    if float(params.get("gate_ir_detail", 0.0)):
        gate = source_gate(ir, vi, float(params["edge_floor"]), float(params["thermal_weight"]), float(params["gate_sigma"]))
        ir_detail = ir - ndimage.gaussian_filter(ir, sigma=float(params["gate_detail_sigma"]), mode="reflect")
        fused += float(params["gate_ir_detail"]) * gate * ir_detail
    if float(params.get("vi_detail_boost", 0.0)):
        vi_detail = vi - ndimage.gaussian_filter(vi, sigma=float(params["gate_detail_sigma"]), mode="reflect")
        fused += float(params["vi_detail_boost"]) * vi_detail

    fused = soft_std_lift(fused, vi, float(params["target_std"]), float(params["std_amount"]))
    fused = cap_gradient_to_sources(fused, spatial_low, ir, vi, float(params["grad_cap"]), float(params["cap_amount"]))
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
    return recombine_y_with_vi_color(vi_img, fused, saturation=float(params["saturation"]))


def candidate_params() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []

    def add(name: str, **updates: Any) -> None:
        params: dict[str, Any] = {
            "family": "wavelet",
            "wavelet": "db2",
            "level": 2,
            "mode": "symmetric",
            "low_weights": {"base33": 0.72, "avg12": 0.18, "vi19": 0.10},
            "detail_weights": {"base33": 0.48, "vi": 0.22, "ir": 0.08, "detail18": 0.12, "cross21": 0.10},
            "spatial_low_weights": {"base33": 0.74, "avg12": 0.18, "vi19": 0.08},
            "spatial_low_blend": 0.25,
            "maxabs_keys": [],
            "maxabs_strength": 0.0,
            "level_gains": [0.92, 0.78],
            "gate_ir_detail": 0.0,
            "vi_detail_boost": 0.0,
            "gate_detail_sigma": 1.5,
            "edge_floor": 0.90,
            "thermal_weight": 0.10,
            "gate_sigma": 1.2,
            "target_std": 43.5 / 255.0,
            "std_amount": 0.34,
            "grad_cap": 1.04,
            "cap_amount": 0.68,
            "saturation": 0.98,
        }
        params.update(updates)
        out.append((name, params))

    add("wave_db2_base_vi18_safe")
    add(
        "wave_db2_base_vi18_sd45",
        target_std=45 / 255.0,
        std_amount=0.48,
        detail_weights={"base33": 0.42, "vi": 0.22, "ir": 0.08, "detail18": 0.18, "cross21": 0.10},
        level_gains=[1.00, 0.84],
        grad_cap=1.055,
        cap_amount=0.62,
    )
    add(
        "wave_db2_r5_anchor",
        low_weights={"base33": 0.66, "avg12": 0.14, "r5_08": 0.20},
        detail_weights={"base33": 0.42, "r5_08": 0.24, "vi": 0.18, "detail18": 0.10, "ir": 0.06},
        spatial_low_weights={"base33": 0.72, "avg12": 0.14, "r5_08": 0.14},
        target_std=44 / 255.0,
        std_amount=0.40,
        level_gains=[0.96, 0.82],
    )
    add(
        "wave_sym4_vi_qabf",
        wavelet="sym4",
        level=2,
        low_weights={"base33": 0.66, "avg12": 0.14, "vi11": 0.08, "vi19": 0.12},
        detail_weights={"base33": 0.36, "vi": 0.34, "vi11": 0.12, "ir": 0.04, "detail18": 0.08, "cross21": 0.06},
        spatial_low_weights={"base33": 0.72, "avg12": 0.14, "vi19": 0.14},
        maxabs_keys=["vi", "vi11", "base33"],
        maxabs_strength=0.10,
        target_std=44 / 255.0,
        std_amount=0.38,
        level_gains=[0.94, 0.84],
        grad_cap=1.045,
        cap_amount=0.66,
    )
    add(
        "wave_sym4_detail18_mid",
        wavelet="sym4",
        level=2,
        low_weights={"base33": 0.70, "avg12": 0.16, "detail18": 0.06, "vi19": 0.08},
        detail_weights={"base33": 0.36, "detail18": 0.26, "vi": 0.18, "cross21": 0.12, "ir": 0.08},
        spatial_low_weights={"base33": 0.72, "avg12": 0.16, "vi19": 0.08, "detail18": 0.04},
        maxabs_keys=["detail18", "vi", "base33"],
        maxabs_strength=0.08,
        target_std=45 / 255.0,
        std_amount=0.48,
        level_gains=[1.02, 0.86],
        grad_cap=1.06,
        cap_amount=0.60,
    )
    add(
        "wave_coif1_cross21",
        wavelet="coif1",
        level=2,
        low_weights={"base33": 0.70, "avg12": 0.16, "cross21": 0.08, "vi19": 0.06},
        detail_weights={"base33": 0.36, "cross21": 0.28, "vi": 0.18, "detail18": 0.10, "ir": 0.08},
        spatial_low_weights={"base33": 0.74, "avg12": 0.14, "vi19": 0.08, "cross21": 0.04},
        target_std=45 / 255.0,
        std_amount=0.46,
        level_gains=[1.00, 0.86],
        grad_cap=1.055,
        cap_amount=0.62,
    )
    add(
        "wave_haar_maxabs_safe",
        wavelet="haar",
        level=3,
        low_weights={"base33": 0.76, "avg12": 0.16, "vi19": 0.08},
        detail_weights={"base33": 0.42, "vi": 0.24, "ir": 0.08, "detail18": 0.16, "cross21": 0.10},
        spatial_low_weights={"base33": 0.78, "avg12": 0.16, "vi19": 0.06},
        maxabs_keys=["base33", "vi", "detail18"],
        maxabs_strength=0.08,
        level_gains=[0.96, 0.88, 0.78],
        target_std=44 / 255.0,
        std_amount=0.38,
        grad_cap=1.045,
        cap_amount=0.68,
    )
    add(
        "wave_bior22_structure",
        wavelet="bior2.2",
        level=2,
        low_weights={"base33": 0.78, "avg12": 0.16, "vi19": 0.06},
        detail_weights={"base33": 0.54, "vi": 0.22, "detail18": 0.12, "cross21": 0.08, "ir": 0.04},
        spatial_low_weights={"base33": 0.80, "avg12": 0.16, "vi19": 0.04},
        target_std=43.5 / 255.0,
        std_amount=0.34,
        level_gains=[0.90, 0.78],
        grad_cap=1.035,
        cap_amount=0.72,
    )
    add(
        "wave_db3_ir_gate",
        wavelet="db3",
        level=2,
        detail_weights={"base33": 0.42, "vi": 0.20, "ir": 0.12, "detail18": 0.16, "cross21": 0.10},
        gate_ir_detail=0.018,
        vi_detail_boost=0.012,
        edge_floor=0.86,
        thermal_weight=0.14,
        target_std=44.5 / 255.0,
        std_amount=0.44,
        level_gains=[0.98, 0.84],
        grad_cap=1.055,
        cap_amount=0.62,
    )
    add(
        "wave_sym4_sd46",
        wavelet="sym4",
        level=2,
        low_weights={"base33": 0.66, "avg12": 0.14, "vi19": 0.10, "detail18": 0.10},
        detail_weights={"base33": 0.34, "detail18": 0.28, "vi": 0.20, "cross21": 0.12, "ir": 0.06},
        spatial_low_weights={"base33": 0.70, "avg12": 0.14, "vi19": 0.08, "detail18": 0.08},
        maxabs_keys=["detail18", "vi", "cross21", "base33"],
        maxabs_strength=0.10,
        target_std=46 / 255.0,
        std_amount=0.54,
        level_gains=[1.04, 0.88],
        grad_cap=1.065,
        cap_amount=0.58,
    )
    add(
        "wave_db2_qabf_lowrisk",
        low_weights={"base33": 0.78, "avg12": 0.14, "vi11": 0.08},
        detail_weights={"base33": 0.46, "vi": 0.30, "vi11": 0.10, "detail18": 0.08, "ir": 0.06},
        spatial_low_weights={"base33": 0.80, "avg12": 0.14, "vi11": 0.06},
        maxabs_keys=["base33", "vi", "vi11"],
        maxabs_strength=0.08,
        target_std=43.5 / 255.0,
        std_amount=0.34,
        level_gains=[0.92, 0.84],
        grad_cap=1.035,
        cap_amount=0.72,
    )
    add(
        "wave_r402_safe",
        low_weights={"base33": 0.72, "avg12": 0.14, "r4_02": 0.14},
        detail_weights={"base33": 0.42, "r4_02": 0.22, "vi": 0.20, "detail18": 0.10, "ir": 0.06},
        spatial_low_weights={"base33": 0.78, "avg12": 0.12, "r4_02": 0.10},
        target_std=43.0 / 255.0,
        std_amount=0.30,
        level_gains=[0.92, 0.80],
        grad_cap=1.03,
        cap_amount=0.74,
    )
    return out


def add_local_rank_scores(rows: list[dict[str, float | str]]) -> None:
    for metric, higher in HIGHER_IS_BETTER.items():
        order = sorted(range(len(rows)), key=lambda i: float(rows[i][metric]), reverse=higher)
        for rank, idx in enumerate(order, start=1):
            rows[idx][f"R_local_{metric}"] = rank
    for row in rows:
        row["LocalPseudoRank"] = float(np.mean([float(row[f"R_local_{metric}"]) for metric in HIGHER_IS_BETTER]))


def add_wavelet_score(rows: list[dict[str, float | str]]) -> None:
    for row in rows:
        sd = float(row["SD"])
        qabf = float(row["Qabf"])
        nabf = float(row["Nabf"])
        ssim = float(row["SSIM"])
        corr = float(row["Base33Corr"])
        mi = float(row["MI"])
        score = float(row["LocalPseudoRank"])
        score += abs(sd - 45.0 / 255.0) * 22.0
        score += max(0.0, 0.625 - qabf) * 8.0
        score += max(0.0, nabf - 0.060) * 24.0
        score += max(0.0, 0.704 - ssim) * 20.0
        score += max(0.0, 0.9982 - corr) * 130.0
        score += max(0.0, 2.70 - mi) * 1.2
        row["Round6Heuristic"] = float(score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--out-root", default="results/candidates_round6")
    parser.add_argument("--score-out", default="results/round6_local_scores.csv")
    args = parser.parse_args()

    dirs = {"ir": Path(args.ir_dir), "vi": Path(args.vi_dir)}
    dirs.update({key: Path(value) for key, value in SOURCE_DEFAULTS.items()})
    for key, path in dirs.items():
        if not path.exists():
            raise FileNotFoundError(f"{key} missing: {path}")

    out_root = ensure_dir(args.out_root)
    names = list_pairs(dirs["ir"], dirs["vi"])
    rows: list[dict[str, float | str]] = []
    for idx, (name, params) in enumerate(candidate_params(), start=1):
        cand_dir = ensure_dir(out_root / f"{idx:02d}_{name}")
        print(cand_dir)
        for image_name in names:
            save_image_like(fuse_wavelet(image_name, dirs, params), cand_dir / image_name)
        (cand_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        _, summary = evaluate_dir(dirs["ir"], dirs["vi"], cand_dir)
        (cand_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        corr_vals = []
        for image_name in names:
            corr_vals.append(corrcoef(gray_source(image_name, "base33", dirs), gray_float(load_image(cand_dir / image_name))))
        row: dict[str, float | str] = {
            "candidate": cand_dir.name,
            "path": str(cand_dir),
            "family": "wavelet",
            "Base33Corr": float(np.mean(corr_vals)),
        }
        row.update(summary)
        rows.append(row)

    add_local_rank_scores(rows)
    add_wavelet_score(rows)
    rows.sort(key=lambda row: (float(row["Round6Heuristic"]), float(row["LocalPseudoRank"])))
    fieldnames = ["candidate", "path", "family", "Round6Heuristic", "LocalPseudoRank", "Base33Corr"] + list(HIGHER_IS_BETTER) + [
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
            f"{float(row['Round6Heuristic']):.3f} local={float(row['LocalPseudoRank']):.3f} "
            f"corr={float(row['Base33Corr']):.5f} {row['candidate']}"
        )
    print(f"wrote {score_out}")


if __name__ == "__main__":
    main()
