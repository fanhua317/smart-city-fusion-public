from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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


SOURCE_DEFAULTS = {
    "base33": "results/candidates_round2/33_mix_avg21_45",
    "avg12": "results/candidates_round2/12_avg_raw",
    "detail18": "results/candidates/18_low_artifact_color",
    "cross21": "results/candidates/21_crossblend_35_color",
    "vi11": "results/candidates_round2/11_vi_only",
    "vi19": "results/candidates_round2/19_vi_anchor_safe_qabf",
    "r4_02": "results/candidates_round4/02_micro_edge_balanced43",
    "r4_04": "results/candidates_round4/04_micro_edge_broad44",
    "r3_10": "results/candidates_round3/10_source_edge_avg_low30",
}


OFFICIAL_PATH_ROOTS = [
    Path("results/candidates"),
    Path("results/candidates_round2"),
    Path("results/candidates_round3"),
    Path("results/candidates_round4"),
]


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
    ir_adv = normalize01(np.maximum(g_ir - edge_floor * g_vi, 0.0))
    gate = np.clip(ir_adv + thermal_weight * thermal, 0.0, 1.0)
    return ndimage.gaussian_filter(gate, sigma=sigma, mode="reflect").astype(np.float32)


def cap_gradient_to_sources(fused: np.ndarray, low: np.ndarray, ir: np.ndarray, vi: np.ndarray, cap: float, amount: float) -> np.ndarray:
    gf = gradient_mag(fused)
    gmax = np.maximum(gradient_mag(ir), gradient_mag(vi))
    excess = normalize01(np.maximum(gf - cap * gmax, 0.0))
    smooth = ndimage.gaussian_filter(fused, sigma=0.8, mode="reflect")
    pulled = (1.0 - amount * excess) * fused + amount * excess * (0.70 * smooth + 0.30 * low)
    return np.clip(pulled, 0.0, 1.0).astype(np.float32)


def gray_from_source(name: str, key: str, dirs: dict[str, Path]) -> np.ndarray:
    return gray_float(load_image(dirs[key] / name))


def weighted_source_y(name: str, dirs: dict[str, Path], weights: dict[str, float]) -> np.ndarray:
    total = sum(float(v) for v in weights.values())
    if total <= 1e-8:
        raise ValueError("weights must sum to a positive number")
    fused: np.ndarray | None = None
    for key, weight in weights.items():
        y = gray_from_source(name, key, dirs)
        scaled = (float(weight) / total) * y
        fused = scaled if fused is None else fused + scaled
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def fuse_pair(name: str, dirs: dict[str, Path], params: dict[str, Any]) -> object:
    ir_img = load_image(dirs["ir"] / name)
    vi_img = load_image(dirs["vi"] / name)
    vi = gray_float(vi_img)
    ir_raw = gray_float(ir_img)
    ir = robust_match(ir_raw, vi)

    family = str(params["family"])
    if family == "official_blend":
        low = weighted_source_y(name, dirs, params["weights"])
        fused = low.copy()
        if float(params.get("vi_detail_weight", 0.0)):
            vi_detail = vi - ndimage.gaussian_filter(vi, sigma=float(params["detail_sigma"]), mode="reflect")
            fused += float(params["vi_detail_weight"]) * vi_detail
        fused = soft_std_lift(fused, vi, float(params["target_std"]), float(params["std_amount"]))
        fused = cap_gradient_to_sources(fused, low, ir, vi, float(params["grad_cap"]), float(params["cap_amount"]))
    elif family == "detail_transfer":
        low = weighted_source_y(name, dirs, params["low_weights"])
        detail_src = gray_from_source(name, str(params["detail_source"]), dirs)
        detail_sigma = float(params["detail_sigma"])
        src_detail = detail_src - ndimage.gaussian_filter(detail_src, sigma=detail_sigma, mode="reflect")
        gate = source_gate(ir, vi, float(params["edge_floor"]), float(params["thermal_weight"]), float(params["gate_sigma"]))
        src_grad_gate = ndimage.gaussian_filter(normalize01(gradient_mag(detail_src)), sigma=1.0, mode="reflect")
        detail_mask = np.clip(float(params["gate_mix"]) * gate + (1.0 - float(params["gate_mix"])) * src_grad_gate, 0.0, 1.0)
        fused = low + float(params["detail_weight"]) * (float(params["detail_bias"]) + (1.0 - float(params["detail_bias"])) * detail_mask) * src_detail
        if float(params.get("vi_detail_weight", 0.0)):
            vi_detail = vi - ndimage.gaussian_filter(vi, sigma=detail_sigma, mode="reflect")
            fused += float(params["vi_detail_weight"]) * vi_detail
        fused = soft_std_lift(fused, vi, float(params["target_std"]), float(params["std_amount"]))
        fused = cap_gradient_to_sources(fused, low, ir, vi, float(params["grad_cap"]), float(params["cap_amount"]))
    elif family == "vi_profile_repair":
        low = weighted_source_y(name, dirs, params["low_weights"])
        vi_low = ndimage.gaussian_filter(vi, sigma=float(params["low_sigma"]), mode="reflect")
        ir_low = ndimage.gaussian_filter(ir, sigma=float(params["low_sigma"]), mode="reflect")
        gate = source_gate(ir, vi, float(params["edge_floor"]), float(params["thermal_weight"]), float(params["gate_sigma"]))
        vi_detail = vi - ndimage.gaussian_filter(vi, sigma=float(params["detail_sigma"]), mode="reflect")
        ir_detail = ir - ndimage.gaussian_filter(ir, sigma=float(params["detail_sigma"]), mode="reflect")
        fused = low + float(params["low_ir_repair"]) * gate * (ir_low - vi_low)
        fused += float(params["vi_detail_weight"]) * vi_detail
        fused += float(params["ir_detail_weight"]) * gate * ir_detail
        fused = soft_std_lift(fused, vi, float(params["target_std"]), float(params["std_amount"]))
        fused = cap_gradient_to_sources(fused, low, ir, vi, float(params["grad_cap"]), float(params["cap_amount"]))
    else:
        raise ValueError(f"Unknown family: {family}")

    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
    return recombine_y_with_vi_color(vi_img, fused, saturation=float(params["saturation"]))


def candidate_params() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []

    def add(name: str, family: str, **updates: Any) -> None:
        params: dict[str, Any] = {"family": family, "saturation": 0.98}
        params.update(updates)
        out.append((name, params))

    add(
        "blend33_vi19_detail18_safe",
        "official_blend",
        weights={"base33": 0.64, "vi19": 0.18, "detail18": 0.10, "avg12": 0.08},
        detail_sigma=1.4,
        vi_detail_weight=0.02,
        target_std=43.5 / 255.0,
        std_amount=0.32,
        grad_cap=1.04,
        cap_amount=0.62,
    )
    add(
        "blend33_vi11_detail18",
        "official_blend",
        weights={"base33": 0.68, "vi11": 0.14, "detail18": 0.10, "avg12": 0.08},
        detail_sigma=1.4,
        vi_detail_weight=0.025,
        target_std=44.0 / 255.0,
        std_amount=0.38,
        grad_cap=1.05,
        cap_amount=0.60,
    )
    add(
        "blend33_cross21_vi11",
        "official_blend",
        weights={"base33": 0.70, "cross21": 0.13, "vi11": 0.10, "avg12": 0.07},
        detail_sigma=1.5,
        vi_detail_weight=0.015,
        target_std=44.0 / 255.0,
        std_amount=0.36,
        grad_cap=1.04,
        cap_amount=0.65,
    )
    add(
        "blend33_r402_detail18",
        "official_blend",
        weights={"base33": 0.66, "r4_02": 0.22, "detail18": 0.06, "avg12": 0.06},
        detail_sigma=1.4,
        vi_detail_weight=0.015,
        target_std=43.0 / 255.0,
        std_amount=0.30,
        grad_cap=1.035,
        cap_amount=0.70,
    )
    add(
        "blend33_r404_avg12",
        "official_blend",
        weights={"base33": 0.72, "r4_04": 0.14, "avg12": 0.14},
        detail_sigma=1.4,
        vi_detail_weight=0.01,
        target_std=42.5 / 255.0,
        std_amount=0.28,
        grad_cap=1.035,
        cap_amount=0.72,
    )
    add(
        "blend33_r310_soft",
        "official_blend",
        weights={"base33": 0.72, "r3_10": 0.10, "avg12": 0.18},
        detail_sigma=1.4,
        vi_detail_weight=0.01,
        target_std=42.0 / 255.0,
        std_amount=0.26,
        grad_cap=1.03,
        cap_amount=0.72,
    )

    low_struct = {"base33": 0.82, "avg12": 0.18}
    add(
        "transfer18_light",
        "detail_transfer",
        low_weights=low_struct,
        detail_source="detail18",
        detail_sigma=1.6,
        detail_weight=0.075,
        detail_bias=0.25,
        edge_floor=0.92,
        thermal_weight=0.10,
        gate_sigma=1.2,
        gate_mix=0.45,
        vi_detail_weight=0.02,
        target_std=43.0 / 255.0,
        std_amount=0.30,
        grad_cap=1.035,
        cap_amount=0.72,
    )
    add(
        "transfer18_mid",
        "detail_transfer",
        low_weights={"base33": 0.78, "avg12": 0.16, "vi19": 0.06},
        detail_source="detail18",
        detail_sigma=1.6,
        detail_weight=0.115,
        detail_bias=0.30,
        edge_floor=0.86,
        thermal_weight=0.14,
        gate_sigma=1.15,
        gate_mix=0.50,
        vi_detail_weight=0.025,
        target_std=44.0 / 255.0,
        std_amount=0.38,
        grad_cap=1.05,
        cap_amount=0.66,
    )
    add(
        "transfer21_light",
        "detail_transfer",
        low_weights={"base33": 0.82, "avg12": 0.12, "vi11": 0.06},
        detail_source="cross21",
        detail_sigma=1.5,
        detail_weight=0.09,
        detail_bias=0.20,
        edge_floor=0.90,
        thermal_weight=0.10,
        gate_sigma=1.2,
        gate_mix=0.40,
        vi_detail_weight=0.02,
        target_std=43.5 / 255.0,
        std_amount=0.34,
        grad_cap=1.04,
        cap_amount=0.70,
    )
    add(
        "transfer_vi11_qabf",
        "detail_transfer",
        low_weights={"base33": 0.76, "avg12": 0.16, "vi19": 0.08},
        detail_source="vi11",
        detail_sigma=1.3,
        detail_weight=0.11,
        detail_bias=0.20,
        edge_floor=0.96,
        thermal_weight=0.06,
        gate_sigma=1.3,
        gate_mix=0.25,
        vi_detail_weight=0.015,
        target_std=43.0 / 255.0,
        std_amount=0.30,
        grad_cap=1.035,
        cap_amount=0.72,
    )
    add(
        "transfer_r404_micro",
        "detail_transfer",
        low_weights={"base33": 0.80, "avg12": 0.14, "vi19": 0.06},
        detail_source="r4_04",
        detail_sigma=1.4,
        detail_weight=0.18,
        detail_bias=0.15,
        edge_floor=0.94,
        thermal_weight=0.08,
        gate_sigma=1.3,
        gate_mix=0.35,
        vi_detail_weight=0.012,
        target_std=43.0 / 255.0,
        std_amount=0.30,
        grad_cap=1.035,
        cap_amount=0.74,
    )
    add(
        "transfer_r402_micro",
        "detail_transfer",
        low_weights={"base33": 0.80, "avg12": 0.16, "vi11": 0.04},
        detail_source="r4_02",
        detail_sigma=1.3,
        detail_weight=0.20,
        detail_bias=0.12,
        edge_floor=0.94,
        thermal_weight=0.08,
        gate_sigma=1.3,
        gate_mix=0.35,
        vi_detail_weight=0.012,
        target_std=42.5 / 255.0,
        std_amount=0.28,
        grad_cap=1.03,
        cap_amount=0.76,
    )

    add(
        "vi_repair_base33_11",
        "vi_profile_repair",
        low_weights={"base33": 0.70, "vi11": 0.14, "avg12": 0.16},
        low_sigma=6.0,
        detail_sigma=1.5,
        low_ir_repair=0.08,
        vi_detail_weight=0.045,
        ir_detail_weight=0.018,
        edge_floor=0.92,
        thermal_weight=0.10,
        gate_sigma=1.3,
        target_std=43.5 / 255.0,
        std_amount=0.34,
        grad_cap=1.04,
        cap_amount=0.68,
    )
    add(
        "vi_repair_base33_19",
        "vi_profile_repair",
        low_weights={"base33": 0.68, "vi19": 0.18, "avg12": 0.14},
        low_sigma=6.0,
        detail_sigma=1.5,
        low_ir_repair=0.10,
        vi_detail_weight=0.05,
        ir_detail_weight=0.02,
        edge_floor=0.88,
        thermal_weight=0.12,
        gate_sigma=1.2,
        target_std=44.0 / 255.0,
        std_amount=0.40,
        grad_cap=1.05,
        cap_amount=0.64,
    )
    add(
        "vi_repair_with18",
        "vi_profile_repair",
        low_weights={"base33": 0.66, "vi19": 0.16, "detail18": 0.08, "avg12": 0.10},
        low_sigma=6.0,
        detail_sigma=1.5,
        low_ir_repair=0.08,
        vi_detail_weight=0.045,
        ir_detail_weight=0.018,
        edge_floor=0.88,
        thermal_weight=0.12,
        gate_sigma=1.2,
        target_std=44.5 / 255.0,
        std_amount=0.42,
        grad_cap=1.055,
        cap_amount=0.62,
    )
    add(
        "vi_repair_highsd",
        "vi_profile_repair",
        low_weights={"base33": 0.62, "vi19": 0.16, "cross21": 0.08, "avg12": 0.14},
        low_sigma=6.0,
        detail_sigma=1.5,
        low_ir_repair=0.08,
        vi_detail_weight=0.055,
        ir_detail_weight=0.018,
        edge_floor=0.88,
        thermal_weight=0.12,
        gate_sigma=1.2,
        target_std=45.5 / 255.0,
        std_amount=0.48,
        grad_cap=1.06,
        cap_amount=0.60,
    )
    return out


def add_local_rank_scores(rows: list[dict[str, float | str]]) -> None:
    for metric, higher in HIGHER_IS_BETTER.items():
        order = sorted(range(len(rows)), key=lambda i: float(rows[i][metric]), reverse=higher)
        for rank, idx in enumerate(order, start=1):
            rows[idx][f"R_local_{metric}"] = rank
    for row in rows:
        row["LocalPseudoRank"] = float(np.mean([float(row[f"R_local_{metric}"]) for metric in HIGHER_IS_BETTER]))


def add_risk_scores(rows: list[dict[str, float | str]]) -> None:
    for row in rows:
        sd = float(row["SD"])
        nabf = float(row["Nabf"])
        ssim = float(row["SSIM"])
        base_corr = float(row["Base33Corr"])
        qabf = float(row["Qabf"])
        mi = float(row["MI"])
        score = float(row["LocalPseudoRank"])
        score += abs(sd - 44.5 / 255.0) * 25.0
        score += max(0.0, 0.045 - (qabf - 0.60)) * 2.0
        score += max(0.0, 2.65 - mi) * 1.0
        score += max(0.0, nabf - 0.065) * 22.0
        score += max(0.0, 0.699 - ssim) * 18.0
        score += max(0.0, 0.9975 - base_corr) * 120.0
        row["Round5Heuristic"] = float(score)


def locate_official_local_metrics(name: str) -> dict[str, float] | None:
    for root in OFFICIAL_PATH_ROOTS:
        path = root / name / "metrics_summary.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def add_official_neighbor_scores(rows: list[dict[str, float | str]], official_csv: Path) -> None:
    if not official_csv.exists():
        return
    official_rows = list(csv.DictReader(official_csv.open("r", encoding="utf-8-sig", newline="")))
    anchors: list[tuple[str, float, dict[str, float]]] = []
    for official in official_rows:
        local = locate_official_local_metrics(official["candidate"])
        if local is None:
            continue
        anchors.append((official["candidate"], float(official["inserted_rank"]), {m: float(local[m]) for m in HIGHER_IS_BETTER}))
    if len(anchors) < 3:
        return

    metrics = list(HIGHER_IS_BETTER)
    matrix = np.array([[anchor[2][m] for m in metrics] for anchor in anchors] + [[float(row[m]) for m in metrics] for row in rows])
    scale = matrix.std(axis=0) + 1e-6
    anchor_matrix = np.array([[anchor[2][m] for m in metrics] for anchor in anchors])
    anchor_z = (anchor_matrix - matrix.mean(axis=0)) / scale
    for row in rows:
        cand = np.array([float(row[m]) for m in metrics])
        z = (cand - matrix.mean(axis=0)) / scale
        dists = np.sqrt(np.mean((anchor_z - z) ** 2, axis=1))
        order = np.argsort(dists)[:3]
        weights = 1.0 / (dists[order] + 1e-4)
        pred = float(np.sum(weights * np.array([anchors[i][1] for i in order])) / np.sum(weights))
        row["NearestOfficialRank"] = pred
        row["NearestOfficialAnchors"] = ";".join(f"{anchors[i][0]}:{anchors[i][1]:.2f}" for i in order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--out-root", default="results/candidates_round5")
    parser.add_argument("--score-out", default="results/round5_local_scores.csv")
    parser.add_argument("--official-csv", default="results/score_tables/current_official_scores.csv")
    args = parser.parse_args()

    dirs = {"ir": Path(args.ir_dir), "vi": Path(args.vi_dir)}
    dirs.update({key: Path(value) for key, value in SOURCE_DEFAULTS.items()})
    for key, path in dirs.items():
        if not path.exists():
            raise FileNotFoundError(f"{key} not found: {path}")

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
        base_corr_vals = []
        for image_name in names:
            base_corr_vals.append(corrcoef(gray_from_source(image_name, "base33", dirs), gray_float(load_image(cand_dir / image_name))))
        row: dict[str, float | str] = {
            "candidate": cand_dir.name,
            "path": str(cand_dir),
            "family": str(params["family"]),
            "Base33Corr": float(np.mean(base_corr_vals)),
        }
        row.update(summary)
        rows.append(row)

    add_local_rank_scores(rows)
    add_risk_scores(rows)
    add_official_neighbor_scores(rows, Path(args.official_csv))
    rows.sort(key=lambda row: (float(row["Round5Heuristic"]), float(row["LocalPseudoRank"])))

    fieldnames = [
        "candidate",
        "path",
        "family",
        "Round5Heuristic",
        "LocalPseudoRank",
        "Base33Corr",
        "NearestOfficialRank",
        "NearestOfficialAnchors",
    ] + list(HIGHER_IS_BETTER) + [f"R_local_{metric}" for metric in HIGHER_IS_BETTER]
    for row in rows:
        row.setdefault("NearestOfficialRank", "")
        row.setdefault("NearestOfficialAnchors", "")
    score_out = Path(args.score_out)
    ensure_dir(score_out.parent)
    with score_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{float(row['Round5Heuristic']):.3f} local={float(row['LocalPseudoRank']):.3f} "
            f"corr={float(row['Base33Corr']):.5f} {row['candidate']} "
            f"near={row.get('NearestOfficialRank', '')}"
        )
    print(f"wrote {score_out}")


if __name__ == "__main__":
    main()
