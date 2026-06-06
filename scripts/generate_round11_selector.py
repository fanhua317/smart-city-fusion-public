from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fusion_common import ensure_dir, gray_float, list_pairs, load_image
from generate_round7_leader_target import LEADER_ALIASES, LOWER_IS_BETTER, METRICS, load_leaderboard, metric_rank
from inspect_dataset import inspect_dataset, write_profile
from local_metrics import HIGHER_IS_BETTER, evaluate_dir, metrics_for_pair, write_csv


MANUAL_POOL = [
    "results/candidates_round10/03_rankpoints_detail_rank",
    "results/candidates_round2/33_mix_avg21_45",
    "results/candidates_round10/05_rankpoints_safe_qabf_rank",
    "results/candidates_round10/06_rankpoints_safe_qabf_balanced",
    "results/candidates_round10/01_rankpoints_balanced_rank",
    "results/candidates_round11_adaptive/01_A_FLIR_flir_ir50_std450_d76_01",
    "results/candidates_round11_adaptive/17_B_D_day_vi46_gate18_std438_01",
    "results/candidates_round11_adaptive/33_C_N_night_g86_str14_ir26_01",
    "results/candidates_round11_adaptive/49_D_GRAY_gray_vi40_std438_d60_01",
    "results/candidates_round11_deep/crossfuse_01_low03_highdeep_s45_cap",
    "results/candidates_remote/crossfuse_target",
]


@dataclass
class Calibrator:
    metric: str
    method: str
    slope: float
    intercept: float
    xs: list[float]
    ys: list[float]

    def predict(self, value: float) -> float:
        if self.method == "linear":
            return float(self.slope * value + self.intercept)
        order = np.argsort(np.asarray(self.xs, dtype=np.float64))
        xs = np.asarray(self.xs, dtype=np.float64)[order]
        ys = np.asarray(self.ys, dtype=np.float64)[order]
        if len(xs) == 0:
            return float(value)
        if len(xs) == 1:
            return float(ys[0])
        if self.method == "isotonic":
            # The fitted isotonic knots are stored in xs/ys.
            return float(np.interp(value, xs, ys, left=ys[0], right=ys[-1]))
        return float(np.interp(value, xs, ys, left=ys[0], right=ys[-1]))


def linear_fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    mx = float(xs.mean())
    my = float(ys.mean())
    var = float(np.sum((xs - mx) ** 2)) + 1e-12
    slope = float(np.sum((xs - mx) * (ys - my)) / var)
    intercept = my - slope * mx
    pred = slope * xs + intercept
    ss = float(np.sum((ys - my) ** 2)) + 1e-12
    r2 = 1.0 - float(np.sum((ys - pred) ** 2)) / ss
    mae = float(np.mean(np.abs(ys - pred)))
    return slope, intercept, r2, mae


def loo_mae(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) <= 2:
        return 0.0
    errors = []
    for idx in range(len(xs)):
        mask = np.ones(len(xs), dtype=bool)
        mask[idx] = False
        slope, intercept, _, _ = linear_fit(xs[mask], ys[mask])
        errors.append(abs(float(ys[idx]) - float(slope * xs[idx] + intercept)))
    return float(np.mean(errors))


def isotonic_or_quantile(xs: np.ndarray, ys: np.ndarray) -> tuple[str, list[float], list[float]]:
    order = np.argsort(xs)
    sorted_x = xs[order]
    sorted_y = ys[order]
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore

        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        fitted = iso.fit_transform(sorted_x, sorted_y)
        return "isotonic", [float(x) for x in sorted_x], [float(y) for y in fitted]
    except Exception:  # noqa: BLE001 - sklearn is optional.
        quant_x = np.quantile(sorted_x, np.linspace(0, 1, min(len(sorted_x), 7)))
        quant_y = np.quantile(sorted_y, np.linspace(0, 1, min(len(sorted_y), 7)))
        return "quantile", [float(x) for x in quant_x], [float(y) for y in quant_y]


def find_local_summary(candidate: str) -> dict[str, float] | None:
    for root in sorted(Path("results").glob("candidates*")) + sorted(Path("results").glob("candidates_remote*")):
        path = root / candidate / "metrics_summary.json"
        if path.exists():
            return {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items() if k in METRICS}
    return None


def build_calibration(official_csv: Path, report_out: Path) -> tuple[dict[str, Calibrator], dict[str, Any]]:
    pairs: list[tuple[dict[str, str], dict[str, float]]] = []
    with official_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            local = find_local_summary(row["candidate"])
            if local is not None:
                pairs.append((row, local))
    if len(pairs) < 2:
        raise RuntimeError(f"Need at least two official/local pairs for calibration, got {len(pairs)}")

    calibrators: dict[str, Calibrator] = {}
    report: dict[str, Any] = {"pair_count": len(pairs), "metrics": {}}
    for metric in METRICS:
        xs = np.asarray([local[metric] for _, local in pairs], dtype=np.float64)
        ys = np.asarray([float(row[metric]) for row, _ in pairs], dtype=np.float64)
        slope, intercept, r2, mae = linear_fit(xs, ys)
        loo = loo_mae(xs, ys)
        method = "linear"
        fit_xs = [float(x) for x in xs]
        fit_ys = [float(y) for y in ys]
        if r2 < 0.80:
            method, fit_xs, fit_ys = isotonic_or_quantile(xs, ys)
        calibrators[metric] = Calibrator(metric, method, slope, intercept, fit_xs, fit_ys)
        report["metrics"][metric] = {
            "r2": r2,
            "mae": mae,
            "loo_mae": loo,
            "method": method,
            "linear_slope": slope,
            "linear_intercept": intercept,
        }
    ensure_dir(report_out.parent)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return calibrators, report


def predict_official(local: dict[str, float], calibration: dict[str, Calibrator]) -> dict[str, float]:
    pred: dict[str, float] = {}
    for metric in METRICS:
        pred[metric] = calibration[metric].predict(float(local[metric]))
    return pred


def base_rank(pred: dict[str, float], leaderboard: list[dict[str, str]]) -> float:
    return float(np.mean([metric_rank(leaderboard, metric, pred[metric]) for metric in METRICS]))


def threshold_penalty(pred: dict[str, float], mode: str, baseline: dict[str, float] | None) -> float:
    nabf_threshold = 0.045
    ssim_threshold = 0.745
    cc_threshold = 0.610
    scd_threshold = 1.700
    qabf_threshold = 0.560
    ag_threshold = 5.95
    sf_threshold = 7.25
    sd_threshold = 43.5
    weights = {
        "Nabf": 10.0,
        "SSIM": 8.0,
        "CC": 6.0,
        "SCD": 5.0,
        "Qabf": 4.0,
        "AG": 2.0,
        "SF": 2.0,
        "SD": 1.5,
    }
    extra = 0.0
    if mode == "attack_qabf":
        qabf_threshold = 0.590
        nabf_threshold = 0.040
        ssim_threshold = 0.750
        weights["Nabf"] = 18.0
        weights["SSIM"] = 14.0
        weights["Qabf"] = 8.0
    elif mode == "attack_mi":
        extra += 8.0 * max(0.0, 1.25 - pred["MI"])
        weights["CC"] = 14.0
        weights["SCD"] = 12.0
        weights["SSIM"] = 10.0
    elif mode == "safe_rank" and baseline is not None:
        margins = {
            "AG": 0.22,
            "CC": 0.010,
            "EN": 0.10,
            "MI": 0.08,
            "MSE": 0.010,
            "Nabf": 0.008,
            "PSNR": 0.25,
            "Qabf": 0.018,
            "SCD": 0.040,
            "SD": 1.20,
            "SF": 0.20,
            "SSIM": 0.010,
            "VIF": 0.030,
        }
        for metric in METRICS:
            if metric in LOWER_IS_BETTER:
                extra += 60.0 * max(0.0, pred[metric] - baseline[metric] - margins[metric])
            else:
                extra += 60.0 * max(0.0, baseline[metric] - margins[metric] - pred[metric])
    if mode.startswith("group_FLIR"):
        ag_threshold, sf_threshold, qabf_threshold = 6.05, 7.35, 0.570
        weights["AG"], weights["SF"], weights["Qabf"] = 5.0, 5.0, 7.0
    elif mode.startswith("group_DN_D"):
        ssim_threshold, cc_threshold, scd_threshold, qabf_threshold = 0.750, 0.615, 1.700, 0.560
        weights["SSIM"], weights["CC"], weights["SCD"] = 12.0, 10.0, 8.0
    elif mode.startswith("group_DN_N"):
        sd_threshold, qabf_threshold = 44.0, 0.565
        extra += 5.0 * max(0.0, 1.18 - pred["MI"])
        weights["SD"], weights["Qabf"], weights["AG"] = 5.0, 6.0, 4.0
    elif mode.startswith("group_GRAY_VI"):
        ssim_threshold, qabf_threshold, nabf_threshold = 0.750, 0.570, 0.040
        weights["SSIM"], weights["Qabf"], weights["Nabf"] = 12.0, 8.0, 16.0

    return float(
        weights["Nabf"] * max(0.0, pred["Nabf"] - nabf_threshold)
        + weights["SSIM"] * max(0.0, ssim_threshold - pred["SSIM"])
        + weights["CC"] * max(0.0, cc_threshold - pred["CC"])
        + weights["SCD"] * max(0.0, scd_threshold - pred["SCD"])
        + weights["Qabf"] * max(0.0, qabf_threshold - pred["Qabf"])
        + weights["AG"] * max(0.0, ag_threshold - pred["AG"])
        + weights["SF"] * max(0.0, sf_threshold - pred["SF"])
        + weights["SD"] * max(0.0, sd_threshold - pred["SD"])
        + extra
    )


def objective_detail(
    local: dict[str, float],
    leaderboard: list[dict[str, str]],
    calibration: dict[str, Calibrator],
    mode: str,
    baseline: dict[str, float] | None,
) -> dict[str, Any]:
    pred = predict_official(local, calibration)
    rank = base_rank(pred, leaderboard)
    penalty = threshold_penalty(pred, mode, baseline)
    ranks = {f"Pred_R_{metric}": metric_rank(leaderboard, metric, pred[metric]) for metric in METRICS}
    return {"Objective": rank + penalty, "PredInsertedRank": rank, "Penalty": penalty, "pred": pred, "ranks": ranks}


def read_profile(path: Path, ir_dir: Path, vi_dir: Path) -> dict[str, set[str]]:
    if not path.exists():
        write_profile(inspect_dataset(ir_dir, vi_dir), path)
    out: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["image"]] = {tag for tag in row["group"].split(";") if tag}
    return out


def candidate_dirs(names: list[str]) -> list[Path]:
    seen: set[Path] = set()
    dirs: list[Path] = []
    for metrics_path in sorted(Path("results").glob("candidates*/**/metrics_summary.json")):
        path = metrics_path.parent
        if path.resolve() in seen:
            continue
        if all((path / name).exists() for name in names):
            seen.add(path.resolve())
            dirs.append(path)
    for raw in MANUAL_POOL:
        path = Path(raw)
        if path.exists() and path.resolve() not in seen and all((path / name).exists() for name in names):
            seen.add(path.resolve())
            dirs.append(path)
    return dirs


def load_summary(path: Path) -> dict[str, float] | None:
    p = path / "metrics_summary.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if not all(metric in data for metric in METRICS):
        return None
    return {metric: float(data[metric]) for metric in METRICS}


def choose_pool(
    dirs: list[Path],
    leaderboard: list[dict[str, str]],
    calibration: dict[str, Calibrator],
    baseline: dict[str, float] | None,
    pool_size: int,
) -> list[Path]:
    scored = []
    for path in dirs:
        local = load_summary(path)
        if local is None:
            continue
        detail = objective_detail(local, leaderboard, calibration, "safe_rank", baseline)
        scored.append((float(detail["Objective"]), path))
    scored.sort(key=lambda item: item[0])
    chosen: list[Path] = [path for _, path in scored[:pool_size]]
    seen = {path.resolve() for path in chosen}
    for raw in MANUAL_POOL:
        path = Path(raw)
        if path.exists() and path.resolve() not in seen:
            chosen.append(path)
            seen.add(path.resolve())
    return chosen


def rank_points(values: np.ndarray, higher: bool) -> np.ndarray:
    order = np.argsort(values)
    if higher:
        order = order[::-1]
    points = np.zeros(len(values), dtype=np.float64)
    denom = max(1, len(values) - 1)
    for rank, idx in enumerate(order):
        points[idx] = 1.0 - rank / denom
    return points


def rankpoint_selection(metrics_cube: np.ndarray, metric_index: dict[str, int], profile: str, tags: list[set[str]]) -> np.ndarray:
    weight_profiles = {
        "safe": {"Nabf": 2.0, "SSIM": 1.7, "CC": 1.4, "SCD": 1.3, "Qabf": 1.4, "AG": 0.7, "SF": 0.7, "SD": 0.5},
        "qabf": {"Qabf": 2.8, "Nabf": 1.8, "SSIM": 1.3, "AG": 1.2, "SF": 1.2, "CC": 0.9, "SCD": 0.9},
        "mi": {"MI": 2.0, "EN": 1.3, "SD": 1.1, "CC": 1.2, "SCD": 1.2, "SSIM": 1.0, "Nabf": 1.0},
        "balanced": {metric: 1.0 for metric in METRICS},
    }
    weights = weight_profiles[profile]
    n_images, n_candidates, _ = metrics_cube.shape
    selection = np.zeros(n_images, dtype=np.int32)
    for image_idx in range(n_images):
        score = np.zeros(n_candidates, dtype=np.float64)
        for metric, weight in weights.items():
            score += weight * rank_points(metrics_cube[image_idx, :, metric_index[metric]], HIGHER_IS_BETTER[metric])
        if "FLIR" in tags[image_idx]:
            score += 0.8 * rank_points(metrics_cube[image_idx, :, metric_index["Qabf"]], True)
        if "DN_D" in tags[image_idx]:
            score += 0.8 * rank_points(metrics_cube[image_idx, :, metric_index["SSIM"]], True)
        if "DN_N" in tags[image_idx]:
            score += 0.8 * rank_points(metrics_cube[image_idx, :, metric_index["MI"]], True)
        if "GRAY_VI" in tags[image_idx]:
            score += 0.8 * rank_points(metrics_cube[image_idx, :, metric_index["Nabf"]], False)
        selection[image_idx] = int(np.argmax(score))
    return selection


def local_from_vec(vec: np.ndarray, metric_index: dict[str, int]) -> dict[str, float]:
    return {metric: float(vec[metric_index[metric]]) for metric in METRICS}


def aggregate_vec(metrics_cube: np.ndarray, selection: np.ndarray) -> np.ndarray:
    return metrics_cube[np.arange(metrics_cube.shape[0]), selection, :].mean(axis=0)


def fixed_selection(pool: list[Path], names: list[str], name_part: str) -> np.ndarray | None:
    for idx, path in enumerate(pool):
        if name_part in path.name:
            return np.full(len(names), idx, dtype=np.int32)
    return None


def greedy_coordinate(
    initial: np.ndarray,
    metrics_cube: np.ndarray,
    metric_index: dict[str, int],
    leaderboard: list[dict[str, str]],
    calibration: dict[str, Calibrator],
    mode: str,
    baseline: dict[str, float] | None,
    image_indices: list[int] | None = None,
    passes: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    selection = initial.copy()
    n_images, n_candidates, _ = metrics_cube.shape
    indices = image_indices if image_indices is not None else list(range(n_images))
    agg = aggregate_vec(metrics_cube, selection)
    best = objective_detail(local_from_vec(agg, metric_index), leaderboard, calibration, mode, baseline)
    for _ in range(passes):
        changed = False
        for image_idx in indices:
            current = int(selection[image_idx])
            current_vec = metrics_cube[image_idx, current, :]
            best_candidate = current
            best_vec = agg
            for cand_idx in range(n_candidates):
                if cand_idx == current:
                    continue
                trial_vec = agg + (metrics_cube[image_idx, cand_idx, :] - current_vec) / n_images
                detail = objective_detail(local_from_vec(trial_vec, metric_index), leaderboard, calibration, mode, baseline)
                if float(detail["Objective"]) + 1e-9 < float(best["Objective"]):
                    best = detail
                    best_candidate = cand_idx
                    best_vec = trial_vec
            if best_candidate != current:
                selection[image_idx] = best_candidate
                agg = best_vec
                changed = True
        if not changed:
            break
    return selection, best


def beam_search(
    starts: list[np.ndarray],
    metrics_cube: np.ndarray,
    metric_index: dict[str, int],
    leaderboard: list[dict[str, str]],
    calibration: dict[str, Calibrator],
    baseline: dict[str, float] | None,
    width: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_images, n_candidates, _ = metrics_cube.shape
    states = []
    for sel in starts:
        vec = aggregate_vec(metrics_cube, sel)
        detail = objective_detail(local_from_vec(vec, metric_index), leaderboard, calibration, "safe_rank", baseline)
        states.append((float(detail["Objective"]), sel.copy(), vec, detail))
    states.sort(key=lambda x: x[0])
    states = states[:width]
    for image_idx in range(n_images):
        candidates = []
        for _, sel, vec, _ in states:
            current = int(sel[image_idx])
            current_vec = metrics_cube[image_idx, current, :]
            for cand_idx in range(n_candidates):
                if cand_idx == current:
                    continue
                trial_sel = sel.copy()
                trial_sel[image_idx] = cand_idx
                trial_vec = vec + (metrics_cube[image_idx, cand_idx, :] - current_vec) / n_images
                detail = objective_detail(local_from_vec(trial_vec, metric_index), leaderboard, calibration, "safe_rank", baseline)
                candidates.append((float(detail["Objective"]), trial_sel, trial_vec, detail))
        candidates.extend(states)
        candidates.sort(key=lambda x: x[0])
        dedup: list[tuple[float, np.ndarray, np.ndarray, dict[str, Any]]] = []
        seen: set[bytes] = set()
        for state in candidates:
            key = state[1].tobytes()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(state)
            if len(dedup) >= width:
                break
        states = dedup
    return states[0][1], states[0][3]


def simulated_annealing(
    starts: list[np.ndarray],
    metrics_cube: np.ndarray,
    metric_index: dict[str, int],
    leaderboard: list[dict[str, str]],
    calibration: dict[str, Calibrator],
    baseline: dict[str, float] | None,
    tags: list[set[str]],
    seeds: list[int],
    iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_images, n_candidates, _ = metrics_cube.shape
    group_indices = {
        group: [idx for idx, row_tags in enumerate(tags) if group in row_tags]
        for group in ["FLIR", "DN_D", "DN_N", "GRAY_VI"]
    }
    best_sel = starts[0].copy()
    best_vec = aggregate_vec(metrics_cube, best_sel)
    best_detail = objective_detail(local_from_vec(best_vec, metric_index), leaderboard, calibration, "safe_rank", baseline)
    for seed in seeds:
        rng = random.Random(seed)
        current = starts[seed % len(starts)].copy()
        current_vec = aggregate_vec(metrics_cube, current)
        current_detail = objective_detail(local_from_vec(current_vec, metric_index), leaderboard, calibration, "safe_rank", baseline)
        for step in range(iterations):
            temp = 0.15 * (0.005 / 0.15) ** (step / max(1, iterations - 1))
            if rng.random() < 0.30:
                group = rng.choice([g for g, idxs in group_indices.items() if idxs])
                change_indices = rng.sample(group_indices[group], k=min(len(group_indices[group]), rng.randint(2, 5)))
            else:
                change_indices = [rng.randrange(n_images)]
            trial = current.copy()
            trial_vec = current_vec.copy()
            for image_idx in change_indices:
                old = int(trial[image_idx])
                new = rng.randrange(n_candidates)
                if new == old:
                    continue
                trial[image_idx] = new
                trial_vec += (metrics_cube[image_idx, new, :] - metrics_cube[image_idx, old, :]) / n_images
            detail = objective_detail(local_from_vec(trial_vec, metric_index), leaderboard, calibration, "safe_rank", baseline)
            delta = float(detail["Objective"]) - float(current_detail["Objective"])
            if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-6)):
                current = trial
                current_vec = trial_vec
                current_detail = detail
                if float(detail["Objective"]) < float(best_detail["Objective"]):
                    best_sel = trial.copy()
                    best_vec = trial_vec.copy()
                    best_detail = detail
    return best_sel, best_detail


def groupwise_selection(
    start: np.ndarray,
    metrics_cube: np.ndarray,
    metric_index: dict[str, int],
    leaderboard: list[dict[str, str]],
    calibration: dict[str, Calibrator],
    baseline: dict[str, float] | None,
    tags: list[set[str]],
) -> tuple[np.ndarray, dict[str, Any]]:
    selection = start.copy()
    detail = objective_detail(local_from_vec(aggregate_vec(metrics_cube, selection), metric_index), leaderboard, calibration, "safe_rank", baseline)
    for group in ["FLIR", "DN_D", "DN_N", "GRAY_VI"]:
        indices = [idx for idx, row_tags in enumerate(tags) if group in row_tags]
        if not indices:
            continue
        selection, detail = greedy_coordinate(
            selection,
            metrics_cube,
            metric_index,
            leaderboard,
            calibration,
            f"group_{group}",
            baseline,
            image_indices=indices,
            passes=2,
        )
    final_detail = objective_detail(local_from_vec(aggregate_vec(metrics_cube, selection), metric_index), leaderboard, calibration, "safe_rank", baseline)
    return selection, final_detail


def validate_candidate(out_dir: Path, names: list[str], ir_dir: Path) -> dict[str, Any]:
    problems: list[str] = []
    stats = []
    for name in names:
        path = out_dir / name
        if not path.exists():
            problems.append(f"missing {name}")
            continue
        fused = load_image(path)
        ir_img = load_image(ir_dir / name)
        if fused.size != ir_img.size:
            problems.append(f"size {name}: {fused.size} != {ir_img.size}")
        arr = gray_float(fused)
        if not np.isfinite(arr).all():
            problems.append(f"nonfinite {name}")
        if float(arr.std()) < 1e-4:
            problems.append(f"flat {name}")
        stats.append({"image": name, "mean": float(arr.mean()), "std": float(arr.std()), "size": fused.size})
    return {"count": len(stats), "expected": len(names), "problems": problems, "stats": stats}


def clean_image_dir(path: Path) -> None:
    ensure_dir(path)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def write_final_candidate(
    out_dir: Path,
    eval_dir: Path,
    names: list[str],
    pool: list[Path],
    selection: np.ndarray,
    ir_dir: Path,
    vi_dir: Path,
    detail: dict[str, Any],
    method: str,
    metric_index: dict[str, int],
    metrics_cube: np.ndarray,
) -> dict[str, Any]:
    clean_image_dir(out_dir)
    ensure_dir(eval_dir)
    manifest = []
    source_counts: dict[str, int] = {}
    for image_idx, name in enumerate(names):
        source_dir = pool[int(selection[image_idx])]
        shutil.copy2(source_dir / name, out_dir / name)
        source_counts[source_dir.name] = source_counts.get(source_dir.name, 0) + 1
        manifest.append({"image": name, "source": str(source_dir), "source_candidate": source_dir.name})
    metric_rows, summary = evaluate_dir(ir_dir, vi_dir, out_dir)
    write_csv(eval_dir / "metrics.csv", metric_rows)
    (eval_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = validate_candidate(out_dir, names, ir_dir)
    (eval_dir / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (eval_dir / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (eval_dir / "source_counts.json").write_text(json.dumps(source_counts, ensure_ascii=False, indent=2), encoding="utf-8")
    (eval_dir / "objective.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    row: dict[str, Any] = {
        "candidate": out_dir.name,
        "path": str(out_dir),
        "method": method,
        "Objective": float(detail["Objective"]),
        "PredInsertedRank": float(detail["PredInsertedRank"]),
        "Penalty": float(detail["Penalty"]),
        "SourceCount": len(source_counts),
        "TopSources": json.dumps(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10], ensure_ascii=False),
        "ValidationProblems": len(validation["problems"]),
    }
    row.update(summary)
    for metric in METRICS:
        row[f"Pred_{metric}"] = float(detail["pred"][metric])
        row[f"Pred_R_{metric}"] = int(detail["ranks"][f"Pred_R_{metric}"])
    return row


def write_upload_strategy(path: Path, rows: list[dict[str, Any]], current_rank: float) -> None:
    by_name = {row["candidate"]: row for row in rows}
    order: list[str] = []
    primary = min([row for row in rows if row["candidate"] in {"r11_09_anneal_best", "r11_08_groupwise_best"}], key=lambda r: float(r["PredInsertedRank"]))
    if float(primary["PredInsertedRank"]) <= current_rank - 1.5:
        order.append(str(primary["candidate"]))
    order.extend(["r11_02_attack_qabf", "r11_01_safe_rank"])
    deep_options = [name for name in ["r11_06_deep_crossfuse_blend", "r11_07_cddfuse_blend"] if name in by_name]
    if deep_options:
        order.append(min(deep_options, key=lambda name: float(by_name[name]["PredInsertedRank"])))
    order.append("reserve_after_feedback")
    dedup: list[str] = []
    for item in order:
        if item not in dedup:
            dedup.append(item)
    lines = [
        "# Round11 Upload Strategy",
        "",
        "Daily upload limit: use at most five submissions.",
        "",
        "| Order | Candidate | PredInsertedRank | Note |",
        "| ---: | --- | ---: | --- |",
    ]
    for idx, name in enumerate(dedup[:5], 1):
        if name == "reserve_after_feedback":
            lines.append(f"| {idx} | reserve_after_feedback |  | Recalibrate after the first official feedback rows. |")
            continue
        note = "primary global optimizer" if name == primary["candidate"] else "diagnostic/backup"
        lines.append(f"| {idx} | {name} | {float(by_name[name]['PredInsertedRank']):.4f} | {note} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--profile", default="results/dataset_profile.csv")
    parser.add_argument("--leaderboard", default="leaderboard_20260605_234547.csv")
    parser.add_argument("--official-csv", default="results/score_tables/current_official_scores.csv")
    parser.add_argument("--out-root", default="results/candidates_round11")
    parser.add_argument("--eval-root", default="results/round11_eval")
    parser.add_argument("--score-out", default="results/round11_local_scores.csv")
    parser.add_argument("--pool-out", default="results/round11_pool.csv")
    parser.add_argument("--calibration-report", default="results/round11_calibration_report.json")
    parser.add_argument("--pool-size", type=int, default=70)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--anneal-iters", type=int, default=2400)
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir)
    vi_dir = Path(args.vi_dir)
    names = list_pairs(ir_dir, vi_dir)
    profile = read_profile(Path(args.profile), ir_dir, vi_dir)
    tags = [profile[name] for name in names]
    leaderboard = load_leaderboard(Path(args.leaderboard))
    calibration, _ = build_calibration(Path(args.official_csv), Path(args.calibration_report))
    baseline_row = None
    with Path(args.official_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("candidate") == "03_rankpoints_detail_rank":
                baseline_row = {metric: float(row[metric]) for metric in METRICS}
    all_dirs = candidate_dirs(names)
    pool = choose_pool(all_dirs, leaderboard, calibration, baseline_row, args.pool_size)
    print(f"pool={len(pool)} all_complete={len(all_dirs)}")

    metric_index = {metric: idx for idx, metric in enumerate(METRICS)}
    metrics_cube = np.zeros((len(names), len(pool), len(METRICS)), dtype=np.float64)
    for image_idx, name in enumerate(names):
        ir = gray_float(load_image(ir_dir / name))
        vi = gray_float(load_image(vi_dir / name))
        for cand_idx, cand_dir in enumerate(pool):
            fused = gray_float(load_image(cand_dir / name))
            row = metrics_for_pair(ir, vi, fused)
            for metric in METRICS:
                metrics_cube[image_idx, cand_idx, metric_index[metric]] = row[metric]
        if (image_idx + 1) % 10 == 0 or image_idx + 1 == len(names):
            print(f"metrics {image_idx + 1}/{len(names)}")

    pool_rows = []
    for path in pool:
        local = load_summary(path)
        if local is None:
            local = local_from_vec(metrics_cube[:, pool.index(path), :].mean(axis=0), metric_index)
        detail = objective_detail(local, leaderboard, calibration, "safe_rank", baseline_row)
        row = {"candidate": path.name, "path": str(path), "PoolObjective": float(detail["Objective"]), "PoolPredInsertedRank": float(detail["PredInsertedRank"])}
        row.update(local)
        pool_rows.append(row)
    pool_rows.sort(key=lambda r: float(r["PoolObjective"]))
    with Path(args.pool_out).open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["candidate", "path", "PoolObjective", "PoolPredInsertedRank"] + METRICS
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pool_rows)

    starts: list[np.ndarray] = []
    for profile_name in ["safe", "qabf", "mi", "balanced"]:
        starts.append(rankpoint_selection(metrics_cube, metric_index, profile_name, tags))
    for fixed in ["03_rankpoints_detail_rank", "33_mix_avg21_45", "rankpoints_safe_qabf", "crossfuse"]:
        fixed_sel = fixed_selection(pool, names, fixed)
        if fixed_sel is not None:
            starts.append(fixed_sel)

    results: dict[str, tuple[np.ndarray, dict[str, Any], str]] = {}
    safe_sel, safe_detail = greedy_coordinate(starts[0], metrics_cube, metric_index, leaderboard, calibration, "safe_rank", baseline_row, passes=3)
    results["r11_01_safe_rank"] = (safe_sel, safe_detail, "coordinate_safe_rank")
    qabf_sel, qabf_detail = greedy_coordinate(starts[1], metrics_cube, metric_index, leaderboard, calibration, "attack_qabf", baseline_row, passes=3)
    results["r11_02_attack_qabf"] = (qabf_sel, qabf_detail, "coordinate_attack_qabf")
    mi_sel, mi_detail = greedy_coordinate(starts[2], metrics_cube, metric_index, leaderboard, calibration, "attack_mi", baseline_row, passes=3)
    results["r11_03_attack_mi"] = (mi_sel, mi_detail, "coordinate_attack_mi")

    flir_indices = [idx for idx, row_tags in enumerate(tags) if "FLIR" in row_tags]
    night_indices = [idx for idx, row_tags in enumerate(tags) if "DN_N" in row_tags]
    flir_sel, flir_detail = greedy_coordinate(safe_sel, metrics_cube, metric_index, leaderboard, calibration, "group_FLIR", baseline_row, flir_indices, passes=3)
    night_sel, night_detail = greedy_coordinate(safe_sel, metrics_cube, metric_index, leaderboard, calibration, "group_DN_N", baseline_row, night_indices, passes=3)
    results["r11_04_flir_boost"] = (flir_sel, flir_detail, "group_flir_boost")
    results["r11_05_night_boost"] = (night_sel, night_detail, "group_night_boost")

    cross_sel = fixed_selection(pool, names, "crossfuse")
    if cross_sel is None:
        cross_sel = safe_sel
    cross_sel, cross_detail = greedy_coordinate(cross_sel, metrics_cube, metric_index, leaderboard, calibration, "safe_rank", baseline_row, passes=2)
    results["r11_06_deep_crossfuse_blend"] = (cross_sel, cross_detail, "deep_crossfuse_blend")
    cdd_sel = fixed_selection(pool, names, "cddfuse")
    if cdd_sel is None:
        cdd_sel = cross_sel.copy()
    cdd_sel, cdd_detail = greedy_coordinate(cdd_sel, metrics_cube, metric_index, leaderboard, calibration, "attack_qabf", baseline_row, passes=2)
    results["r11_07_cddfuse_blend"] = (cdd_sel, cdd_detail, "deep_cddfuse_or_fallback_blend")

    group_sel, group_detail = groupwise_selection(safe_sel, metrics_cube, metric_index, leaderboard, calibration, baseline_row, tags)
    results["r11_08_groupwise_best"] = (group_sel, group_detail, "group_aware_selector")
    anneal_sel, anneal_detail = simulated_annealing(starts, metrics_cube, metric_index, leaderboard, calibration, baseline_row, tags, [3, 11, 29, 41], args.anneal_iters)
    results["r11_09_anneal_best"] = (anneal_sel, anneal_detail, "simulated_annealing")
    beam_sel, beam_detail = beam_search(starts[:6], metrics_cube, metric_index, leaderboard, calibration, baseline_row, args.beam_width)
    results["r11_10_beam_best"] = (beam_sel, beam_detail, "beam_coordinate_descent")

    out_root = ensure_dir(args.out_root)
    eval_root = ensure_dir(args.eval_root)
    rows: list[dict[str, Any]] = []
    for name, (selection, detail, method) in results.items():
        row = write_final_candidate(out_root / name, eval_root / name, names, pool, selection, ir_dir, vi_dir, detail, method, metric_index, metrics_cube)
        rows.append(row)

    rows.sort(key=lambda row: float(row["PredInsertedRank"]))
    fields = (
        ["candidate", "path", "method", "Objective", "PredInsertedRank", "Penalty", "SourceCount", "TopSources", "ValidationProblems"]
        + METRICS
        + [f"Pred_{metric}" for metric in METRICS]
        + [f"Pred_R_{metric}" for metric in METRICS]
    )
    with Path(args.score_out).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    current_rank = float(baseline_row.get("inserted_rank", 22.3077)) if baseline_row and "inserted_rank" in baseline_row else 22.3077
    write_upload_strategy(eval_root / "recommended_upload_order.md", rows, current_rank)
    for row in rows:
        print(
            f"{row['candidate']} pred={float(row['PredInsertedRank']):.3f} obj={float(row['Objective']):.3f} "
            f"q={float(row['Pred_Qabf']):.3f} nabf={float(row['Pred_Nabf']):.4f} ssim={float(row['Pred_SSIM']):.3f}"
        )
    print(f"wrote {args.score_out}")


if __name__ == "__main__":
    main()
