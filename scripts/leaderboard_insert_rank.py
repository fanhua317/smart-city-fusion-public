from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ["Ag", "Cc", "En", "Mi", "Mse", "Nabf", "Psnr", "Qabf", "Scd", "Sd", "Sf", "Ssim", "Vif"]
LOWER_IS_BETTER = {"Mse", "Nabf"}
JSON_ALIASES = {
    "Ag": "AG",
    "Cc": "CC",
    "En": "EN",
    "Mi": "MI",
    "Mse": "MSE",
    "Nabf": "Nabf",
    "Psnr": "PSNR",
    "Qabf": "Qabf",
    "Scd": "SCD",
    "Sd": "SD",
    "Sf": "SF",
    "Ssim": "SSIM",
    "Vif": "VIF",
}


def load_leaderboard(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No leaderboard rows in {path}")
    missing = [m for m in METRICS if m not in rows[0]]
    if missing:
        raise RuntimeError(f"Missing leaderboard metric columns: {missing}")
    return rows


def load_metrics(path: str | Path) -> dict[str, float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    values: dict[str, float] = {}
    for metric, alias in JSON_ALIASES.items():
        if metric in raw:
            values[metric] = float(raw[metric])
        elif alias in raw:
            values[metric] = float(raw[alias])
        else:
            raise RuntimeError(f"{path} is missing {metric}/{alias}")
    return values


def metric_rank(leader_rows: list[dict[str, str]], metric: str, value: float) -> int:
    if metric in LOWER_IS_BETTER:
        return 1 + sum(float(row[metric]) < value for row in leader_rows)
    return 1 + sum(float(row[metric]) > value for row in leader_rows)


def summarize_candidate(leader_rows: list[dict[str, str]], name: str, metrics: dict[str, float]) -> dict[str, float | str]:
    first = leader_rows[0]
    out: dict[str, float | str] = {"candidate": name}
    ranks: list[int] = []
    for metric in METRICS:
        value = float(metrics[metric])
        rank = metric_rank(leader_rows, metric, value)
        ranks.append(rank)
        out[metric] = value
        out[f"R_{metric}"] = rank
        out[f"D_first_{metric}"] = value - float(first[metric])
    out["InsertedRank"] = sum(ranks) / len(ranks)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", default="leaderboard_20260605_234547.csv")
    parser.add_argument("--metrics-json", action="append", default=[])
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--out", default="results/leaderboard_insert_ranks.csv")
    args = parser.parse_args()

    if not args.metrics_json:
        raise SystemExit("Pass at least one --metrics-json file.")
    if args.name and len(args.name) != len(args.metrics_json):
        raise SystemExit("--name count must match --metrics-json count.")

    leader_rows = load_leaderboard(args.leaderboard)
    rows: list[dict[str, float | str]] = []
    for idx, metrics_path in enumerate(args.metrics_json):
        path = Path(metrics_path)
        name = args.name[idx] if args.name else path.parent.name
        rows.append(summarize_candidate(leader_rows, name, load_metrics(path)))

    rows.sort(key=lambda row: float(row["InsertedRank"]))
    fieldnames = (
        ["candidate", "InsertedRank"]
        + METRICS
        + [f"R_{metric}" for metric in METRICS]
        + [f"D_first_{metric}" for metric in METRICS]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"{row['InsertedRank']:.3f} {row['candidate']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
