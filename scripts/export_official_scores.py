from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = ROOT / "results" / "official_round2"
FINAL_MANIFEST = ROOT / "results" / "final" / "eval" / "selection_manifest.json"
INSERT_RANKS = OFFICIAL_DIR / "insert_ranks_all.csv"
OUT_DIR = ROOT / "results" / "score_tables"

METRICS = ["AG", "CC", "EN", "MI", "MSE", "Nabf", "PSNR", "Qabf", "SCD", "SD", "SF", "SSIM", "VIF"]


def load_insert_ranks() -> dict[str, dict[str, str]]:
    if not INSERT_RANKS.exists():
        return {}
    with INSERT_RANKS.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["candidate"]: row for row in csv.DictReader(handle)}


def load_final_candidate() -> str:
    if not FINAL_MANIFEST.exists():
        return ""
    raw = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8-sig"))
    return str(raw.get("chosen_candidate", ""))


def official_rows() -> list[dict[str, str | float]]:
    insert = load_insert_ranks()
    chosen = load_final_candidate()
    rows: list[dict[str, str | float]] = []
    for metrics_path in sorted(OFFICIAL_DIR.glob("*/official_metrics_summary.json")):
        candidate = metrics_path.parent.name
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row: dict[str, str | float] = {
            "candidate": candidate,
            "is_current_best": "yes" if candidate == chosen else "",
            "inserted_rank": float(insert.get(candidate, {}).get("InsertedRank", "nan")),
        }
        for metric in METRICS:
            row[metric] = float(metrics[metric])
        for metric in ["R_Ag", "R_Cc", "R_En", "R_Mi", "R_Mse", "R_Nabf", "R_Psnr", "R_Qabf", "R_Scd", "R_Sd", "R_Sf", "R_Ssim", "R_Vif"]:
            if candidate in insert and metric in insert[candidate]:
                row[metric] = insert[candidate][metric]
        rows.append(row)
    rows.sort(key=lambda row: float(row["inserted_rank"]))
    return rows


def write_csv(rows: list[dict[str, str | float]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rank_fields = ["R_Ag", "R_Cc", "R_En", "R_Mi", "R_Mse", "R_Nabf", "R_Psnr", "R_Qabf", "R_Scd", "R_Sd", "R_Sf", "R_Ssim", "R_Vif"]
    fieldnames = ["candidate", "is_current_best", "inserted_rank", *METRICS, *rank_fields]
    with (OUT_DIR / "current_official_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict[str, str | float]]) -> None:
    cols = ["candidate", "is_current_best", "inserted_rank", "AG", "CC", "EN", "MI", "MSE", "Nabf", "PSNR", "Qabf", "SCD", "SD", "SF", "SSIM", "VIF"]
    lines = [
        "# 当前官方分数表",
        "",
        "数据来源：`results/official_round2/*/official_metrics_summary.json` 和 `results/official_round2/insert_ranks_all.csv`。",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |")
    lines.append("")
    lines.append("说明：`inserted_rank` 是把候选插入当时排行榜后的 13 项平均名次，数值越低越好。")
    (OUT_DIR / "current_official_scores.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = official_rows()
    write_csv(rows)
    write_markdown(rows)
    print(f"wrote {OUT_DIR / 'current_official_scores.csv'}")
    print(f"wrote {OUT_DIR / 'current_official_scores.md'}")


if __name__ == "__main__":
    main()
