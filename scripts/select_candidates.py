from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from fusion_common import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="results/candidate_scores.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-dir", default="results/top_candidates")
    parser.add_argument("--final-dir", default="results/final/fused")
    args = parser.parse_args()

    with Path(args.scores).open("r", encoding="utf-8-sig", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["PseudoRank"]))
    if not rows:
        raise RuntimeError("No score rows")

    out_dir = ensure_dir(args.out_dir)
    top = rows[: args.top_k]
    for rank, row in enumerate(top, start=1):
        src = Path(row["path"])
        dst = out_dir / f"rank{rank:02d}_{src.name}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    best_src = Path(top[0]["path"])
    final_dir = Path(args.final_dir)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(best_src, final_dir)
    manifest = {
        "best": top[0],
        "top": top,
        "final_dir": str(final_dir),
    }
    ensure_dir("results/final/eval")
    Path("results/final/eval/selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"best={top[0]['candidate']} pseudo_rank={top[0]['PseudoRank']} final={final_dir}")


if __name__ == "__main__":
    main()
