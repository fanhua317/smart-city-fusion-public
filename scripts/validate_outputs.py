from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion_common import gray_float, list_pairs, load_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--fused-dir", default="results/final/fused")
    parser.add_argument("--out", default="results/final/eval/validation.json")
    args = parser.parse_args()

    names = list_pairs(args.ir_dir, args.vi_dir)
    problems: list[str] = []
    stats = []
    for name in names:
        f_path = Path(args.fused_dir) / name
        if not f_path.exists():
            problems.append(f"missing {name}")
            continue
        ir_img = load_image(Path(args.ir_dir) / name)
        fused = load_image(f_path)
        if fused.size != ir_img.size:
            problems.append(f"size {name}: {fused.size} != {ir_img.size}")
        arr = gray_float(fused)
        if not np.isfinite(arr).all():
            problems.append(f"nonfinite {name}")
        if float(arr.std()) < 1e-4:
            problems.append(f"flat {name}")
        stats.append({"image": name, "mean": float(arr.mean()), "std": float(arr.std()), "size": fused.size})
    report = {"count": len(stats), "expected": len(names), "problems": problems, "stats": stats}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(stats), "expected": len(names), "problem_count": len(problems)}, ensure_ascii=False))
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
