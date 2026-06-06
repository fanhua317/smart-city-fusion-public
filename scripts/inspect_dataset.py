from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from fusion_common import ensure_dir, gradient_mag, gray_float, iter_image_files, load_image
from local_metrics import average_gradient


FIELDNAMES = [
    "image",
    "width",
    "height",
    "ext",
    "ir_mode",
    "vi_mode",
    "group",
    "ir_mean",
    "vi_mean",
    "ir_std",
    "vi_std",
    "ir_ag",
    "vi_ag",
    "ir_grad_gt_vi_ratio",
]


def tags_for(name: str, vi_mode: str) -> list[str]:
    stem = Path(name).stem
    tags: list[str] = []
    if name.startswith("FLIR_"):
        tags.append("FLIR")
    if stem.endswith("D"):
        tags.append("DN_D")
    if stem.endswith("N"):
        tags.append("DN_N")
    if vi_mode == "L":
        tags.append("GRAY_VI")
    return tags


def inspect_dataset(ir_dir: str | Path, vi_dir: str | Path, expected_count: int = 93) -> list[dict[str, str | float | int]]:
    ir_dir = Path(ir_dir)
    vi_dir = Path(vi_dir)
    ir_names = sorted(path.name for path in iter_image_files(ir_dir))
    vi_names = sorted(path.name for path in iter_image_files(vi_dir))
    problems: list[str] = []
    if ir_names != vi_names:
        problems.append(
            "filename mismatch: "
            f"missing_vi={sorted(set(ir_names) - set(vi_names))[:8]} "
            f"missing_ir={sorted(set(vi_names) - set(ir_names))[:8]}"
        )
    if len(ir_names) != expected_count:
        problems.append(f"expected {expected_count} pairs, got {len(ir_names)}")

    rows: list[dict[str, str | float | int]] = []
    for name in ir_names:
        ir_path = ir_dir / name
        vi_path = vi_dir / name
        if not vi_path.exists():
            continue
        try:
            ir_img = load_image(ir_path)
            vi_img = load_image(vi_path)
        except Exception as exc:  # noqa: BLE001 - report every decode failure together.
            problems.append(f"decode {name}: {exc}")
            continue
        if ir_img.size != vi_img.size:
            problems.append(f"size mismatch {name}: ir={ir_img.size} vi={vi_img.size}")
            continue

        ir = gray_float(ir_img)
        vi = gray_float(vi_img)
        g_ir = gradient_mag(ir)
        g_vi = gradient_mag(vi)
        tags = tags_for(name, vi_img.mode)
        rows.append(
            {
                "image": name,
                "width": ir_img.size[0],
                "height": ir_img.size[1],
                "ext": ir_path.suffix.lower(),
                "ir_mode": ir_img.mode,
                "vi_mode": vi_img.mode,
                "group": ";".join(tags),
                "ir_mean": float(np.mean(ir)),
                "vi_mean": float(np.mean(vi)),
                "ir_std": float(np.std(ir)),
                "vi_std": float(np.std(vi)),
                "ir_ag": average_gradient(ir),
                "vi_ag": average_gradient(vi),
                "ir_grad_gt_vi_ratio": float(np.mean(g_ir > g_vi)),
            }
        )

    if problems:
        raise RuntimeError("\n".join(problems))
    return rows


def write_profile(rows: list[dict[str, str | float | int]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--out", default="results/dataset_profile.csv")
    parser.add_argument("--expected-count", type=int, default=93)
    args = parser.parse_args()

    rows = inspect_dataset(args.ir_dir, args.vi_dir, args.expected_count)
    write_profile(rows, args.out)
    group_counts: dict[str, int] = {}
    for row in rows:
        for tag in str(row["group"]).split(";"):
            if tag:
                group_counts[tag] = group_counts.get(tag, 0) + 1
    print({"count": len(rows), "groups": group_counts, "out": args.out})


if __name__ == "__main__":
    main()
