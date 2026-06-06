# Smart City Infrared-Visible Image Fusion

Coursework project for infrared-visible image fusion ranking experiments.

This public version is desensitized:

- No test images are included.
- No official `Evaluation.exe` binary is included.
- No encrypted `.dt` submission files are included.
- No raw leaderboard CSV, student IDs, usernames, passwords, or private LAN host details are included.
- Remote GPU helper scripts use placeholder host/user values.

## Contents

- `scripts/`: fusion generation, local metrics, candidate selection, and remote helper scripts.
- `docs/`: experiment notes and upload strategy records.
- `results/score_tables/public_scores.*`: anonymized own-candidate score summaries.
- `results/score_tables/anonymized_leaderboard_*.csv`: full anonymized leaderboard metrics for rank-aware tuning.

## Reproduce Locally

Prepare paired image folders:

```powershell
py -3 scripts\fusion_generate.py --ir-dir target\ir --vi-dir target\vi --out-dir results\demo_fused --preset balanced
py -3 scripts\local_metrics.py --ir-dir target\ir --vi-dir target\vi --fused-dir results\demo_fused --out results\demo_scores.csv
```

Official evaluation must be run locally with the course-provided tool and dataset, which are intentionally not included here.
