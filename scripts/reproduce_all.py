#!/usr/bin/env python3
"""
Reproduce the main table end to end.

    python scripts/reproduce_all.py --seeds 42 123 456 789 1024

This is 20 runs (4 cycle lengths x 5 seeds) at roughly 2 GPU-hours each, so
budget about 40 GPU-hours for the main results alone.  Use --dry-run first
to check the plan without spending anything.

If you only want to see the phenomenon rather than the full table, one run
at k=3 is enough:

    python scripts/train_model.py --k 3 --seed 42 --out runs/demo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], dry: bool) -> None:
    print("  $", " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    n_runs = len(args.k) * len(args.seeds)
    print(f"plan: {n_runs} training runs, ~{2 * n_runs} GPU-hours\n")

    if not args.skip_data:
        print("[1/3] generating data")
        run(
            [
                sys.executable, "scripts/generate_data.py",
                "--out", str(args.data),
                "--k", *map(str, args.k),
                "--topologies", "ER", "BA", "SBM", "WS",
            ],
            args.dry_run,
        )

    print("\n[2/3] training")
    t0 = time.time()
    for k in args.k:
        for seed in args.seeds:
            out = args.runs / f"k{k}_s{seed}"
            run(
                [
                    sys.executable, "scripts/train_model.py",
                    "--k", str(k), "--seed", str(seed),
                    "--steps", str(args.steps),
                    "--data", str(args.data), "--out", str(out),
                ],
                args.dry_run,
            )

    print("\n[3/3] evaluating")
    summary = []
    for k in args.k:
        for seed in args.seeds:
            out = args.runs / f"k{k}_s{seed}"
            run(
                [
                    sys.executable, "scripts/evaluate.py",
                    "--ckpt", str(out / "checkpoint.pt"),
                    "--k", str(k), "--data", str(args.data),
                ],
                args.dry_run,
            )
            f = out / "evaluation.json"
            if f.exists():
                summary.append({"k": k, "seed": seed, **json.loads(f.read_text())})

    if summary and not args.dry_run:
        agg: dict[int, list[float]] = {}
        for row in summary:
            agg.setdefault(row["k"], []).append(
                row["accuracy_by_split"]["final_ood"]
            )
        print("\nfinal-OOD exact match (mean over seeds)")
        for k in sorted(agg):
            v = agg[k]
            mean = sum(v) / len(v)
            sd = (sum((x - mean) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
            print(f"  k={k}  {mean:5.1f} +- {sd:.1f}   (n={len(v)})")
        (args.runs / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nelapsed {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
