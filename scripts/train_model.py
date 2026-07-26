#!/usr/bin/env python3
"""
Train one model under the grokking protocol.

    python scripts/train_model.py --k 3 --seed 42 --out runs/k3_s42

One run is about 2 GPU-hours on an A100.  The transition begins near 20
minutes and is complete by 40; the remaining time is circuit cleanup, during
which OOD accuracy is already high and the parameter norm keeps falling.

Watch the printed |theta| column.  If it never drops by half, the run has
not grokked, whatever the training accuracy says -- that is the whole point
of the complexity-collapse diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graphs import GraphSample
from src.model import ModelConfig
from src.train import TrainConfig, detect_complexity_collapse, train


def load_cell(path: Path, k: int, split: str) -> list[GraphSample]:
    z = np.load(path, allow_pickle=True)
    out = []
    for i in range(len(z["n_nodes"])):
        n = int(z["n_nodes"][i])
        out.append(
            GraphSample(
                graph_id=str(z["graph_id"][i]),
                adjacency=z["adjacency"][i][:n, :n],
                n_nodes=n,
                topology="ER",
                requested_mean_degree=float(z["requested_mean_degree"][i]),
                realized_mean_degree=float(z["realized_mean_degree"][i]),
                edge_count=int(z["edge_count"][i]),
                generation_seed=int(z["generation_seed"][i]),
                split=split,
                k=k,
                true_count=int(z["true_count"][i]),
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1.0,
        help="lambda; the protocol needs 1.0, values <= 0.1 do not grok",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--pooling", default="sum", choices=["sum"])
    args = ap.parse_args()

    train_set = load_cell(args.data / f"train_k{args.k}_ER.npz", args.k, "train")
    ood_set = load_cell(
        args.data / f"final_ood_k{args.k}_ER.npz", args.k, "final_ood"
    )
    print(f"train {len(train_set)} graphs | final-OOD {len(ood_set)} graphs")

    mc = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
    )
    tc = TrainConfig(
        k=args.k,
        seed=args.seed,
        steps=args.steps,
        weight_decay=args.weight_decay,
        lr=args.lr,
    )

    from src.model import CountingTransformer

    print(f"parameters: {CountingTransformer(mc).parameter_count():,}")

    model, traj = train(train_set, ood_set, mc, tc, args.out)

    groks, step = detect_complexity_collapse(traj)
    verdict = {
        "groks": groks,
        "collapse_step": step,
        "final_ood_accuracy": traj[-1]["ood_accuracy"] if traj else None,
        "peak_parameter_norm": max(r["parameter_norm"] for r in traj),
        "final_parameter_norm": traj[-1]["parameter_norm"] if traj else None,
    }
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))

    print(f"\ngroks: {groks}")
    if groks:
        print(f"complexity collapse at step {step:,}")
    print(f"final OOD exact match: {verdict['final_ood_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
