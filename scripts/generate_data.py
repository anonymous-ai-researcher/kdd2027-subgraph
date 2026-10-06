#!/usr/bin/env python3
"""
Build the full dataset: generate graphs, verify labels three ways, and write
one .npz per (split, k, topology) cell.

    python scripts/generate_data.py --out data/ --k 3 4 5 6

Runtime is dominated by ground-truth counting at k=6 on N=100, which is why
the default is to generate all splits once and cache them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.counting import count_cycles_verified, expected_count
from src.graphs import SPLIT_INDEX, SPLIT_SIZES, GraphSample, deduplicate, generate_split, generate_graph, to_adjacency

# graphs per (split, k) cell, per the protocol in the paper
# Cell sizes follow the paper's data statistics (Appx. G.1, Table 7 caption):
# 400 graphs per cell, except at k in {3, 6}, where SCALE_VAL pools 600 and
# FINAL_OOD pools 1,400 (200 per N; the first 400 in size-cycled order form the
# main evaluation subset, 58/57/57/57/57/57/57 across N = 40..100).
N_GRAPHS = {
    "train": {3: 2000, 4: 2000, 5: 2000, 6: 2000},
    "id": {3: 400, 4: 400, 5: 400, 6: 400},
    "scale_val": {3: 600, 4: 400, 5: 400, 6: 600},
    "final_ood": {3: 1400, 4: 400, 5: 400, 6: 1400},
}
MAIN_EVAL = 400  # size of the main FINAL_OOD evaluation subset


def build_cell(split, k, topology, n_graphs, base_seed, verbose=True, seen=None):
    t0 = time.time()
    samples = list(generate_split(split, k, n_graphs, topology, base_seed))
    samples = deduplicate(samples, seen)   # `seen` is shared across cells

    for s in samples:
        s.true_count = count_cycles_verified(s.adjacency, k)

    if verbose:
        counts = np.array([s.true_count for s in samples])
        exp = np.mean(
            [expected_count(s.n_nodes, k, s.requested_mean_degree) for s in samples]
        )
        print(
            f"  {split:>10s} k={k} {topology:>4s}  "
            f"n={len(samples):>4d}  "
            f"count mean {counts.mean():8.1f} (expected {exp:8.1f})  "
            f"max {counts.max():>7d}  "
            f"{time.time() - t0:5.1f}s"
        )
    return samples


def save_cell(samples: list[GraphSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        adjacency=np.array(
            [
                np.pad(
                    s.adjacency,
                    ((0, 100 - s.n_nodes), (0, 100 - s.n_nodes)),
                )
                for s in samples
            ],
            dtype=np.uint8,
        ),
        n_nodes=np.array([s.n_nodes for s in samples], dtype=np.int16),
        true_count=np.array([s.true_count for s in samples], dtype=np.int64),
        graph_id=np.array([s.graph_id for s in samples]),
        requested_mean_degree=np.array(
            [s.requested_mean_degree for s in samples], dtype=np.float32
        ),
        realized_mean_degree=np.array(
            [s.realized_mean_degree for s in samples], dtype=np.float32
        ),
        edge_count=np.array([s.edge_count for s in samples], dtype=np.int32),
        generation_seed=np.array([s.generation_seed for s in samples], dtype=np.int32),
        main_eval_included=np.array(
            [i < MAIN_EVAL for i in range(len(samples))], dtype=bool
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--k", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "id", "scale_val", "final_ood"],
    )
    ap.add_argument(
        "--topologies",
        nargs="+",
        default=["ER"],
        help="ER for the main splits; add BA SBM WS for transfer",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = []
    # One WL-hash set per k, shared by every split in generation order
    # (train first), so deduplication is across splits as the paper states.
    seen_by_k = {k: set() for k in args.k}
    for k in args.k:
        print(f"\nk = {k}")
        for split in args.splits:
            for topo in args.topologies:
                if topo != "ER" and split != "final_ood":
                    continue  # transfer is evaluated only out of distribution
                # Salt by split index, not by the split's first size: train
                # and id share the size range 15..25 and previously collided.
                base = args.seed + 100_000 * k + 10_000 * SPLIT_INDEX[split]
                samples = build_cell(
                    split, k, topo, N_GRAPHS[split][k], base, seen=seen_by_k[k]
                )
                path = args.out / f"{split}_k{k}_{topo}.npz"
                save_cell(samples, path)
                manifest.append(
                    {
                        "path": str(path),
                        "split": split,
                        "k": k,
                        "topology": topo,
                        "n_graphs": len(samples),
                        "sizes": sorted({s.n_nodes for s in samples}),
                    }
                )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} cells to {args.out}/")


if __name__ == "__main__":
    main()
