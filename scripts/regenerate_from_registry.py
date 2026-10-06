#!/usr/bin/env python3
"""
Regenerate the shipped evaluation graphs from the registry, exactly.

    python scripts/regenerate_from_registry.py --registry EXP00_GRAPH_REGISTRY__GRAPH_METADATA.csv --out data/ [--topology ER]

For every registry row the generator is run with the row's (N, mean degree,
generation_seed); the regenerated graph must match the registry's edge count
and true count, or the script fails.  ER graphs regenerate exactly with
nx.gnm_random_graph(N, round(mean_degree * N / 2), seed).  The transfer
topologies (BA, SBM, WS) carry an exact edge-count matching step whose random
stream is not part of this release; they are shipped as graph files.
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.graphs as G  # noqa: E402
from src.counting import count_cycles_verified  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--topology", default="ER")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.registry)) if r["topology"] == args.topology]
    if args.limit:
        rows = rows[: args.limit]
    cells = collections.defaultdict(list)
    bad = []
    for r in rows:
        n, d, s, k = int(r["N"]), float(r["requested_mean_degree"]), int(r["generation_seed"]), int(r["k"])
        g = G.generate_graph(args.topology, n, d, s)
        a = G.to_adjacency(g, n)
        c = count_cycles_verified(a, k)
        if g.number_of_edges() != int(r["edge_count"]) or c != int(r["true_count"]):
            bad.append(r["graph_id"])
        cells[(r["split"].lower(), k)].append((r["graph_id"], a, n, c, d, s, r.get("main_eval_included", "True") in ("True", "1", "true")))
    args.out.mkdir(parents=True, exist_ok=True)
    for (split, k), items in cells.items():
        np.savez_compressed(args.out / f"{split}_k{k}_{args.topology}.npz",
                            graph_id=np.array([i[0] for i in items]),
                            adjacency=np.array([np.pad(i[1], ((0, 100 - i[2]), (0, 100 - i[2]))) for i in items], dtype=np.int8),
                            n_nodes=np.array([i[2] for i in items]), true_count=np.array([i[3] for i in items]),
                            requested_mean_degree=np.array([i[4] for i in items]), generation_seed=np.array([i[5] for i in items]),
                            main_eval_included=np.array([i[6] for i in items], dtype=bool))
    print(f"regenerated {len(rows)} {args.topology} graphs into {len(cells)} cells; mismatches: {len(bad)}")
    if bad:
        print("  first mismatches:", bad[:5])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
