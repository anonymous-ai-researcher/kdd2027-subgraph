#!/usr/bin/env python3
"""
Validate the released code against the paper and the shipped data.

    python scripts/validate_release.py --registry EXP00_GRAPH_REGISTRY__GRAPH_METADATA.csv \
                                       --stats EXP07_DATASET_STATISTICS__DERIVED_DATASET_STATS.csv

Checks
  V1  mean-degree ranges equal Appx. G.1
  V2  generation seeds differ across splits; train and id are disjoint (bytes and WL hash)
  V3  deduplication is shared across splits (train first)
  V4  the training loop never reads a Final-OOD split; padding uses MAX_N
  V5  regenerated ID counts, stratified by mean degree, match the shipped data
  V6  regenerating the shipped (N, mean degree) pairs reproduces the shipped count statistics

Exit status 1 if any check fails.
"""
import argparse
import collections
import csv
import re
import statistics
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.graphs as G  # noqa: E402
from src.counting import count_cycles_verified  # noqa: E402

PAPER_RANGES = {3: (4, 7), 4: (3, 5), 5: (2, 4), 6: (2, 3)}


def wl(s):
    return nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(s.adjacency), iterations=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # V1
    check("V1 DEGREE_RANGE == Appx. G.1", G.DEGREE_RANGE == PAPER_RANGES, f"{G.DEGREE_RANGE}")

    # V2
    bases = {}
    for k in [3, 4, 5, 6]:
        for split in G.SPLIT_INDEX:
            bases[(k, split)] = args.seed + 100_000 * k + 10_000 * G.SPLIT_INDEX[split]
    distinct = len(set(bases.values())) == len(bases)
    tr = list(G.generate_split("train", 6, 400, "ER", bases[(6, "train")]))
    idd = list(G.generate_split("id", 6, 400, "ER", bases[(6, "id")]))
    same = sum(1 for a, b in zip(tr, idd) if a.n_nodes == b.n_nodes and (a.adjacency == b.adjacency).all())
    inter = len({wl(s) for s in tr} & {wl(s) for s in idd})
    check("V2 seeds distinct across splits; train/id disjoint", distinct and same == 0 and inter == 0,
          f"identical adjacency {same}/400, WL intersection {inter}")

    # V3
    seen = set()
    d_tr = G.deduplicate(tr, seen)
    d_id = G.deduplicate(idd, seen)
    tr_dup = list(G.generate_split("id", 6, 400, "ER", bases[(6, "train")]))  # same stream as train
    d_dup = G.deduplicate(tr_dup, seen)
    check("V3 shared-seen dedup removes cross-split duplicates", len(d_dup) == 0 and len(d_tr) > 0 and len(d_id) > 0,
          f"train kept {len(d_tr)}, id kept {len(d_id)}, re-fed train stream kept {len(d_dup)} (expect 0)")

    # V4
    src = (Path(__file__).resolve().parents[1] / "src" / "train.py").read_text()
    check("V4 train.py never touches Final-OOD; padding from MAX_N",
          not re.search(r"final_ood|ood_samples", src) and "max_n = MAX_N" in src, "")

    # V5 / V6 need the shipped files
    if args.registry and args.stats and args.registry.exists() and args.stats.exists():
        reg = [r for r in csv.DictReader(open(args.registry)) if r["split"] == "ID"]
        stats = {(r["split"], int(r["k"])): r for r in csv.DictReader(open(args.stats))}
        v5_ok, v6_ok, det5, det6 = True, True, [], []
        for k in [3, 4, 5, 6]:
            rows = [r for r in reg if int(r["k"]) == k]
            ship_by_d = collections.defaultdict(list)
            for r in rows:
                ship_by_d[round(float(r["requested_mean_degree"]))].append(int(r["true_count"]))
            # V5: uniform draw with the fixed ranges, stratified by degree
            gen = list(G.generate_split("id", k, 400, "ER", bases[(k, "id")]))
            gen_by_d = collections.defaultdict(list)
            for s in gen:
                gen_by_d[int(s.requested_mean_degree)].append(count_cycles_verified(s.adjacency, k))
            for d in sorted(ship_by_d):
                ms, mg = statistics.mean(ship_by_d[d]), statistics.mean(gen_by_d.get(d, [0]))
                rel = abs(mg - ms) / max(ms, 1.0)
                det5.append(f"k={k} d={d}: shipped {ms:.1f} vs code {mg:.1f}")
                if rel > 0.35:
                    v5_ok = False
            # V6: regenerate the shipped (N, d) pairs with the code's generator
            cs = []
            for i, r in enumerate(rows):
                n, d = int(r["N"]), float(r["requested_mean_degree"])
                g = G.generate_graph("ER", n, d, bases[(k, "id")] + i)
                cs.append(count_cycles_verified(G.to_adjacency(g, n), k))
            cs.sort()
            st = stats[("ID", k)]
            mean_rel = abs(statistics.mean(cs) - float(st["mean"])) / max(float(st["mean"]), 1.0)
            det6.append(f"k={k}: mean {statistics.mean(cs):.2f} vs {float(st['mean']):.2f}, median {statistics.median(cs):.0f} vs {float(st['median']):.0f}, "
                        f"p95 {cs[int(0.95 * len(cs))]} vs {float(st['p95']):.0f}, max {cs[-1]} vs {st['max']}")
            if mean_rel > 0.25:
                v6_ok = False
        check("V5 per-degree ID count means match shipped data (within 35%)", v5_ok, "; ".join(det5))
        check("V6 regenerating shipped (N, degree) pairs reproduces ID statistics (mean within 25%)", v6_ok, " | ".join(det6))

    print("\n=== RELEASE VALIDATION ===")
    for n, ok, d in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        if d:
            print(f"         {d}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
