#!/usr/bin/env python3
"""
Evaluate a trained checkpoint and run the mechanistic analysis.

    python scripts/evaluate.py --ckpt runs/k3_s42/checkpoint.pt --k 3

Produces, in order:

  1. exact-match accuracy on every split and topology
  2. per-N breakdown, which is where the flat-versus-decaying contrast lives
  3. linear-probe R^2 for A^2 and A^3
  4. single-head ablation for all eight heads

Step 4 is the one that matters most.  Three heads should be critical and
five should be inert; if ablating an unpredicted head hurt as much as
ablating a predicted one, the circuit story would be wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import ablate_head, probe_matrix_power
from src.model import CountingTransformer, ModelConfig
from src.train import exact_match

from train_model import load_cell  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-probes", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location=device)
    model = CountingTransformer(ModelConfig(**blob["model_config"])).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    results: dict = {"checkpoint": str(args.ckpt), "k": args.k}

    # ---- 1. accuracy by split -------------------------------------------
    print("exact-match accuracy")
    per_split = {}
    for split in ["id", "scale_val", "final_ood"]:
        path = args.data / f"{split}_k{args.k}_ER.npz"
        if not path.exists():
            continue
        samples = load_cell(path, args.k, split)
        max_len = 1 + max(s.n_nodes for s in samples) ** 2
        if split == "final_ood":
            # Table 1 reports the 400-graph main subset; k in {3, 6} also ship
            # a 200-per-N pool (1,400 graphs) used for the per-N figures.
            flags = np.load(path, allow_pickle=True)
            main = (flags["main_eval_included"].astype(bool)
                    if "main_eval_included" in flags.files
                    else np.ones(len(samples), dtype=bool))
            main_samples = [s for s, f in zip(samples, main) if f]
            acc = exact_match(model, main_samples, max_len, device)
            per_split[split] = acc
            print(f"  {split:>10s}  {acc:6.2f}%   (main subset, n={len(main_samples)})")
            if len(main_samples) < len(samples):
                acc_pool = exact_match(model, samples, max_len, device)
                per_split["final_ood_per_n_pool"] = acc_pool
                print(f"  {'per-N pool':>10s}  {acc_pool:6.2f}%   (n={len(samples)})")
            continue
        acc = exact_match(model, samples, max_len, device)
        per_split[split] = acc
        print(f"  {split:>10s}  {acc:6.2f}%")
    results["accuracy_by_split"] = per_split

    # ---- 2. topology transfer -------------------------------------------
    per_topo = {}
    for topo in ["BA", "SBM", "WS"]:
        path = args.data / f"final_ood_k{args.k}_{topo}.npz"
        if not path.exists():
            continue
        samples = load_cell(path, args.k, "final_ood")
        max_len = 1 + max(s.n_nodes for s in samples) ** 2
        per_topo[topo] = exact_match(model, samples, max_len, device)
    if per_topo:
        print("topology transfer (zero-shot)")
        for t, a in per_topo.items():
            print(f"  {t:>10s}  {a:6.2f}%")
        results["topology_transfer"] = per_topo

    # ---- 3. per-N -------------------------------------------------------
    path = args.data / f"final_ood_k{args.k}_ER.npz"
    if path.exists():
        samples = load_cell(path, args.k, "final_ood")
        max_len = 1 + max(s.n_nodes for s in samples) ** 2
        by_n = {}
        for n in sorted({s.n_nodes for s in samples}):
            subset = [s for s in samples if s.n_nodes == n]
            by_n[n] = exact_match(model, subset, max_len, device)
        print("per-N accuracy")
        for n, a in by_n.items():
            print(f"  N={n:>4d}  {a:6.2f}%")
        results["accuracy_by_n"] = by_n

    # ---- 4. probes ------------------------------------------------------
    if not args.skip_probes:
        subset = samples[:64]
        max_len = 1 + max(s.n_nodes for s in subset) ** 2
        probes = {
            f"A{p}": probe_matrix_power(model, subset, max_len, device, power=p)
            for p in (2, 3)
        }
        print("linear-probe R^2")
        for name, r2 in probes.items():
            print(f"  {name:>10s}  {r2:.3f}")
        results["probe_r2"] = probes

    # ---- 5. head ablation ----------------------------------------------
    subset = samples[:200]
    max_len = 1 + max(s.n_nodes for s in subset) ** 2
    base = exact_match(model, subset, max_len, device)
    print(f"head ablation (baseline {base:.2f}%)")
    ablations = {}
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            acc = ablate_head(model, subset, max_len, device, layer, head)
            tag = f"L{layer}-H{head}"
            ablations[tag] = acc
            flag = "  <-- critical" if base - acc > 20 else ""
            print(f"  {tag}  {acc:6.2f}%{flag}")
    results["head_ablation"] = {"baseline": base, "ablated": ablations}

    out = args.out or args.ckpt.parent / "evaluation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
