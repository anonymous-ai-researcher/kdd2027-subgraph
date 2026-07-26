"""
Graph generation for exact k-cycle counting.

The generation protocol fixes the *mean degree* rather than the edge
probability.  This is the design decision that makes exact-match accuracy a
meaningful metric at every graph size: with p_edge = d_bar / (N - 1), the
expected k-cycle count is

    C(N, k) * (k-1)!/2 * p^k  ~=  d_bar^k / (2k),

which is independent of N.  Had p_edge been fixed instead, counts would grow
as Theta(N^k) and exceed 10^7 at N = 100, making exact match hopeless on
large graphs for reasons that have nothing to do with the model.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Iterator, Literal

import networkx as nx
import numpy as np

Topology = Literal["ER", "BA", "SBM", "WS"]

# Per-k mean-degree ranges.  Larger patterns need denser graphs to give a
# non-degenerate count distribution; smaller patterns saturate if the graph
# is too dense.
DEGREE_RANGE: dict[int, tuple[int, int]] = {
    3: (4, 8),
    4: (3, 7),
    5: (3, 6),
    6: (3, 6),
}

# Evaluation splits.  ID is a contiguous range; the two out-of-distribution
# splits sample a fixed discrete ladder of sizes.
SPLIT_SIZES: dict[str, list[int]] = {
    "train": list(range(15, 26)),
    "id": list(range(15, 26)),
    "scale_val": [26, 30, 35],
    "final_ood": [40, 50, 60, 70, 80, 90, 100],
}


@dataclass
class GraphSample:
    """A single generated graph together with its provenance."""

    graph_id: str
    adjacency: np.ndarray  # (N, N) uint8, symmetric, zero diagonal
    n_nodes: int
    topology: Topology
    requested_mean_degree: float
    realized_mean_degree: float
    edge_count: int
    generation_seed: int
    split: str
    k: int
    true_count: int = -1  # filled in by counting.py

    @property
    def graph_pool_hash(self) -> str:
        """SHA-256 of the exact adjacency, for provenance tracking."""
        return hashlib.sha256(self.adjacency.tobytes()).hexdigest()[:16]


def _target_edges(n: int, mean_degree: float) -> int:
    return int(math.floor(mean_degree * n / 2))


def _match_edge_count(g: nx.Graph, target: int, rng: random.Random) -> nx.Graph:
    """Add or remove edges uniformly at random until |E| == target."""
    g = g.copy()

    while g.number_of_edges() > target:
        u, v = rng.choice(list(g.edges()))
        g.remove_edge(u, v)

    nodes = list(g.nodes())
    guard = 0
    while g.number_of_edges() < target and guard < 100 * target:
        u, v = rng.sample(nodes, 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v)
        guard += 1

    return g


def generate_graph(
    topology: Topology,
    n: int,
    mean_degree: float,
    seed: int,
) -> nx.Graph:
    """
    Generate one graph with the requested topology and an edge count that
    matches floor(mean_degree * n / 2) exactly.

    Every family is post-processed to hit the same edge count, so a topology
    comparison isolates *structure* rather than density.  Without this step a
    Barabasi-Albert graph and an Erdos-Renyi graph at the same nominal degree
    differ in edge count, and any accuracy gap between them is confounded.
    """
    rng = random.Random(seed)
    target = _target_edges(n, mean_degree)

    if topology == "ER":
        p = mean_degree / (n - 1)
        g = nx.gnp_random_graph(n, p, seed=seed)

    elif topology == "BA":
        # m edges per new node; m = round(d_bar / 2) reproduces the degree.
        m = max(1, int(round(mean_degree / 2)))
        g = nx.barabasi_albert_graph(n, m, seed=seed)

    elif topology == "SBM":
        # Two equal communities, 4:1 intra/inter ratio.
        half = n // 2
        sizes = [half, n - half]
        p_in = min(1.0, 1.6 * mean_degree / (n - 1))
        p_out = p_in / 4.0
        g = nx.stochastic_block_model(
            sizes, [[p_in, p_out], [p_out, p_in]], seed=seed
        )

    elif topology == "WS":
        # The ring construction admits only even degrees, so we round up to
        # the next even number and then remove edges down to the target.
        # This second step is what realizes the odd mean degrees 3, 5 and 7.
        k_ring = 2 * math.ceil(mean_degree / 2)
        g = nx.watts_strogatz_graph(n, k_ring, 0.3, seed=seed)

    else:  # pragma: no cover
        raise ValueError(f"unknown topology: {topology}")

    g = _match_edge_count(g, target, rng)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def to_adjacency(g: nx.Graph, n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.uint8)
    for u, v in g.edges():
        a[u, v] = 1
        a[v, u] = 1
    return a


def generate_split(
    split: str,
    k: int,
    n_graphs: int,
    topology: Topology = "ER",
    base_seed: int = 0,
) -> Iterator[GraphSample]:
    """
    Yield `n_graphs` samples for one (split, k) cell.

    Sizes are cycled deterministically through SPLIT_SIZES[split] so that
    every listed size receives the same number of graphs.
    """
    lo, hi = DEGREE_RANGE[k]
    sizes = SPLIT_SIZES[split]

    for i in range(n_graphs):
        seed = base_seed + i
        rng = random.Random(seed)
        n = sizes[i % len(sizes)]
        d_bar = float(rng.randint(lo, hi))

        g = generate_graph(topology, n, d_bar, seed)
        a = to_adjacency(g, n)
        m = int(a.sum() // 2)

        yield GraphSample(
            graph_id=f"{split}_{topology}_k{k}_N{n}_s{seed}",
            adjacency=a,
            n_nodes=n,
            topology=topology,
            requested_mean_degree=d_bar,
            realized_mean_degree=2.0 * m / n,
            edge_count=m,
            generation_seed=seed,
            split=split,
            k=k,
        )


def deduplicate(samples: list[GraphSample]) -> list[GraphSample]:
    """
    Remove isomorphic duplicates using the Weisfeiler-Lehman graph hash.

    Duplicates across the train/test boundary would let memorization look
    like generalization, so this runs before any split is written to disk.
    """
    seen: set[str] = set()
    out: list[GraphSample] = []

    for s in samples:
        g = nx.from_numpy_array(s.adjacency)
        h = nx.weisfeiler_lehman_graph_hash(g, iterations=3)
        if h not in seen:
            seen.add(h)
            out.append(s)

    return out
