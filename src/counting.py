"""
Ground-truth k-cycle counting, verified three independent ways.

A single off-by-one in the symmetry factor would silently rescale every
number in the paper, so the label pipeline never trusts one implementation.
Where two methods apply, they must agree exactly; where three apply, all
three must agree.

    method              applies to      independence
    ------------------  --------------  ----------------------------------
    trace identity      k = 3 only      pure linear algebra, no traversal
    DFS enumeration     all k           explicit path search
    networkx            all k           third-party, different algorithm

Counts are *non-induced*: a 6-cycle with a chord still contains six 4-cycles
and three 6-cycles.  Induced counting is a strictly harder problem and is
reported separately in the paper as a negative result.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def count_triangles_trace(adjacency: np.ndarray) -> int:
    """
    tr(A^3) / 6.

    Each triangle is traversed twice (two directions) from each of its three
    vertices, giving the factor of six.  Exact in int64 for the sizes used
    here: the largest count observed is 1.7e7, far below 2^63.
    """
    a = adjacency.astype(np.int64)
    return int(np.trace(a @ a @ a) // 6)


def count_cycles_dfs(adjacency: np.ndarray, k: int) -> int:
    """
    Enumerate simple k-cycles by depth-first search.

    Each cycle is found 2k times (k starting vertices, two directions), so
    the raw path count is divided by 2k.  To avoid double counting we only
    start from the smallest vertex in the cycle.
    """
    n = adjacency.shape[0]
    neighbours = [np.flatnonzero(adjacency[i]).tolist() for i in range(n)]
    total = 0

    def walk(start: int, current: int, depth: int, visited: set[int]) -> None:
        nonlocal total
        if depth == k:
            if start in neighbours[current]:
                total += 1
            return
        for nxt in neighbours[current]:
            if nxt <= start or nxt in visited:
                continue
            visited.add(nxt)
            walk(start, nxt, depth + 1, visited)
            visited.discard(nxt)

    for s in range(n):
        walk(s, s, 1, {s})

    # Each cycle is traversed twice (clockwise and anticlockwise) from its
    # smallest vertex; starting-vertex multiplicity is already removed by
    # the `nxt <= start` guard.
    return total // 2


def count_cycles_networkx(adjacency: np.ndarray, k: int) -> int:
    """Filter networkx.simple_cycles by length."""
    g = nx.from_numpy_array(adjacency)
    return sum(1 for c in nx.simple_cycles(g, length_bound=k) if len(c) == k)


def count_cycles_verified(adjacency: np.ndarray, k: int) -> int:
    """
    Return the k-cycle count, raising if the applicable methods disagree.

    This is the only function the data pipeline should call.
    """
    dfs = count_cycles_dfs(adjacency, k)
    nxc = count_cycles_networkx(adjacency, k)

    if dfs != nxc:
        raise AssertionError(
            f"DFS ({dfs}) and networkx ({nxc}) disagree at k={k}"
        )

    if k == 3:
        trace = count_triangles_trace(adjacency)
        if trace != dfs:
            raise AssertionError(
                f"trace identity ({trace}) disagrees with DFS ({dfs})"
            )

    return dfs


def expected_count(n: int, k: int, mean_degree: float) -> float:
    """
    Expected k-cycle count in G(n, p) with p = d_bar / (n - 1):

        C(n, k) * (k-1)!/2 * p^k  ->  d_bar^k / (2k)   as n grows.

    Used only as a sanity check on the generated data; the labels themselves
    always come from `count_cycles_verified`.
    """
    from math import comb, factorial

    p = mean_degree / (n - 1)
    return comb(n, k) * factorial(k - 1) / 2 * p**k


def count_pattern(adjacency: np.ndarray, pattern: nx.Graph) -> int:
    """
    Non-induced count of an arbitrary connected pattern, by subgraph
    isomorphism.  Used for the non-cycle patterns (paw, bull, tadpole,
    C4-plus-pendant) and, in the negative-results section, for K4.
    """
    g = nx.from_numpy_array(adjacency)
    matcher = nx.algorithms.isomorphism.GraphMatcher(g, pattern)
    automorphisms = sum(
        1 for _ in nx.algorithms.isomorphism.GraphMatcher(
            pattern, pattern
        ).isomorphisms_iter()
    )
    return sum(1 for _ in matcher.subgraph_monomorphisms_iter()) // automorphisms
