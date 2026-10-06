<div align="center">

# Grokking Graph Structure

### When Transformers Learn Exact Subgraph Counting

**A 100K-parameter Transformer, given nothing but a flattened adjacency matrix,
learns an exact algorithm for counting $k$-cycles — and we can find that
algorithm in its weights.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

## The result in one table

Exact-match accuracy on graphs **4× larger than anything seen in training**
(mean ± std over 5 seeds):

| Model | $C_3$ | $C_4$ | $C_5$ | $C_6$ |
|:---|---:|---:|---:|---:|
| **Ours** (grokked, 100K params) | **98.9** ± 0.4 | **98.8** ± 0.4 | **98.5** ± 0.3 | **98.1** ± 0.9 |
| I²-GNN (prior best, 200K params) | 73.0 ± 6.2 | 65.4 ± 4.2 | 55.0 ± 4.9 | 40.8 ± 5.5 |
| GIN | 31.0 ± 5.8 | 24.6 ± 3.6 | 16.5 ± 2.7 | 10.0 ± 2.0 |
| LSTM (40× larger) | 16.2 ± 4.1 | 14.4 ± 2.8 | 12.8 ± 2.1 | 11.7 ± 2.7 |
| Constant predictor (floor) | 4.5 | 6.0 | 8.8 | 12.8 |

*Exact match* means the rounded prediction equals the true count. An error of
one still counts as wrong.

**Zero-shot topology transfer** — trained only on Erdős–Rényi graphs:

| ER | Barabási–Albert | Stochastic block model | Watts–Strogatz |
|---:|---:|---:|---:|
| 99.4 | 99.2 | 99.5 | 99.3 |

---

## Why this is interesting

Message-passing GNNs **provably cannot** count triangles — the 1-Weisfeiler-Lehman
barrier stops them. Recurrent models fail for a different reason: a
fixed-dimensional state cannot stream past the all-pairs comparisons that
counting demands. Our recurrent baselines never exceed 20%.

So we removed every structural prior and handed a plain Transformer the
flattened adjacency matrix: $N^2$ binary tokens, zeros and ones, nothing
else. Standard training squanders this on memorization.

**Grokking does not.** Under strong weight decay and training two orders of
magnitude past the point where the training loss converges, the model
undergoes a phase transition into an exact counting algorithm.

And the algorithm is not a black box. We derive the circuit *before*
training — from the requirement that the model compute $\mathrm{tr}(A^3)/6$
with matrix products — and then locate it in the trained weights.

---

## Three ingredients, none optional

<table>
<tr><th width="22%">Ingredient</th><th width="43%">Why</th><th width="35%">What happens without it</th></tr>

<tr><td><b>Axial node codes</b></td>
<td>Serialized row-major, entries <code>A[u,v]</code> and <code>A[v,w]</code> share endpoint <code>v</code>, but their offset depends on <code>u</code> and <code>w</code> individually. A 1-D positional encoding sees only offsets, so it cannot bridge them. Axial codes address by <i>identity</i>: position <code>(u,v)</code> carries <code>[φ(u); φ(v)]</code> with fixed random sign vectors.</td>
<td><b>5.5%</b> with no positional encoding.<br><b>36.2%</b> with absolute positions.<br>The strongest 1-D scheme we tested (RoPE) reaches <b>59.9%</b> — still 39 points short.</td></tr>

<tr><td><b>Size token</b></td>
<td>Softmax normalizes by the attended set, so a head returns a <i>mean</i>, not a sum. Recovering degree from that mean needs <code>N</code>. One prepended token carrying <code>1/N</code> supplies it.</td>
<td>Removing the token but leaving sequence length visible costs <b>7 pp</b>. Hiding every size cue costs <b>35 pp</b>.</td></tr>

<tr><td><b>Grokking protocol</b></td>
<td>Both the memorizing and the generalizing circuit fit the training data, but the generalizing one reaches the same loss at smaller parameter norm. Weight decay selects it — <i>eventually</i>.</td>
<td>At λ ≤ 0.1 the model stays memorized (<b>13–28%</b>). At λ = 0 it reaches <b>3.5%</b>, below the constant-predictor floor.</td></tr>
</table>

---

## The circuit, found in the weights

The theory predicts a three-head circuit at depth 2. Ablating each of the
eight heads, one at a time:

```
                     zero-ablated accuracy
  L0-H0   degree recovery        4.5%   ←  critical
  L0-H1                         99.4%
  L0-H2   column gather         12.4%   ←  critical
  L0-H3                         98.9%
  L1-H0                         99.3%
  L1-H1   contraction           18.7%   ←  critical
  L1-H2                         99.2%
  L1-H3                         98.8%
```

Three heads are critical, five are inert. That is what the construction
predicts. Linear probes recover the intermediate matrix powers from the
hidden states (mean $R^2 > 0.92$ for both $A^2$ and $A^3$).

---

## Quickstart

### Install

```bash
git clone <this-repo> && cd <this-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### See the phenomenon (one run, ~2 GPU-hours)

```bash
python scripts/generate_data.py --out data/ --k 3
python scripts/train_model.py --k 3 --seed 42 --out runs/demo
```

Watch the `|theta|` column. This is the whole story:

```
step    1000  loss 2.14e-03  train  99.85  scale_val  19.20  |theta|   31.4
step   10000  loss 1.02e-05  train 100.00  scale_val  20.10  |theta|   37.4   ← peak
step   15000  loss 8.71e-06  train 100.00  scale_val  52.20  |theta|   35.9   ← collapse begins
step   30000  loss 4.33e-06  train 100.00  scale_val  99.40  |theta|    9.9   ← done
step  100000  loss 2.18e-06  train 100.00  scale_val  99.20  |theta|    9.7
```

The training loss is essentially zero by step 1,000. Everything after that —
the part conventional training skips — is where the algorithm forms.

### Find the circuit

```bash
python scripts/evaluate.py --ckpt runs/demo/checkpoint.pt --k 3
```

### Reproduce the main table (20 runs, ~40 GPU-hours)

```bash
python scripts/reproduce_all.py --dry-run   # check the plan first
python scripts/reproduce_all.py
```

---

## Repository layout

```
├── src/
│   ├── graphs.py        graph generation, four topologies, exact edge matching
│   ├── counting.py      ground truth, verified three independent ways
│   ├── model.py         the Transformer: axial codes, size token, sum pooling
│   ├── train.py         grokking protocol + complexity-collapse detection
│   └── analysis.py      probes, head ablation, attention statistics
├── scripts/
│   ├── generate_data.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── reproduce_all.py
├── configs/default.yaml
├── results/             measured results behind every table in the paper
└── docs/                extended notes
```

---

## Design decisions worth knowing about

**Mean degree is fixed, not edge probability.** With $p = \bar{d}/(N-1)$, the
expected count is $\bar{d}^k / 2k$ — independent of $N$. Had we fixed $p$
instead, counts would grow as $\Theta(N^k)$ and exceed $10^7$ at $N=100$,
making exact match meaningless on large graphs for reasons having nothing to
do with the model.

**Counts are non-induced.** A 6-cycle with a chord still contains six
4-cycles. Induced counting is strictly harder and is reported separately as a
negative result.

**Ground truth is verified three ways.** The trace identity ($k=3$ only), DFS
enumeration, and `networkx.simple_cycles`. All applicable methods must agree
on all 9,600 evaluation graphs. A single off-by-one in a symmetry factor
would silently rescale every number.

**Graphs are deduplicated by Weisfeiler-Lehman hash across splits** (one hash
set per k, training split first), so a train/test duplicate cannot make
memorization look like generalization.

**The Final-OOD split is never used for any design or selection decision.**
Hyperparameters and checkpoints were selected on the in-distribution and
scale-validation splits only; the training loop logs the scale-validation
split, and Final-OOD is evaluated once, by `scripts/evaluate.py`, after
training. (The paper's training-dynamics figures were produced by logging
Final-OOD during the original runs; that logging never fed a decision.)

---

## Two results that run against us

We report these because the alternative is publishing a claim we know to be
incomplete.

**4-cliques are impossible, at any depth.** No summed matrix-power readout can
count $K_4$ — witnessed on the four Chang graphs, which share every matrix
power with identical coefficients but have $K_4$ counts of 280, 248, 240 and
240. Any such model must give them all the same answer.

An earlier 93.5% figure on $K_4$ was not evidence of counting: the
zero-predictor floor for that distribution is 68.8%.

**Induced $C_6$ reaches only 71.3%** — exactly where the homomorphism basis
says a summable circuit must stop.

The same construction that kills $K_4$ *confirms* cycle counting: all four
Chang graphs have identical cycle counts (336 triangles, 1890 4-cycles,
14448 5-cycles, 122080 6-cycles), so the answer the model is forced to share
is the correct one.

Both results delimit the claim rather than qualify it: **the method counts
what the circuit class can count, and no more.**

---

## Cost

| | Training | Inference |
|:---|---:|---:|
| DFS enumeration | none | 4.7 ms (CPU) |
| GIN | minutes | ~1 ms |
| I²-GNN | ~6 GPU-h | ~40 ms |
| **Ours** | **~2 GPU-h** | **~32 ms** |

This is **slower than exact enumeration** and offers no capability a
practitioner lacks from a combinatorial library. The contribution is
scientific, not operational: it is a demonstration that gradient descent can
discover an exact algorithm, and a method for verifying that it did.

Reproducing the full experiment suite is approximately 4,400 GPU-hours. The
gap between that and the 2 hours of a single run is the protocol: grokking
requires training far past convergence, so the cost is the object of study
rather than incidental overhead. If you want the trained behaviour, start
from a checkpoint; if you want the transition, budget accordingly.

---

## Data and ethics

Every graph is generated programmatically from classical random-graph
models. No graph derives from a social network, citation record, molecular
database, or any collection built from or about people or institutions.
Nodes carry no attributes: a graph is a binary adjacency matrix and nothing
else. There is no personal, private, proprietary or identifiable data
anywhere in the pipeline, and no human subjects at any stage.

The one component we would flag for downstream care is the
**complexity-collapse** diagnostic. It is a claim about when a model has
internalized a rule rather than a lookup table, and we have validated it only
on the counting tasks studied here. We would caution against treating it as a
general certificate of algorithmic generalization without further evidence.

---

## License

MIT


---

## Release notes: what changed and how to check it

This release corrects four defects in the previously published code. None of
the paper's reported numbers depend on the previous code; every number comes
from the shipped data, which the corrected code regenerates exactly.

| Defect | Where | Fix |
|:---|:---|:---|
| Mean-degree ranges were `[4,8]/[3,7]/[3,6]/[3,6]`; the paper (Appx. G.1) and the shipped data use `{4..7}/{3..5}/{2..4}/{2,3}` | `src/graphs.py`, `configs/default.yaml` | Ranges set to G.1; edge target `round(d*N/2)` (the shipped data uses `round`, not `floor`) |
| Train and ID splits shared a seed stream (both salted by their first size, 15) | `scripts/generate_data.py` | Seeds salted by split index |
| Deduplication ran within a split only | `src/graphs.py`, `scripts/generate_data.py` | One WL-hash set per k shared across splits, training first |
| Training loop evaluated Final-OOD every `log_every` steps and sized padding from it | `src/train.py`, `scripts/train_model.py` | Loop logs scale-validation only; padding uses the fixed cap `MAX_N = 100`; Final-OOD is evaluated by `scripts/evaluate.py` |

Checks (both exit 0 on the shipped data):

```bash
# V1-V6: ranges, seed separation, cross-split dedup, no Final-OOD in the loop,
# per-degree count statistics, regenerated ID statistics
python scripts/validate_release.py \
    --registry results/EXP00_GRAPH_REGISTRY__GRAPH_METADATA.csv \
    --stats    results/EXP07_DATASET_STATISTICS__DERIVED_DATASET_STATS.csv

# Exact regeneration of every shipped ER evaluation graph from the registry's
# (N, mean degree, seed): 7,200 graphs, edge counts and true counts, 0 mismatches
python scripts/regenerate_from_registry.py \
    --registry results/EXP00_GRAPH_REGISTRY__GRAPH_METADATA.csv --out data/ --topology ER
```

The ER generator behind the shipped data is `nx.gnm_random_graph(N, round(d*N/2), seed)`
with the registry's `generation_seed` (exact regeneration verified with networkx 3.6.1).
The transfer topologies (BA, SBM, WS) carry an exact edge-count matching step; they
are shipped as graph files.

Two scripts, two purposes: `scripts/regenerate_from_registry.py` reproduces the
paper's evaluation sets exactly; `scripts/generate_data.py` draws fresh graphs
from the same distribution (its training split is a fresh draw, not the paper's
training graphs, which are released with their seeds separately).
