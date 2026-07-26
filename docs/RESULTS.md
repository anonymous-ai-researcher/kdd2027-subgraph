# Measured results

Every CSV in `results/` is a measurement, not an assertion.

| File | Contents |
|:---|:---|
| `EXP01_MAIN_RESULTS__SEED_LEVEL_ACCURACY.csv` | 20 models × 4 cycle lengths × 5 seeds × 2 splits |
| `EXP02_TOPOLOGY_TRANSFER__DERIVED_TOPOLOGY_RESULTS.csv` | zero-shot transfer to BA / SBM / WS |
| `EXP03_PER_N__DERIVED_PER_N.csv` | accuracy at each evaluated graph size |
| `EXP04_TRAINING_DYNAMICS__DERIVED_MEAN_TRAJECTORIES.csv` | loss, accuracy, parameter norm, per-head entropy vs step |
| `EXP05_..._HEAD_PATCHING_SUMMARY.csv` | single-head ablation, three intervention methods |
| `EXP05_..._ATTENTION_MASS_AND_PROBE_SUMMARY.csv` | attention decomposition and linear-probe R² |
| `EXP06_ABLATIONS__SEED_LEVEL_ABLATIONS.csv` | architecture and weight-decay sweeps |
| `EXP07_DATASET_STATISTICS__DERIVED_DATASET_STATS.csv` | count distributions per split |
| `EXP08_ERROR_ANALYSIS__GRAPH_LEVEL_ERROR_PROFILE.csv` | per-graph predictions and error flags |
| `EXP21_COST_BENCHMARK__SUMMARY.csv` | latency, FLOPs, GPU-hours |
| `EXP22_NONCYCLE_PATTERNS__SUMMARY.csv` | paw, bull, tadpole, C₄+pendant, induced C₆ |
| `EXP31_GROUND_TRUTH_CHECKS__SUMMARY.csv` | three-way label verification, 9,600 graphs |
| `significance_19_baselines.csv` | Welch t, Holm-corrected p, Cohen's d for 19 baselines |

## Reading the trajectory file

`EXP04` is the one to look at first. The `parameter_norm` column is the
diagnostic: under the grokking protocol it rises during memorization, peaks
around step 1.5×10⁴, then falls by roughly 74%. Under the no-grokking
protocol it never falls, and OOD accuracy stays near 20% however long you
train.

## Statistical protocol

Welch's t-test (unequal variances) with Holm–Bonferroni correction within
each (k, test-set) family, treating the 19 comparisons against our method as
one family. Cohen's d uses the pooled standard deviation. Every Welch p-value
is recoverable from the mean, standard deviation and n = 5 printed in the
tables; the rank-based Mann–Whitney statistic additionally needs the
seed-level values, which are released here.
