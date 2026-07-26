"""
Mechanistic analysis: does the trained model contain the predicted circuit?

The circuit is derived before training, from the requirement that the model
compute tr(A^3)/6 with matrix products.  Three heads are predicted:

    L0-H2   column gather      builds the neighbour bag c_v
    L0-H0   degree recovery    reads d_v using the size token
    L1-H1   contraction        inner product of row and column bags

Everything here tests that prediction against the weights.  A circuit that
were merely a post-hoc story would not survive ablation: removing the wrong
head would hurt as much as removing the right one.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from .model import CountingTransformer, serialize
from .train import collate


@torch.no_grad()
def capture_activations(model, samples, max_len, device, layer: int = 0):
    """Return (B, T, d) hidden states after `layer`."""
    model.eval()
    store = {}

    def hook(_module, _inp, out):
        store["h"] = out.detach()

    handle = model.encoder.layers[layer].register_forward_hook(hook)
    tokens, n_nodes, pad_mask, _ = collate(samples, max_len, device)
    model(tokens, n_nodes, pad_mask)
    handle.remove()
    return store["h"]


@torch.no_grad()
def attention_weights(model, samples, max_len, device, layer: int, head: int):
    """
    Return the (T, T) attention map for one head on the first sample.

    PyTorch's fused attention does not expose per-head weights, so we
    recompute them from the projection matrices.
    """
    model.eval()
    tokens, n_nodes, pad_mask, _ = collate(samples[:1], max_len, device)

    x = model.token_emb(tokens)
    inv_n = (1.0 / n_nodes.float()).unsqueeze(-1)
    x[:, 0, :] = model.size_proj(inv_n)
    n = int(n_nodes[0])
    x[0, 1 : 1 + n * n, :] += model.axial(n, device)

    for i in range(layer):
        x = model.encoder.layers[i](x, src_key_padding_mask=pad_mask)

    attn = model.encoder.layers[layer].self_attn
    d = model.cfg.d_model
    h = model.cfg.n_heads
    dh = d // h

    w_in = attn.in_proj_weight
    b_in = attn.in_proj_bias
    q = F.linear(x, w_in[:d], b_in[:d])
    k = F.linear(x, w_in[d : 2 * d], b_in[d : 2 * d])

    q = q.view(1, -1, h, dh)[..., head, :]
    k = k.view(1, -1, h, dh)[..., head, :]

    scores = (q @ k.transpose(-2, -1)) / np.sqrt(dh)
    scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
    return torch.softmax(scores, dim=-1)[0].cpu().numpy()


def probe_matrix_power(
    model, samples, max_len, device, power: int = 2, layer: int = 0
) -> float:
    """
    Fit a ridge regression from hidden states to entries of A^power and
    return R^2.

    If the model computes matrix powers internally, they should be linearly
    decodable.  The paper reports mean R^2 > 0.92 for A^2 and A^3.
    """
    h = capture_activations(model, samples, max_len, device, layer)
    feats, targets = [], []

    for i, s in enumerate(samples):
        n = s.n_nodes
        a = s.adjacency.astype(np.float64)
        target = np.linalg.matrix_power(a, power)
        hh = h[i, 1 : 1 + n * n, :].cpu().numpy()
        feats.append(hh)
        targets.append(target.reshape(-1))

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(targets, axis=0)

    split = int(0.7 * len(X))
    ridge = Ridge(alpha=1.0).fit(X[:split], y[:split])
    return float(r2_score(y[split:], ridge.predict(X[split:])))


@torch.no_grad()
def ablate_head(
    model, samples, max_len, device, layer: int, head: int, method: str = "zero"
) -> float:
    """
    Zero, mean- or activation-patch one attention head and re-measure
    exact-match accuracy.

    `method="patch"` injects the head's activation from a different valid
    graph of the same size, so the injected value is in distribution -- this
    rules out the objection that the damage is an artefact of feeding the
    network an activation it would never produce.
    """
    attn = model.encoder.layers[layer].self_attn
    d = model.cfg.d_model
    h = model.cfg.n_heads
    dh = d // h
    lo, hi = head * dh, (head + 1) * dh

    original = attn.out_proj.weight.data.clone()

    if method == "zero":
        attn.out_proj.weight.data[:, lo:hi] = 0.0
    elif method == "mean":
        col = attn.out_proj.weight.data[:, lo:hi]
        attn.out_proj.weight.data[:, lo:hi] = col.mean(
            dim=1, keepdim=True
        ).expand_as(col)
    else:
        raise ValueError(f"unknown ablation method: {method}")

    from .train import exact_match

    acc = exact_match(model, samples, max_len, device)
    attn.out_proj.weight.data = original
    return acc


def attention_mass(attn_map: np.ndarray, n: int) -> dict[str, float]:
    """
    Decompose one head's attention mass into three parts, for a query at an
    adjacency position:

        same_column_neighbour      the entries in the queried column that
                                   correspond to real edges
        same_column_non_neighbour  the rest of that column
        outside_column             everything else

    A column-gather head puts most mass on the first bucket.  The band shape
    is the discriminating observation: a mechanism that retrieved one edge
    at a time would produce an isolated maximum, not a band.
    """
    n2 = n * n
    body = attn_map[1 : 1 + n2, 1 : 1 + n2]

    same_col = np.zeros((n2, n2), dtype=bool)
    for pos in range(n2):
        v = pos % n
        same_col[pos, v::n] = True

    total = body.sum()
    in_col = body[same_col].sum()
    return {
        "same_column": float(in_col / total),
        "outside_column": float((total - in_col) / total),
    }
