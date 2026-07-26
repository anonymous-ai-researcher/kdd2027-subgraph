"""
The counting Transformer.

Input is the flattened adjacency matrix and nothing else: N^2 binary tokens,
stripped of every structural prior.  What makes counting reachable is the
positional scheme, not the token content.

--------------------------------------------------------------------------
Why axial codes
--------------------------------------------------------------------------
Serialize A row-major, so entry (u, v) lands at position u*N + v.  A
one-dimensional positional encoding can only see the *offset* between two
positions.  The two entries A[u,v] and A[v,w] share the endpoint v, but
their offset is (v-u)*N + (w-v), which depends on u and w individually.  A
head that must attend from one to the other therefore needs an offset that
varies with the query -- exactly what a fixed relative encoding cannot do.

Axial codes address by *identity* instead.  Draw a random sign vector
h(u) in {+1,-1}^{d_p} per node index and set

    phi(u) = h(u) / sqrt(d_p),

then give position (u, v) the embedding

    x[pos(u,v)] = Emb[A[u,v]] + [phi(u); 0] + [0; phi(v)].

Two entries in the same column now share the second block exactly, so a
head can retrieve them with a content-addressed query regardless of how far
apart they sit in the sequence.  Near-orthogonality of random sign codes
(Johnson-Lindenstrauss) keeps the cross-talk below the rounding margin.

--------------------------------------------------------------------------
Why a size token
--------------------------------------------------------------------------
Softmax attention normalizes by the mass of the set it attends to, so a head
cannot return an unnormalized neighbour sum -- it returns a *mean*.
Recovering the degree from that mean requires N.  A single prepended token
carrying 1/N supplies it.  The token is visible to every head but excluded
from the pooled readout.

--------------------------------------------------------------------------
Why sum pooling
--------------------------------------------------------------------------
The trace is a sum over matrix entries, so the readout must be a sum.  Mean
pooling would divide by N^2, a factor that changes at test time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, ZERO, ONE, SIZE = 0, 1, 2, 3  # token ids


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    d_pos: int = 24          # per-axis code width; 2 * d_pos <= d_model
    max_nodes: int = 128     # largest N the code table supports
    dropout: float = 0.0
    code_seed: int = 0       # seed for the fixed axial codes

    def __post_init__(self) -> None:
        if 2 * self.d_pos > self.d_model:
            raise ValueError(
                f"2 * d_pos ({2 * self.d_pos}) exceeds d_model ({self.d_model})"
            )


class AxialEncoding(nn.Module):
    """Fixed, randomly drawn node-identity codes.  Never trained."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        g = torch.Generator().manual_seed(cfg.code_seed)
        signs = torch.randint(0, 2, (cfg.max_nodes, cfg.d_pos), generator=g)
        codes = (signs.float() * 2.0 - 1.0) / math.sqrt(cfg.d_pos)
        self.register_buffer("codes", codes, persistent=True)
        self.d_pos = cfg.d_pos
        self.d_model = cfg.d_model

    def forward(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Return (n*n, d_model): the axial part of the embedding for one graph.

        Row index u fills the first d_pos dimensions, column index v fills
        the next d_pos.  The remainder is zero and left for the token
        embedding to use.
        """
        row = self.codes[:n].to(device)                      # (n, d_pos)
        out = torch.zeros(n, n, self.d_model, device=device)
        out[:, :, : self.d_pos] = row.unsqueeze(1)           # phi(u)
        out[:, :, self.d_pos : 2 * self.d_pos] = row.unsqueeze(0)  # phi(v)
        return out.reshape(n * n, self.d_model)


class CountingTransformer(nn.Module):
    """Bidirectional encoder over the serialized adjacency matrix."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg

        self.token_emb = nn.Embedding(4, c.d_model, padding_idx=PAD)
        self.axial = AxialEncoding(c)
        self.size_proj = nn.Linear(1, c.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.d_ff,
            dropout=c.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=c.n_layers)
        self.readout = nn.Linear(c.d_model, 1)

    def forward(
        self,
        tokens: torch.Tensor,      # (B, 1 + L_max) long, position 0 is SIZE
        n_nodes: torch.Tensor,     # (B,) long
        pad_mask: torch.Tensor,    # (B, 1 + L_max) bool, True where padding
    ) -> torch.Tensor:
        b, total = tokens.shape
        device = tokens.device

        x = self.token_emb(tokens)                            # (B, T, d)

        # Size token: overwrite position 0 with a projection of 1/N.
        inv_n = (1.0 / n_nodes.float()).unsqueeze(-1)         # (B, 1)
        x[:, 0, :] = self.size_proj(inv_n)

        # Axial codes, per graph (sizes differ within a batch).
        for i in range(b):
            n = int(n_nodes[i])
            x[i, 1 : 1 + n * n, :] = x[i, 1 : 1 + n * n, :] + self.axial(
                n, device
            )

        h = self.encoder(x, src_key_padding_mask=pad_mask)    # (B, T, d)

        # Sum pooling over adjacency positions only: the size token and any
        # padding contribute nothing.
        keep = (~pad_mask).float()
        keep[:, 0] = 0.0
        pooled = (h * keep.unsqueeze(-1)).sum(dim=1)          # (B, d)

        return self.readout(pooled).squeeze(-1)              # (B,)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def serialize(adjacency, max_len: int) -> tuple[list[int], int]:
    """
    Flatten one adjacency matrix into the token sequence, prefixed by the
    size token and right-padded to `max_len`.

    Returns (tokens, true_length_including_size_token).
    """
    import numpy as np

    a = np.asarray(adjacency, dtype=np.uint8)
    n = a.shape[0]
    body = [ONE if v else ZERO for v in a.reshape(-1).tolist()]
    seq = [SIZE] + body
    true_len = len(seq)
    seq = seq + [PAD] * (max_len - true_len)
    return seq, true_len
