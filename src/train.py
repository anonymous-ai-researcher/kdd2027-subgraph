"""
The grokking protocol.

Two settings do the work, and neither is a tuning detail:

  weight decay lambda = 1.0
      Both the memorizing and the generalizing circuit fit the training set,
      but the generalizing one reaches the same loss at a smaller parameter
      norm.  Decoupled weight decay therefore selects it -- eventually.  At
      lambda <= 0.1 the pressure is too weak and the model stays memorized;
      at lambda >= 3 it destroys both circuits.

  training steps = 1e5
      The data loss reaches ~0 within about 1e3 steps.  Everything after
      that is circuit formation and cleanup.  Stopping at convergence of the
      training loss -- the conventional choice -- stops two orders of
      magnitude too early and sees only memorization.

Full-batch training is a convenience, not a requirement: batch 100 still
groks, just later and less cleanly.  We use full batch because it makes the
norm trajectory legible.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .graphs import MAX_N
from .model import CountingTransformer, ModelConfig, serialize


@dataclass
class TrainConfig:
    k: int = 3
    lr: float = 1e-3
    weight_decay: float = 1.0        # lambda
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    steps: int = 100_000
    batch_size: int = 2000           # full batch
    seed: int = 42
    log_every: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def collate(samples, max_len: int, device: str):
    """Build one padded batch from a list of GraphSample."""
    toks, lens, ns, ys = [], [], [], []
    for s in samples:
        t, L = serialize(s.adjacency, max_len)
        toks.append(t)
        lens.append(L)
        ns.append(s.n_nodes)
        ys.append(float(s.true_count))

    tokens = torch.tensor(toks, dtype=torch.long, device=device)
    n_nodes = torch.tensor(ns, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.float, device=device)

    ar = torch.arange(max_len, device=device).unsqueeze(0)
    pad_mask = ar >= torch.tensor(lens, device=device).unsqueeze(1)

    return tokens, n_nodes, pad_mask, y


@torch.no_grad()
def exact_match(model, samples, max_len: int, device: str, chunk: int = 256):
    """Fraction of graphs whose rounded prediction equals the true count."""
    model.eval()
    correct = 0
    for i in range(0, len(samples), chunk):
        part = samples[i : i + chunk]
        tokens, n_nodes, pad_mask, y = collate(part, max_len, device)
        pred = model(tokens, n_nodes, pad_mask)
        correct += int((pred.round() == y).sum())
    model.train()
    return 100.0 * correct / len(samples)


def parameter_norm(model) -> float:
    return float(
        torch.sqrt(sum((p**2).sum() for p in model.parameters())).item()
    )


def train(
    train_samples,
    val_samples,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    out_dir: Path,
):
    """
    Run one grokking training and write the trajectory.

    The trajectory is the primary artefact: the transition is only visible
    in it, and the complexity-collapse diagnostic is computed from it.
    """
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Padding length comes from the task-level cap, never from a test split.
    max_n = MAX_N
    max_len = 1 + max_n * max_n

    device = train_cfg.device
    model = CountingTransformer(model_cfg).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        eps=train_cfg.eps,
        weight_decay=train_cfg.weight_decay,
    )

    tokens, n_nodes, pad_mask, y = collate(train_samples, max_len, device)
    trajectory = []

    for step in range(1, train_cfg.steps + 1):
        pred = model(tokens, n_nodes, pad_mask)
        loss = F.mse_loss(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(
            torch.sqrt(
                sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)
            ).item()
        )
        opt.step()

        if step % train_cfg.log_every == 0 or step == 1:
            train_acc = exact_match(model, train_samples, max_len, device)
            val_acc = exact_match(model, val_samples, max_len, device)  # SCALE_VAL only
            row = {
                "step": step,
                "data_loss": float(loss.item()),
                "train_accuracy": train_acc,
                "scale_val_accuracy": val_acc,
                "parameter_norm": parameter_norm(model),
                "gradient_norm": grad_norm,
            }
            trajectory.append(row)
            print(
                f"step {step:>7d}  loss {row['data_loss']:.4e}  "
                f"train {train_acc:6.2f}  scale_val {val_acc:6.2f}  "
                f"|theta| {row['parameter_norm']:7.2f}"
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            "step": train_cfg.steps,
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))

    return model, trajectory


def detect_complexity_collapse(
    trajectory: list[dict], threshold: float = 0.5
) -> tuple[bool, int | None]:
    """
    Complexity collapse: a sustained decrease in ||theta||_2 of at least
    `threshold` from its post-memorization peak.

    Returns (groks, collapse_step).  The collapse step is the point of
    maximum negative slope in a smoothed norm trace.

    Empirically the two populations are well separated -- runs that
    generalize erase 52-71% of the peak norm, runs that do not erase under
    6% -- so the exact threshold in [0.1, 0.5] does not change the verdict.
    """
    if not trajectory:
        return False, None

    steps = np.array([r["step"] for r in trajectory])
    norms = np.array([r["parameter_norm"] for r in trajectory])
    train_acc = np.array([r["train_accuracy"] for r in trajectory])

    memorized = np.flatnonzero(train_acc > 99.0)
    if memorized.size == 0:
        return False, None
    start = int(memorized[0])

    peak_idx = start + int(np.argmax(norms[start:]))
    peak = norms[peak_idx]
    final = norms[-1]

    kappa = (peak - final) / peak if peak > 0 else 0.0
    if kappa < threshold:
        return False, None

    # Smooth with a centred moving average, then take the steepest descent.
    w = max(3, len(norms) // 50)
    kernel = np.ones(w) / w
    smooth = np.convolve(norms, kernel, mode="same")
    slope = np.gradient(smooth)
    collapse_idx = peak_idx + int(np.argmin(slope[peak_idx:]))

    return True, int(steps[collapse_idx])
