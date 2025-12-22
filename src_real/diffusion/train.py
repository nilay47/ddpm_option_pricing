# src_real/diffusion/train.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class TrainStats:
    losses: list


def train_ddpm_blocks(
    model: torch.nn.Module,
    train_loader: DataLoader,
    alphas_bar: torch.Tensor,
    T: int,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
) -> TrainStats:
    """
    Train DDPM on H-dimensional standardized return blocks z in R^H.

    The model predicts epsilon in:
        x_t = sqrt(a_bar_t) * x0 + sqrt(1-a_bar_t) * eps

    Args:
        model: epsilon predictor, maps (x_t, t01) -> eps_hat
        train_loader: yields batches shaped (B,H)
        alphas_bar: (T,) cumulative product schedule on device
        T: number of diffusion steps
    """
    model.to(device)
    model.train()

    alphas_bar = alphas_bar.to(device, dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    stats = TrainStats(losses=[])

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # support TensorDataset yielding (x,) or direct x
            if isinstance(batch, (tuple, list)):
                x0 = batch[0]
            else:
                x0 = batch

            x0 = x0.to(device, dtype=torch.float32)  # (B,H)
            B = x0.shape[0]

            t = torch.randint(0, T, (B,), device=device).long()  # (B,)
            a_bar_t = alphas_bar[t].view(B, 1)                   # (B,1)

            eps = torch.randn_like(x0)                           # (B,H)
            x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * eps

            t01 = (t.float() + 0.5) / float(T)                   # (B,)
            eps_hat = model(x_t, t01)                            # (B,H)

            loss = torch.nn.functional.mse_loss(eps_hat, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            lv = float(loss.item())
            stats.losses.append(lv)
            pbar.set_postfix(loss=f"{lv:.4f}")

    return stats
