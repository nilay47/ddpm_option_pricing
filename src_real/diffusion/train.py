from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    Train DDPM on H-dim standardized return blocks x0 in R^H.

    Forward process:
        x_t = sqrt(a_bar_t) * x0 + sqrt(1 - a_bar_t) * eps
    Model predicts eps.
    """
    model = model.to(device)
    model.train()

    alphas_bar = alphas_bar.to(device=device, dtype=torch.float32)

    # ---- hard safety checks (prevents CUDA device-side asserts) ----
    if alphas_bar.ndim != 1:
        raise ValueError(f"alphas_bar must be 1D, got shape {tuple(alphas_bar.shape)}")

    if int(alphas_bar.shape[0]) != int(T):
        raise ValueError(
            f"Mismatch: T={T} but len(alphas_bar)={int(alphas_bar.shape[0])}. "
            "You must build the schedule with the same T you pass into training."
        )

    if not torch.isfinite(alphas_bar).all():
        raise ValueError("alphas_bar contains NaN or inf")

    # alphas_bar should be decreasing and in (0,1]
    if float(alphas_bar[0]) > 1.0 + 1e-6 or float(alphas_bar[0]) <= 0.0:
        raise ValueError(f"alphas_bar[0] looks invalid: {float(alphas_bar[0])}")
    if float(alphas_bar[-1]) <= 0.0:
        raise ValueError(f"alphas_bar[-1] must be > 0, got {float(alphas_bar[-1])}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    stats = TrainStats(losses=[])

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
            x0 = x0.to(device=device, dtype=torch.float32)  # (B,H)
            B = x0.shape[0]

            t = torch.randint(low=0, high=T, size=(B,), device=device, dtype=torch.long)  # (B,)
            a_bar_t = alphas_bar[t].view(B, 1)  # (B,1)

            eps = torch.randn_like(x0)
            x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * eps

            t01 = (t.float() + 0.5) / float(T)  # (B,)
            eps_hat = model(x_t, t01)            # (B,H)

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