from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


def linear_beta_schedule(T: int, beta_min: float = 1e-4, beta_max: float = 2e-2) -> torch.Tensor:
    betas = np.linspace(beta_min, beta_max, T, dtype=np.float64)
    return torch.tensor(betas, dtype=torch.float32)


def cosine_beta_schedule(T: int, s: float = 0.008, beta_max: float = 2e-2) -> torch.Tensor:
    # cosine cumulative alpha_bar then convert to betas, but cap beta_max aggressively
    steps = T + 1
    x = np.linspace(0, T, steps, dtype=np.float64)
    alphas_bar = np.cos(((x / T) + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = np.clip(betas, 1e-6, float(beta_max))  # IMPORTANT: never 0.999
    return torch.tensor(betas, dtype=torch.float32)


def make_alpha_schedule(
    T: int,
    device: torch.device,
    schedule: str = "linear",      # "linear" or "cosine"
    beta_min: float = 1e-4,
    beta_max: float = 2e-2,
    s: float = 0.008,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if schedule == "linear":
        betas = linear_beta_schedule(T, beta_min=beta_min, beta_max=beta_max).to(device)
    elif schedule == "cosine":
        betas = cosine_beta_schedule(T, s=s, beta_max=beta_max).to(device)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    # basic sanity
    if float(betas.max()) >= 0.2:
        raise ValueError(f"beta_max too large for stable sampling: {float(betas.max())}")
    if float(alphas_bar[-1]) <= 0.0:
        raise ValueError(f"alphas_bar[-1] must be > 0, got {float(alphas_bar[-1])}")

    return betas.float(), alphas.float(), alphas_bar.float()


def sinusoidal_time_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    if t.ndim != 1:
        raise ValueError(f"t must be shape (B,), got {tuple(t.shape)}")
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1e-4), math.log(1e4), half, device=t.device))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=t.device)], dim=1)
    return emb