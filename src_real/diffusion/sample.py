# src_real/diffusion/sample.py

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def sample_blocks_ddpm(
    model: torch.nn.Module,
    n_samples: int,
    H: int,
    alphas: torch.Tensor,
    alphas_bar: torch.Tensor,
    betas: torch.Tensor,
    device: torch.device,
    chunk: int = 5000,
) -> np.ndarray:
    """
    Ancestral DDPM sampler in standardized block space.

    Produces z ~ p_theta(z0) where z0 is an H-dim standardized return block.

    Args:
        model: epsilon predictor, (x_t, t01) -> eps_hat with shape (B,H)
        n_samples: number of blocks to sample
        H: block dimension
        alphas, alphas_bar, betas: schedules of length T
        device: cuda/cpu
        chunk: sample in chunks to limit memory

    Returns:
        z_blocks: (n_samples, H) numpy float32
    """
    model.eval()

    alphas = alphas.to(device, dtype=torch.float32)
    alphas_bar = alphas_bar.to(device, dtype=torch.float32)
    betas = betas.to(device, dtype=torch.float32)

    T = int(alphas_bar.shape[0])

    def once(B: int) -> torch.Tensor:
        x = torch.randn(B, H, device=device)  # x_T ~ N(0,I)

        for t in reversed(range(T)):
            a_bar_t = alphas_bar[t]
            t01 = torch.full((B,), (t + 0.5) / T, device=device)

            eps_hat = model(x, t01)  # (B,H)

            if t > 0:
                post_var = (1.0 - alphas_bar[t - 1]) / (1.0 - a_bar_t) * betas[t]
                post_var = torch.clamp(post_var, min=1e-12)

                mean = (1.0 / torch.sqrt(alphas[t])) * (
                    x - (betas[t] / torch.sqrt(1.0 - a_bar_t)) * eps_hat
                )
                x = mean + torch.sqrt(post_var) * torch.randn_like(x)
            else:
                x = (1.0 / torch.sqrt(alphas[t])) * (
                    x - (betas[t] / torch.sqrt(1.0 - a_bar_t)) * eps_hat
                )

        return x  # (B,H), standardized block

    outs = []
    left = int(n_samples)
    chunk = int(min(chunk, left)) if left > 0 else 1

    while left > 0:
        m = min(chunk, left)
        outs.append(once(m).cpu())
        left -= m

    z = torch.cat(outs, dim=0)  # (n_samples,H)
    return z.numpy().astype(np.float32)
