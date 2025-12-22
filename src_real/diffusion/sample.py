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
    beta_min: float = 1e-8,
    beta_max: float = 0.02,
    var_floor: float = 1e-12,
) -> np.ndarray:
    """
    Ancestral DDPM sampler in standardized block space.
    Produces z ~ p_theta(x0) where x0 is an H-dim standardized return block.
    """
    model.eval()

    alphas = alphas.to(device, dtype=torch.float32)
    alphas_bar = alphas_bar.to(device, dtype=torch.float32)
    betas = betas.to(device, dtype=torch.float32)

    # Hard safety: a sane DDPM beta schedule should never approach 1.
    betas = torch.clamp(betas, min=beta_min, max=beta_max)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    T = int(alphas_bar.shape[0])

    def once(B: int) -> torch.Tensor:
        x = torch.randn(B, H, device=device)  # x_T ~ N(0,I)

        for t in reversed(range(T)):
            a_bar_t = alphas_bar[t]  # scalar
            t01 = torch.full((B,), (t + 0.5) / T, device=device)

            eps_hat = model(x, t01)  # (B,H)

            if t > 0:
                a_t = alphas[t]
                a_bar_prev = alphas_bar[t - 1]
                beta_t = betas[t]

                # x0 estimate from epsilon prediction
                x0_hat = (x - torch.sqrt(1.0 - a_bar_t) * eps_hat) / torch.sqrt(a_bar_t)

                # True DDPM posterior mean (Ho et al.)
                c1 = torch.sqrt(a_bar_prev) * beta_t / (1.0 - a_bar_t)
                c2 = torch.sqrt(a_t) * (1.0 - a_bar_prev) / (1.0 - a_bar_t)
                mean = c1 * x0_hat + c2 * x

                # True DDPM posterior variance
                post_var = beta_t * (1.0 - a_bar_prev) / (1.0 - a_bar_t)
                post_var = torch.clamp(post_var, min=var_floor)

                x = mean + torch.sqrt(post_var) * torch.randn_like(x)

            else:
                # At t=0, output x0 directly
                x = (x - torch.sqrt(1.0 - a_bar_t) * eps_hat) / torch.sqrt(a_bar_t)

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