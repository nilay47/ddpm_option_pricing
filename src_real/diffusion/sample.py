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
    model.eval()

    alphas = alphas.to(device, dtype=torch.float32)
    alphas_bar = alphas_bar.to(device, dtype=torch.float32)
    betas = betas.to(device, dtype=torch.float32)

    T = int(alphas_bar.shape[0])

    def once(B: int) -> torch.Tensor:
        x = torch.randn(B, H, device=device)  # x_T ~ N(0, I)

        for t in reversed(range(T)):
            a_bar_t = alphas_bar[t]
            t01 = torch.full((B,), (t + 0.5) / T, device=device)

            eps_hat = model(x, t01)  # (B,H)

            # predict x0
            x0_hat = (x - torch.sqrt(1.0 - a_bar_t) * eps_hat) / torch.sqrt(a_bar_t)

            if t == 0:
                x = x0_hat
                continue

            a_t = alphas[t]
            a_bar_prev = alphas_bar[t - 1]
            beta_t = betas[t]

            # posterior mean coefficients
            c1 = torch.sqrt(a_bar_prev) * beta_t / (1.0 - a_bar_t)
            c2 = torch.sqrt(a_t) * (1.0 - a_bar_prev) / (1.0 - a_bar_t)
            mean = c1 * x0_hat + c2 * x

            # posterior variance (beta_tilde)
            post_var = beta_t * (1.0 - a_bar_prev) / (1.0 - a_bar_t)
            post_var = torch.clamp(post_var, min=1e-12)

            x = mean + torch.sqrt(post_var) * torch.randn_like(x)

        return x

    outs = []
    left = int(n_samples)
    chunk = int(min(chunk, left)) if left > 0 else 1

    while left > 0:
        m = min(chunk, left)
        outs.append(once(m).cpu())
        left -= m

    z = torch.cat(outs, dim=0)
    return z.numpy().astype(np.float32)