import math
import numpy as np
import torch

from src_v2.finance.risk_neutral import rn_noise_shift_std


@torch.no_grad()
def sample_returns_Q_epsilon_shift(
    model: torch.nn.Module,
    n_paths: int,
    H_steps: int,
    mu: float,
    r: float,
    sigma: float,
    dt: float,
    alphas: torch.Tensor,
    alphas_bar: torch.Tensor,
    betas: torch.Tensor,
    s0: float,
    device: torch.device,
    apply_mean_projection: bool = False,
) -> np.ndarray:
    """
    Legacy-style sampler: apply RN shift in epsilon space inside the reverse loop.
    Returns log-returns Y of shape (n_paths, H_steps).
    """
    model.eval()
    T = alphas_bar.shape[0]

    noise_shift_std = rn_noise_shift_std(
        mu=mu,
        r=r,
        dt=dt,
        s0=s0,
        alphas_bar=alphas_bar,
        device=device,
    )

    def once(B: int) -> torch.Tensor:
        y = torch.randn(B, 1, device=device)

        for t in reversed(range(T)):
            a_bar_t = alphas_bar[t]
            t01 = torch.full((B,), (t + 0.5) / T, device=device)

            eps_hat = model(y, t01)
            eps_Q = eps_hat - noise_shift_std[t].view(1, 1)

            if t > 0:
                post_var = (1.0 - alphas_bar[t - 1]) / (1.0 - a_bar_t) * betas[t]
                post_var = torch.clamp(post_var, min=1e-12)

                mean = (1.0 / torch.sqrt(alphas[t])) * (
                    y - (betas[t] / torch.sqrt(1.0 - a_bar_t)) * eps_Q
                )
                y = mean + torch.sqrt(post_var) * torch.randn_like(y)
            else:
                y = (1.0 / torch.sqrt(alphas[t])) * (
                    y - (betas[t] / torch.sqrt(1.0 - a_bar_t)) * eps_Q
                )

        return y  # standardized shock-like output, shape (B,1)

    needed = n_paths * H_steps
    outs = []
    left = needed
    chunk = min(10_000, needed)

    while left > 0:
        m = min(chunk, left)
        outs.append(once(m).cpu())
        left -= m

    Z = torch.cat(outs, 0).view(n_paths, H_steps)  # standardized space

    mQ = (r - 0.5 * sigma**2) * dt
    Y = s0 * Z + mQ

    if apply_mean_projection:
        emp = Y.mean()
        Y = Y + (mQ - emp)

    return Y.numpy()
