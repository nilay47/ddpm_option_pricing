import math
import numpy as np
import torch

from src_v2.diffusion.sampler import sample_ddpm_x0
from src_v2.finance.risk_neutral import rn_noise_shift_std


@torch.no_grad()
def sample_returns_iid(
    model: torch.nn.Module,
    n_paths: int,
    H_steps: int,
    alphas: torch.Tensor,
    alphas_bar: torch.Tensor,
    betas: torch.Tensor,
    device: torch.device,
    # finance params
    mu: float,
    r: float,
    sigma: float,
    dt: float,
    # normalization
    s0: float,
    # toggles
    apply_rn_shift: bool = True,
    apply_mean_projection: bool = False,
) -> np.ndarray:
    """
    Samples iid log-returns under (approx) Q via epsilon-shift.

    Returns:
        np.ndarray of shape (n_paths, H_steps)
    """
    T = alphas_bar.shape[0]

    # sample standardized x0 in training space, shape (n_paths*H_steps, 1)
    x0 = sample_ddpm_x0(
        model=model,
        shape=(n_paths * H_steps, 1),
        alphas=alphas,
        alphas_bar=alphas_bar,
        betas=betas,
        device=device,
    )

    # IMPORTANT: At this stage, x0 is what the model was trained to generate.
    # We interpret it as standardized shocks Z ~ something like N(0,1).
    Z = x0.view(n_paths, H_steps).detach()

    # map to log-return space
    mQ = (r - 0.5 * sigma**2) * dt
    Y = s0 * Z + mQ  # (n_paths, H_steps)

    if apply_mean_projection:
        emp = Y.mean()
        Y = Y + (mQ - emp)

    return Y.cpu().numpy()
