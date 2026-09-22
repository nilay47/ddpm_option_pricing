"""
The frozen reverse sampler.

This is `src_v2.diffusion.sampler.sample_ddpm_x0` with two additions and no
other change to the arithmetic:

  * `t_start`: the chain is initialised with y ~ N(0, I) at step t_start and
    runs t_start, t_start-1, ..., 0. With the cosine schedule beta_{T-1} is
    clipped to 0.999, so the step t = T-1 multiplies the network's eps error
    by 1/sqrt(alpha_{T-1}) = 31.6. For a 21-dim path that single step throws
    the chain off the data manifold (57.7 % of paths beyond the cap in the
    dry run; 0 % when it is skipped). x_{T-2} is N(0, I) to within
    abar_{T-2} = 2.4e-6, so starting there loses nothing. P_theta is DEFINED
    with t_start = last unclipped step (DECISIONS.md section 3).

  * `eps_correction`: optional hook (y, t01, t) -> delta added to eps_hat.
    None for P_theta; the learned h_psi correction in step 4. Keeping the hook
    inside the same loop is what "one sampler everywhere" means.

`test_shapes.py` asserts that with t_start = T-1 and no hook the output is
bitwise identical to `sample_ddpm_x0` for the same seed.
"""

import torch


def last_unclipped_step(betas: torch.Tensor, clip: float = 0.999) -> int:
    """Largest t with beta_t < clip. For the cosine schedule at T=1000: 998."""
    idx = torch.nonzero(betas < clip).flatten()
    if idx.numel() == 0:
        raise ValueError("every beta is at the clip; schedule is degenerate")
    return int(idx.max())


@torch.no_grad()
def reverse_ancestral(model, shape, alphas, alphas_bar, betas, device,
                      t_start: int = None, eps_correction=None):
    """
    Ancestral DDPM reverse chain (same arithmetic as sample_ddpm_x0).

    Args:
        model: eps predictor, model(y, t01) with t01 = (t + 0.5) / T
        shape: (B, D)
        alphas, alphas_bar, betas: full length-T schedules (T sets the time
            encoding; never pass a truncated schedule)
        t_start: first step of the chain; default T-1 (= sample_ddpm_x0)
        eps_correction: None, or f(y, t01, t) -> tensor like eps_hat
    Returns:
        y: (B, D) sample in the model's (standardized) space
    """
    model.eval()
    B, D = shape
    T = alphas_bar.shape[0]
    t_start = T - 1 if t_start is None else int(t_start)
    if not (0 <= t_start < T):
        raise ValueError(f"t_start={t_start} outside [0, {T-1}]")

    y = torch.randn(B, D, device=device)

    for t in reversed(range(t_start + 1)):
        a_bar_t = alphas_bar[t]
        t01 = torch.full((B,), (t + 0.5) / T, device=device)

        eps_hat = model(y, t01)
        if eps_correction is not None:
            eps_hat = eps_hat + eps_correction(y, t01, t)

        if t > 0:
            post_var = (1.0 - alphas_bar[t - 1]) / (1.0 - a_bar_t) * betas[t]
            post_var = torch.clamp(post_var, min=1e-12)

            mean = (1.0 / torch.sqrt(alphas[t])) * (
                y - (betas[t] / torch.sqrt(1.0 - a_bar_t)) * eps_hat
            )

            y = mean + torch.sqrt(post_var) * torch.randn_like(y)
        else:
            y = (1.0 / torch.sqrt(alphas[t])) * (
                y - (betas[t] / torch.sqrt(1.0 - a_bar_t)) * eps_hat
            )

    return y
