import torch


@torch.no_grad()
def rn_noise_shift_std(
    mu: float,
    r: float,
    dt: float,
    s0: float,
    alphas_bar: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns noise_shift_std[t] used in epsilon space:
        eps_Q = eps_P - noise_shift_std[t]
    Derived for the Gaussian (affine) case.
    """
    alpha_bar = alphas_bar.to(device, dtype=torch.float32)  # (T,)
    dm_std = (r - mu) * dt / s0
    eta_t_std = torch.sqrt(alpha_bar) * dm_std
    noise_shift_std = eta_t_std * torch.sqrt(1 - alpha_bar)
    return noise_shift_std