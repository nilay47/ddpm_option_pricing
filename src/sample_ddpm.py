import math
import numpy as np
import torch

@torch.no_grad()
def rn_constants_std(
        mu: float,
        sigma: float,
        r: float,
        dt: float,
        alphas_bar: torch.Tensor,
        s0: float,
        device: torch.device
    ) -> torch.Tensor:
    """
    Standardized Risk-Neutral shift 
    Returns the standardized constants for the Risk-Neutral diffusion process
    Args:
        mu (float): Drift term.
        sigma (float): Volatility term.
        r (float): Risk-free rate.
        dt (float): Time increment.
        alphas_bar (torch.Tensor): Cumulative product of alphas for the diffusion process.
        s0 (float): Daily log return standard deviation of the asset.
        device (torch.device): Device to place the tensors on.
    Returns:
        torch.Tensor: Standardized constants for the Risk-Neutral diffusion process.
    """
    alpha_bar = alphas_bar.to(device, dtype=torch.float32)      # (T,)
    alpha_t = torch.sqrt(alpha_bar)

    dm_std = (r - mu) * dt / s0
    eta_t_std = alpha_t * dm_std

    noise_shift_std = eta_t_std * torch.sqrt(1 - alpha_bar)     # (T,)
    return noise_shift_std                                      # (T,)

def sample_returns_Q_std(
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
        m_hat: float,
        s_hat:float,
        s0: float,
        T: int,
        device: torch.device,
        use_ddim: bool = False,
        n_ddim_steps: int = 250,
    ) -> np.ndarray:
    """
    Sample asset returns under the Risk-Neutral measure using a trained DDPM model.
    Args:
        model (torch.nn.Module): Trained DDPM model.
        n_paths (int): Number of paths to sample.
        H_steps (int): Horizon length for each path.
        mu (float): Drift term.
        r (float): Risk-free rate.
        sigma (float): Volatility term.
        dt (float): Time increment.
        alphas (torch.Tensor): Tensor of alphas for the diffusion process.
        alphas_bar (torch.Tensor): Cumulative product of alphas for the diffusion process.
        betas (torch.Tensor): Tensor of betas for the diffusion process.
        m_hat (float): Mean for data normalization.
        s_hat (float): Standard deviation for data normalization.
    Returns:
        np.ndarray: Sampled asset returns under the Risk-Neutral measure.
    """
    model.eval()
    noise_shift_std = rn_constants_std(mu, sigma, r, dt, alphas_bar, s0, device)

    def once(B: int) -> torch.Tensor:
        y = torch.randn(B, 1, device=device)

        if use_ddim:
            """
            # Evenly spaced sub-grid of indices (descending)
            t_grid = torch.linspace(T - 1, 0, steps=n_ddim_steps, device=device).long()
            for i in range(len(t_grid)):
                t = t_grid[i].item()
                a_bar_t = alphas_bar[t]
                t01 = torch.full((B,), (t + 0.5) / T, device=device)
                eps_hat = model(y, t01)
                eps_Q = eps_hat - noise_shift_std[t].view(1, 1)
                # deterministic step
                x0 = (y - torch.sqrt(1.0 - a_bar_t) * eps_Q) / torch.sqrt(a_bar_t)
                if i < len(t_grid) - 1:
                    t_prev = t_grid[i + 1].item()
                    a_bar_prev = alphas_bar[t_prev]
                    y = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(1.0 - a_bar_prev) * eps_Q
                else:
                    y = x0
            """
        else:
            # Ancestral DDPM (matches notebook)
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

        # P-check calibration
        y = (y - m_hat) / (s_hat if s_hat > 1e-8 else 1.0)
        return y  # (B,1)

    needed = n_paths * H_steps
    outs = []
    left = needed
    chunk = min(1000, needed)
    while left > 0:
        m = min(chunk, left)
        outs.append(once(m))
        torch.cuda.empty_cache()
        left -= m

    Y_std = torch.cat(outs, 0).view(n_paths, H_steps, 1).squeeze(-1)  # (n_paths, H_steps)


    ### very important steps of de-standardizing
    # De-standardize to Q
    mQ = (r - 0.5 * sigma**2) * dt
    Y = s0 * Y_std + mQ

    # todo: check this step!
    # RN mean nudge
    emp = Y.mean()
    Y = Y + (mQ - emp)

    return Y.detach().cpu().numpy()