import torch

@torch.no_grad()
def sample_ddpm_x0(
    model: torch.nn.Module,
    shape: tuple,
    alphas: torch.Tensor,
    alphas_bar: torch.Tensor,
    betas: torch.Tensor,
    device: torch.device,
):
    """
    Pure ancestral DDPM sampler.

    Args:
        model: trained noise-prediction model
        shape: (B, D) output shape
        alphas, alphas_bar, betas: diffusion schedules (length T)
        device: torch device

    Returns:
        x0_hat: sampled x0 in the SAME space the model was trained on
    """
    model.eval()

    B, D = shape
    T = alphas_bar.shape[0]

    y = torch.randn(B, D, device=device)

    for t in reversed(range(T)):
        a_bar_t = alphas_bar[t]
        t01 = torch.full((B,), (t + 0.5) / T, device=device)

        eps_hat = model(y, t01)

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
