import math
from typing import Tuple

import numpy as np
import torch

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Create a cosine beta schedule.

    Args:
        timesteps (int): Number of timesteps in the diffusion process.
        s (float): Small offset to prevent singularities.

    Returns:
        torch.Tensor: A tensor of beta values for each timestep.
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 1e-6, 0.999)
    
    return torch.tensor(betas, dtype=torch.float32)

def make_alpha_schedule(
        T: int,
        device: torch.device,
        s: float = 0.008
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create alpha and beta schedules for diffusion models using a cosine schedule.
    Args:
        T (int): Number of timesteps.
        device (torch.device): Device to place the tensors on.
        s (float): Small offset to prevent singularities.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors of alphas and betas.
    """
    betas = cosine_beta_schedule(timesteps=T, s=s).to(device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar

def sinusoidal_time_embedding(timesteps: torch, dim: int = 64) -> torch.Tensor:
    """
    Create sinusoidal time embeddings.

    Args:
        timesteps (torch.Tensor): Tensor of timesteps.
        dim (int): Dimension of the embedding.

    Returns:
        torch.Tensor: Sinusoidal time embeddings.
    """
    half_dim = dim // 2
    frequencies = torch.exp(
        torch.linspace(
            math.log(1e-4),
            math.log(1e4),
            half_dim,
            device=timesteps.device
        )
    )
    args = timesteps[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros(timesteps.shape[0], 1, device=timesteps.device)], dim=1)
    return embedding
