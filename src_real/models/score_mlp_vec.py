# src_real/models/score_mlp_vec.py

from __future__ import annotations

import torch
import torch.nn as nn

from src_real.diffusion.schedules import sinusoidal_time_embedding


class ScoreMLPVec(nn.Module):
    """
    Vector score network for block diffusion.

    Input:
        x: (B, H)   noisy block at diffusion time t
        t: (B,)     time in (0,1)

    Output:
        eps_hat: (B, H) predicted noise
    """

    def __init__(self, H: int, hidden_dim: int = 512, time_emb_dim: int = 64):
        super().__init__()
        self.H = int(H)
        self.time_emb_dim = int(time_emb_dim)

        self.net = nn.Sequential(
            nn.Linear(self.H + self.time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.H),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H)
            t: (B,) or (B,1) with values in (0,1)

        Returns:
            (B, H)
        """
        if x.ndim != 2 or x.shape[1] != self.H:
            raise ValueError(f"x must have shape (B,{self.H}), got {tuple(x.shape)}")

        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"t must have shape (B,), got {tuple(t.shape)} for B={x.shape[0]}")

        t_emb = sinusoidal_time_embedding(t, dim=self.time_emb_dim)  # (B, time_emb_dim)
        h = torch.cat([x, t_emb], dim=1)
        return self.net(h)
