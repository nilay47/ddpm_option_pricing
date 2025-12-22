import torch
import torch.nn as nn
from src.schedules import sinusoidal_time_embedding


class ScoreMLP(nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int = 512, time_emb_dim: int = 64):
        super().__init__()
        self.data_dim = data_dim
        self.time_emb_dim = time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, dim=self.time_emb_dim)
        x = torch.cat([x, t_emb], dim=1)
        return self.net(x)
