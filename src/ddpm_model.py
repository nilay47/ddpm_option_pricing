import torch
import torch.nn as nn
from src.schedules import sinusoidal_time_embedding

class ScoreMLP(nn.Module):
    """
    A simple MLP model for score estimation in diffusion models.
    Args:
        input_dim (int): Dimension of the input data.
        hidden_dim (int): Dimension of the hidden layers.
        time_emb_dim (int): Dimension of the time embedding.
    """
    def __init__(self, hidden_dim: int = 256, time_emb_dim: int = 64):
        super(ScoreMLP, self).__init__()
        self.time_emb_dim = time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(1 + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    """
    Forward pass of the ScoreMLP.
    Args:
        x (torch.Tensor): Input data tensor.
        t (torch.Tensor): Timestep tensor.
    Returns:
        torch.Tensor: Output tensor after passing through the network.
    """
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, dim=self.time_emb_dim)
        x = torch.cat([x, t_emb], dim=1)
        return self.net(x)