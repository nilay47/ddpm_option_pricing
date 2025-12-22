import numpy as np
import torch
from torch.utils.data import Dataset


class ReturnBlockDataset(Dataset):
    """
    Takes a 1D array of log returns and returns rolling windows of length H.
    """
    def __init__(self, returns: np.ndarray, H: int):
        self.returns = np.asarray(returns, dtype=np.float32).reshape(-1)
        self.H = int(H)
        if len(self.returns) <= self.H:
            raise ValueError("Not enough returns for given block length H.")

    def __len__(self):
        return len(self.returns) - self.H

    def __getitem__(self, idx: int):
        x = self.returns[idx: idx + self.H]          # (H,)
        return torch.from_numpy(x).view(1, self.H)   # (1,H) or (H,) depending on preference
