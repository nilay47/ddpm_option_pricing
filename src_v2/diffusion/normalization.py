from dataclasses import dataclass
import torch


@dataclass
class Standardizer:
    mean: float
    std: float

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        s = self.std if self.std > 1e-12 else 1.0
        return (x - self.mean) / s

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean