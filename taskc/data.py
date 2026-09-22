"""
Data for the path DDPM: Heston-P return paths -> globally standardized z.

The only two coordinate transforms in Task C live here:

    to_z:     Y  -> z = (Y - m) / s          (once, at dataset construction)
    to_paths: z  -> Y = s z + m -> S = S0 exp(cumsum Y)   (once, at sampler output)

with ONE scalar (m, s) fitted on the training returns (DECISIONS.md section 1).
Everything downstream (constraints, L*, exotics, diagnostics) consumes the
`Paths` object and never sees z.
"""

from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import taskc  # noqa: F401
from config import P_PARAMS          # taskb
from heston import Paths, simulate   # taskb
from taskc.config import TaskCConfig, CFG


@dataclass
class PathStandardizer:
    m: float
    s: float
    S0: float
    r: float
    dt: float

    @classmethod
    def fit(cls, Y: np.ndarray, S0: float, r: float, dt: float) -> "PathStandardizer":
        Y = np.asarray(Y, dtype=np.float64)
        return cls(m=float(Y.mean()), s=float(Y.std()), S0=float(S0), r=float(r), dt=float(dt))

    # ---- Y <-> z -------------------------------------------------------------
    def to_z(self, Y):
        return (Y - self.m) / self.s

    def to_Y(self, z):
        return z * self.s + self.m

    # ---- z -> price paths -----------------------------------------------------
    def to_paths(self, z: np.ndarray, world: str = "Ptheta") -> Paths:
        """
        z: (n, H) standardized returns (numpy). Returns a taskb `Paths` with
        S[:, 0] == S0 exactly and v=None, so the constraint builder accepts it.
        """
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2:
            raise ValueError(f"expected (n, H), got {z.shape}")
        Y = self.to_Y(z)
        logS = np.log(self.S0) + np.cumsum(Y, axis=1)
        S = np.empty((z.shape[0], z.shape[1] + 1), dtype=np.float64)
        S[:, 0] = self.S0
        S[:, 1:] = np.exp(logS)
        return Paths(S=S, v=None, r=self.r, dt=self.dt, world=world, S0=self.S0)

    def state_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, d: dict) -> "PathStandardizer":
        return cls(**d)


# --------------------------------------------------------------------------

def apply_cap(z: np.ndarray, z_cap: float):
    """Outlier rule (DECISIONS.md section 4): keep rows with max_i |z_i| <= z_cap."""
    keep = np.abs(z).max(axis=1) <= z_cap
    return z[keep], int((~keep).sum())


@dataclass
class TrainingSet:
    z: np.ndarray                 # (n_kept, H) float32, standardized, capped
    std: PathStandardizer
    n_rejected: int
    max_abs_z: float              # before capping, for the record
    Y_raw: np.ndarray             # (n, H) float64 uncapped returns (diagnostics only)


def build_training_set(cfg: TaskCConfig = CFG) -> TrainingSet:
    """
    Simulate Heston-P with taskb's simulator on the Task B seed, fit the global
    standardizer on ALL returns (the cap is a generator-pathology rule; it
    rejects zero real paths, see DECISIONS.md), then apply the cap.
    """
    paths = simulate(cfg.heston, cfg.train_sim(), world="P")
    Y = paths.returns()                                   # (n, H) float64
    std = PathStandardizer.fit(Y, S0=cfg.S0, r=cfg.r, dt=cfg.dt)
    z = std.to_z(Y)
    max_abs = float(np.abs(z).max())
    z_kept, n_rej = apply_cap(z, cfg.z_cap)
    return TrainingSet(z=z_kept.astype(np.float32), std=std, n_rejected=n_rej,
                       max_abs_z=max_abs, Y_raw=Y)


def make_loader(z: np.ndarray, batch_size: int, seed: int = 0) -> DataLoader:
    """DataLoader in the (batch,) tuple form `src.train_ddpm.train_ddpm` expects."""
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(np.asarray(z, dtype=np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=False)


def reference_paths(cfg: TaskCConfig = CFG) -> Paths:
    """Independent sample of the prior's own simulator for the gate (variance stripped)."""
    return simulate(cfg.heston, cfg.ref_sim(), world="P").without_variance()
