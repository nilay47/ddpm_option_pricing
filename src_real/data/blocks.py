# src_real/data/blocks.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ReturnBlocks:
    """
    Container for H-step log-return blocks built from a 1D price series.
    """
    blocks: np.ndarray          # shape (N_blocks, H)
    H: int
    dt: float                   # in years, e.g. 1/252
    start_indices: np.ndarray   # shape (N_blocks,), index in returns where block starts

    mean_vec: np.ndarray        # shape (H,)
    std_vec: np.ndarray         # shape (H,)
    global_mean: float
    global_std: float


def log_returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """
    Compute log returns y_t = log(S_t / S_{t-1}) from a price series.

    Args:
        prices: shape (T_prices,)

    Returns:
        returns: shape (T_prices-1,)
    """
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 1:
        raise ValueError(f"prices must be 1D, got shape {prices.shape}")
    if len(prices) < 2:
        raise ValueError("prices must have length >= 2")
    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive")

    return np.log(prices[1:] / prices[:-1])


def build_return_blocks(
    returns: np.ndarray,
    H: int,
    dt: float,
    stride: int = 1,
    drop_last_incomplete: bool = True,
) -> ReturnBlocks:
    """
    Build overlapping (or strided) blocks of log returns.

    Example:
        returns = [y0,y1,...,y_{T-1}]
        H=21 -> blocks[i] = returns[i : i+21]

    Args:
        returns: shape (T,)
        H: block length
        dt: step size in years
        stride: starting index stride (1 = fully overlapping)
        drop_last_incomplete: if True, only keep full blocks

    Returns:
        ReturnBlocks with blocks shape (N_blocks, H)
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 1:
        raise ValueError(f"returns must be 1D, got shape {returns.shape}")
    if H <= 0:
        raise ValueError("H must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if len(returns) < H:
        raise ValueError(f"Need len(returns) >= H. Got {len(returns)} and H={H}")

    if drop_last_incomplete:
        last_start = len(returns) - H
    else:
        last_start = len(returns) - 1  # will pad/truncate later (not implemented)

    starts = np.arange(0, last_start + 1, stride, dtype=int)
    blocks = np.stack([returns[i : i + H] for i in starts], axis=0)  # (N,H)

    # per-dimension stats (recommended for block models)
    mean_vec = blocks.mean(axis=0)
    std_vec = blocks.std(axis=0, ddof=1)
    std_vec = np.where(std_vec < 1e-12, 1.0, std_vec)

    # global stats (sometimes useful for logging)
    global_mean = float(blocks.mean())
    global_std = float(blocks.std(ddof=1))
    if global_std < 1e-12:
        global_std = 1.0

    return ReturnBlocks(
        blocks=blocks.astype(np.float32),
        H=H,
        dt=float(dt),
        start_indices=starts,
        mean_vec=mean_vec.astype(np.float32),
        std_vec=std_vec.astype(np.float32),
        global_mean=global_mean,
        global_std=global_std,
    )


def standardize_blocks_per_dim(rb: ReturnBlocks) -> np.ndarray:
    """
    Standardize blocks using per-dimension mean/std.

    Returns:
        z: shape (N_blocks, H)
    """
    X = rb.blocks.astype(np.float32)
    return (X - rb.mean_vec[None, :]) / rb.std_vec[None, :]


def unstandardize_blocks_per_dim(z: np.ndarray, rb: ReturnBlocks) -> np.ndarray:
    """
    Invert per-dimension standardization.
    """
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != rb.H:
        raise ValueError(f"z must have shape (N,{rb.H}), got {z.shape}")
    return rb.std_vec[None, :] * z + rb.mean_vec[None, :]
