# src_real/finance/pricing_smile.py

from __future__ import annotations

import math
from typing import Sequence, Tuple, Dict

import numpy as np
import pandas as pd

# reuse your existing, battle-tested functions
from src.experiments import black_scholes_price, implied_vol_call


def terminal_prices_from_return_blocks(
    return_blocks: np.ndarray,
    S0: float,
) -> np.ndarray:
    """
    return_blocks: (N,H) log returns
    S_T = S0 * exp(sum_h returns_h)
    """
    X = np.asarray(return_blocks, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"return_blocks must be 2D, got {X.shape}")
    return S0 * np.exp(X.sum(axis=1))


def price_calls_from_ST(
    ST: np.ndarray,
    Ks: Sequence[float],
    r: float,
    T: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo call prices and standard errors from terminal prices.
    """
    ST = np.asarray(ST, dtype=np.float64)
    Ks = np.asarray(Ks, dtype=np.float64)
    disc = math.exp(-r * T)

    prices = []
    stderrs = []
    n = len(ST)

    for K in Ks:
        payoff = np.maximum(ST - float(K), 0.0)
        prices.append(disc * payoff.mean())
        stderrs.append(disc * payoff.std(ddof=1) / math.sqrt(n))

    return np.array(prices, dtype=float), np.array(stderrs, dtype=float)


def smile_table_from_prices(
    S0: float,
    Ks: Sequence[float],
    prices: np.ndarray,
    stderrs: np.ndarray,
    r: float,
    T: float,
) -> pd.DataFrame:
    """
    Build a DataFrame with call prices and implied vols (DDPM or GBM outputs).
    """
    Ks = np.asarray(Ks, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)
    stderrs = np.asarray(stderrs, dtype=np.float64)

    ivs = []
    for K, p in zip(Ks, prices):
        ivs.append(implied_vol_call(float(p), S0=float(S0), K=float(K), T=float(T), r=float(r)))

    df = pd.DataFrame(
        {
            "K": Ks,
            "K_over_S0": Ks / float(S0),
            "price": prices,
            "stderr": stderrs,
            "iv": np.array(ivs, dtype=float),
        }
    )
    return df


def bs_smile_table(
    S0: float,
    Ks: Sequence[float],
    r: float,
    T: float,
    sigma: float,
) -> pd.DataFrame:
    """
    Black-Scholes prices and implied vols (should return ~sigma everywhere).
    """
    Ks = np.asarray(Ks, dtype=np.float64)
    prices = np.array([black_scholes_price(S0, float(K), T, r, sigma, option_type="call") for K in Ks], dtype=float)
    stderrs = np.zeros_like(prices)
    return smile_table_from_prices(S0, Ks, prices, stderrs, r, T)