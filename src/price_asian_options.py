# src/price_asian_options.py

import math
from typing import Dict, Tuple, Optional, Sequence

import numpy as np
import torch

from .sample_ddpm import sample_returns_Q_std
from .price_options import (
    martingale_check,
    terminal_ks_test,
)


# ---------------------------------------------------------------------
# Price path construction from DDPM log returns
# ---------------------------------------------------------------------

def simulate_price_paths_from_returns(
    returns_Q: np.ndarray,
    S0: float,
) -> np.ndarray:
    """
    Construct full price paths from a panel of log returns under Q.

    Args:
        returns_Q: Array of shape (n_paths, H_steps) containing log returns.
        S0: Spot price.

    Returns:
        Array of shape (n_paths, H_steps + 1) with
            S_paths[:, 0]   = S0
            S_paths[:, h+1] = S_paths[:, h] * exp(returns_Q[:, h])
    """
    if returns_Q.ndim != 2:
        raise ValueError(f"returns_Q must have shape (n_paths, H_steps), got {returns_Q.shape}")

    n_paths, H_steps = returns_Q.shape
    S_paths = np.empty((n_paths, H_steps + 1), dtype=np.float64)
    S_paths[:, 0] = S0

    for h in range(H_steps):
        S_paths[:, h + 1] = S_paths[:, h] * np.exp(returns_Q[:, h])

    return S_paths


# ---------------------------------------------------------------------
# Asian option pricing from DDPM returns
# ---------------------------------------------------------------------

def ddpm_asian_price_from_returns(
    returns_Q: np.ndarray,
    S0: float,
    K: float,
    r: float,
    dt: float,
    option_type: str = "call",
    include_S0_in_average: bool = False,
) -> Tuple[float, float]:
    """
    Monte Carlo price of a discrete arithmetic Asian option
    from DDPM log returns under the risk neutral measure.

    Payoff uses arithmetic average:
        if include_S0_in_average is False:
            A = (1 / H) * sum_{h=1}^H S_{t_h}
        else:
            A = (1 / (H+1)) * sum_{h=0}^H S_{t_h}

        payoff = max( sign * (A - K), 0 )
        where sign = +1 for call, -1 for put.

    Args:
        returns_Q: Array of shape (n_paths, H_steps) of log returns under Q.
        S0: Spot price.
        K: Strike.
        r: Risk free rate.
        dt: Time step size.
        option_type: "call" or "put".
        include_S0_in_average: Include S0 in the average if True.

    Returns:
        (price, standard_error)
    """
    if returns_Q.ndim != 2:
        raise ValueError(f"returns_Q must have shape (n_paths, H_steps), got {returns_Q.shape}")

    n_paths, H_steps = returns_Q.shape
    T_total = H_steps * dt

    # Build full price paths from log returns
    S_paths = simulate_price_paths_from_returns(returns_Q, S0)  # (n_paths, H_steps+1)

    if include_S0_in_average:
        # average over t_0,...,t_H
        S_avg = S_paths.mean(axis=1)
    else:
        # average over monitoring dates t_1,...,t_H
        S_avg = S_paths[:, 1:].mean(axis=1)

    if option_type == "call":
        payoff = np.maximum(S_avg - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S_avg, 0.0)
    else:
        raise ValueError(f"Unknown option_type: {option_type}")

    disc = math.exp(-r * T_total)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / math.sqrt(n_paths)
    return float(price), float(stderr)


# ---------------------------------------------------------------------
# GBM Monte Carlo benchmark for Asian options
# ---------------------------------------------------------------------

def gbm_mc_asian_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    H_steps: int,
    option_type: str = "call",
    include_S0_in_average: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Monte Carlo price for a discrete arithmetic Asian option under GBM.

    Under Q:
        dS_t / S_t = r dt + sigma dW_t

    We simulate log returns:
        Y_h ~ N(m_Q, v0)
        m_Q = (r - 0.5 * sigma^2) * dt
        v0  = sigma^2 * dt

    Then build paths and compute arithmetic average payoff.

    Args:
        S0: Spot price.
        K: Strike.
        T: Maturity in years.
        r: Risk free rate.
        sigma: Volatility.
        n_paths: Number of Monte Carlo paths.
        H_steps: Number of time steps per path (monitoring dates).
        option_type: "call" or "put".
        include_S0_in_average: Include S0 in the average if True.
        rng: Optional numpy random Generator.

    Returns:
        (price, standard_error)
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = T / H_steps
    mQ = (r - 0.5 * sigma ** 2) * dt
    v0 = sigma ** 2 * dt

    # Simulate log returns
    Y = rng.normal(loc=mQ, scale=math.sqrt(v0), size=(n_paths, H_steps))

    # Build GBM price paths
    S_paths = np.empty((n_paths, H_steps + 1), dtype=np.float64)
    S_paths[:, 0] = S0
    for h in range(H_steps):
        S_paths[:, h + 1] = S_paths[:, h] * np.exp(Y[:, h])

    # Arithmetic average
    if include_S0_in_average:
        S_avg = S_paths.mean(axis=1)
    else:
        S_avg = S_paths[:, 1:].mean(axis=1)

    if option_type == "call":
        payoff = np.maximum(S_avg - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S_avg, 0.0)
    else:
        raise ValueError(f"Unknown option_type: {option_type}")

    disc = math.exp(-r * T)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / math.sqrt(n_paths)
    return float(price), float(stderr)


# ---------------------------------------------------------------------
# High level Asian option experiment using DDPM
# ---------------------------------------------------------------------

def price_asian_with_ddpm(
    model: torch.nn.Module,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    mu: float,
    H_steps: int,
    n_paths: int,
    alphas: torch.Tensor,
    alphas_bar: torch.Tensor,
    betas: torch.Tensor,
    m_hat_P: float,
    s_hat_P: float,
    dt: float,
    device: torch.device,
    use_ddim: bool = False,
    n_ddim_steps: int = 250,
    checkpoints: Sequence[int] = (21, 63, 126, 252),
    option_type: str = "call",
    include_S0_in_average: bool = False,
) -> Dict[str, object]:
    """
    End to end DDPM pricing experiment for an arithmetic Asian option.

    Steps:
        1) Sample RN log returns via DDPM (risk neutral shifted).
        2) Compute summary stats of returns.
        3) Martingale checks at selected horizons.
        4) KS test of terminal log price vs GBM.
        5) Asian price from DDPM paths.
        6) GBM Monte Carlo benchmark for the same Asian payoff.

    Returns:
        Dict with keys:
            "P_m_hat", "P_s_hat",
            "returns_mean", "returns_std",
            "martingale",
            "ks_stat", "ks_pvalue",
            "ddpm_asian_price", "ddpm_asian_stderr",
            "gbm_asian_price", "gbm_asian_stderr",
            "abs_diff".
    """
    T_total = H_steps * dt

    # 1. Sample RN log returns via DDPM
    returns_Q = sample_returns_Q_std(
        model=model,
        n_paths=n_paths,
        H_steps=H_steps,
        mu=mu,
        r=r,
        sigma=sigma,
        dt=dt,
        alphas=alphas,
        alphas_bar=alphas_bar,
        betas=betas,
        m_hat=m_hat_P,
        s_hat=s_hat_P,
        s0=math.sqrt(sigma ** 2 * dt),  # daily log return std
        T=alphas_bar.shape[0],
        device=device,
        use_ddim=use_ddim,
        n_ddim_steps=n_ddim_steps,
    )

    # 2. Summary stats of returns
    returns_mean = float(returns_Q.mean())
    returns_std = float(returns_Q.std(ddof=1))

    # 3. Martingale checks
    martingale_vals = martingale_check(
        returns_Q=returns_Q,
        S0=S0,
        r=r,
        dt=dt,
        checkpoints=checkpoints,
    )

    # 4. Terminal KS test vs GBM N(0,1) reference
    ks_stat, ks_pvalue = terminal_ks_test(
        returns_Q=returns_Q,
        S0=S0,
        r=r,
        sigma=sigma,
        dt=dt,
    )

    # 5. Asian option price via DDPM returns
    ddpm_price, ddpm_stderr = ddpm_asian_price_from_returns(
        returns_Q=returns_Q,
        S0=S0,
        K=K,
        r=r,
        dt=dt,
        option_type=option_type,
        include_S0_in_average=include_S0_in_average,
    )

    # 6. GBM Monte Carlo benchmark for the same Asian payoff
    gbm_price, gbm_stderr = gbm_mc_asian_price(
        S0=S0,
        K=K,
        T=T_total,
        r=r,
        sigma=sigma,
        n_paths=n_paths,
        H_steps=H_steps,
        option_type=option_type,
        include_S0_in_average=include_S0_in_average,
    )

    return {
        "P_m_hat": float(m_hat_P),
        "P_s_hat": float(s_hat_P),
        "returns_mean": returns_mean,
        "returns_std": returns_std,
        "martingale": martingale_vals,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "ddpm_asian_price": float(ddpm_price),
        "ddpm_asian_stderr": float(ddpm_stderr),
        "gbm_asian_price": float(gbm_price),
        "gbm_asian_stderr": float(gbm_stderr),
        "abs_diff": float(abs(ddpm_price - gbm_price)),
    }