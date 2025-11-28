# src/experiments_ddpm.py

import math
from typing import Dict, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, kstest
from scipy.optimize import brentq
import torch

from .sample_ddpm import sample_returns_Q_std


# ---------------------------------------------------------------------
# Black–Scholes analytic pricing
# ---------------------------------------------------------------------

def black_scholes_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Black–Scholes price for a European call or put on a non-dividend-paying stock.
    """
    if T <= 0.0:
        if option_type == "call":
            return max(S0 - K, 0.0)
        elif option_type == "put":
            return max(K - S0, 0.0)
        else:
            raise ValueError(f"Unknown option_type: {option_type}")

    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "call":
        return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option_type: {option_type}")


# ---------------------------------------------------------------------
# GBM Monte Carlo under Q (baseline)
# ---------------------------------------------------------------------

def gbm_mc_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    H_steps: int,
    option_type: str = "call",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Monte Carlo price for a European option using exact GBM under Q.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = T / H_steps
    mQ = (r - 0.5 * sigma ** 2) * dt
    v0 = sigma ** 2 * dt

    Y = rng.normal(loc=mQ, scale=math.sqrt(v0), size=(n_paths, H_steps))
    S = np.full((n_paths,), S0, dtype=np.float64)
    for h in range(H_steps):
        S *= np.exp(Y[:, h])

    disc = math.exp(-r * T)
    if option_type == "call":
        payoff = np.maximum(S - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S, 0.0)
    else:
        raise ValueError(f"Unknown option_type: {option_type}")

    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / math.sqrt(n_paths)
    return price, stderr


# ---------------------------------------------------------------------
# Helpers that operate on DDPM generated returns
# ---------------------------------------------------------------------

def simulate_terminal_prices_from_returns(
    returns_Q: np.ndarray,
    S0: float,
) -> np.ndarray:
    """
    Construct terminal prices from a panel of log returns under Q.
    """
    if returns_Q.ndim != 2:
        raise ValueError(f"returns_Q must have shape (n_paths, H_steps), got {returns_Q.shape}")

    n_paths, H_steps = returns_Q.shape
    S = np.full((n_paths,), S0, dtype=np.float64)
    for h in range(H_steps):
        S *= np.exp(returns_Q[:, h])
    return S


def martingale_check(
    returns_Q: np.ndarray,
    S0: float,
    r: float,
    dt: float,
    checkpoints: Sequence[int] = (21, 63, 126, 252),
) -> Dict[int, float]:
    """
    Check e^{-rt} E_Q[S_t] ~ S0 at selected horizons.
    """
    if returns_Q.ndim != 2:
        raise ValueError(f"returns_Q must have shape (n_paths, H_steps), got {returns_Q.shape}")

    n_paths, H_steps = returns_Q.shape
    out: Dict[int, float] = {}

    for t in checkpoints:
        if t <= 0 or t > H_steps:
            continue
        S_t = np.full((n_paths,), S0, dtype=np.float64)
        for h in range(t):
            S_t *= np.exp(returns_Q[:, h])
        disc = math.exp(-r * (t * dt))
        out[t] = float(disc * S_t.mean())

    return out

def martingale_trajectory(
    returns_Q: np.ndarray,
    S0: float,
    r: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full martingale trajectory:
        M_t = e^{-rt} E_Q[S_t]
    for t = 0,1,...,H_steps.

    Args:
        returns_Q: (n_paths, H_steps) log-returns.
        S0: spot.
        r: risk free rate.
        dt: time step.

    Returns:
        times: shape (H_steps+1,), in years.
        M:     shape (H_steps+1,), discounted expected prices.
    """
    if returns_Q.ndim != 2:
        raise ValueError(f"returns_Q must have shape (n_paths, H_steps), got {returns_Q.shape}")

    n_paths, H_steps = returns_Q.shape
    S = np.full((n_paths,), S0, dtype=np.float64)

    times = [0.0]
    M = [S0]  # at t = 0, e^{-0} E[S0] = S0

    for t in range(1, H_steps + 1):
        S *= np.exp(returns_Q[:, t - 1])
        tau = t * dt
        disc = math.exp(-r * tau)
        M.append(disc * S.mean())
        times.append(tau)

    return np.array(times), np.array(M)


def terminal_ks_test(
    returns_Q: np.ndarray,
    S0: float,
    r: float,
    sigma: float,
    dt: float,
) -> Tuple[float, float]:
    """
    Kolmogorov–Smirnov test of terminal log price against the GBM-based Gaussian.
    """
    if returns_Q.ndim != 2:
        raise ValueError(f"returns_Q must have shape (n_paths, H_steps), got {returns_Q.shape}")

    n_paths, H_steps = returns_Q.shape
    T_total = H_steps * dt

    S_T = simulate_terminal_prices_from_returns(returns_Q, S0)

    mean_theory = math.log(S0) + (r - 0.5 * sigma ** 2) * T_total
    var_theory = sigma ** 2 * T_total

    z = (np.log(S_T) - mean_theory) / math.sqrt(var_theory)
    ks_res = kstest(z, "norm")
    return float(ks_res.statistic), float(ks_res.pvalue)


def ddpm_price_from_returns(
    returns_Q: np.ndarray,
    S0: float,
    K: float,
    r: float,
    T_total: float,
    option_type: str = "call",
) -> Tuple[float, float]:
    """
    Monte Carlo price from a given panel of DDPM log returns.
    """
    S_T = simulate_terminal_prices_from_returns(returns_Q, S0)
    disc = math.exp(-r * T_total)

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(f"Unknown option_type: {option_type}")

    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / math.sqrt(len(S_T))
    return float(price), float(stderr)


# ---------------------------------------------------------------------
# Single-strike full DDPM experiment
# ---------------------------------------------------------------------

def price_vanilla_with_ddpm(
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
) -> Dict[str, object]:
    """
    End to end DDPM pricing experiment for a single European option.
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
        s0=math.sqrt(sigma ** 2 * dt),  # daily log-return std
        T=alphas_bar.shape[0],
        device=device,
        use_ddim=use_ddim,
        n_ddim_steps=n_ddim_steps,
    )

    # 2. Moment summary of returns
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

    # 4. KS test
    ks_stat, ks_pvalue = terminal_ks_test(
        returns_Q=returns_Q,
        S0=S0,
        r=r,
        sigma=sigma,
        dt=dt,
    )

    # 5. DDPM option price
    ddpm_price, ddpm_stderr = ddpm_price_from_returns(
        returns_Q=returns_Q,
        S0=S0,
        K=K,
        r=r,
        T_total=T_total,
        option_type=option_type,
    )

    # 6. Black–Scholes benchmark
    bs_price = black_scholes_price(
        S0=S0,
        K=K,
        T=T_total,
        r=r,
        sigma=sigma,
        option_type=option_type,
    )

    return {
        "P_m_hat": float(m_hat_P),
        "P_s_hat": float(s_hat_P),
        "returns_mean": returns_mean,
        "returns_std": returns_std,
        "martingale": martingale_vals,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "ddpm_price": ddpm_price,
        "ddpm_stderr": ddpm_stderr,
        "bs_price": bs_price,
        "abs_diff": float(abs(ddpm_price - bs_price)),
    }


# ---------------------------------------------------------------------
# Implied vol + multi-strike and ablation
# ---------------------------------------------------------------------

def implied_vol_call(
    price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    vol_lower: float = 1e-6,
    vol_upper: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Implied volatility from call price using Black–Scholes and Brent.
    """
    intrinsic = max(S0 - K * math.exp(-r * T), 0.0)
    if price < intrinsic or T <= 0.0:
        return float("nan")

    def f(sig: float) -> float:
        return black_scholes_price(S0, K, T, r, sig, option_type="call") - price

    try:
        lo, hi = vol_lower, vol_upper
        f_lo, f_hi = f(lo), f(hi)
        if math.copysign(1.0, f_lo) == math.copysign(1.0, f_hi):
            for factor in [2.0, 4.0, 8.0]:
                hi_try = vol_upper * factor
                f_hi = f(hi_try)
                if math.copysign(1.0, f_lo) != math.copysign(1.0, f_hi):
                    hi = hi_try
                    break
            else:
                return float("nan")
        return float(brentq(f, lo, hi, xtol=tol, maxiter=max_iter))
    except Exception:
        return float("nan")


def ddpm_multistrike_with_iv(
    model: torch.nn.Module,
    S0: float,
    strikes: Sequence[float],
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
    option_type: str = "call",
    use_ddim: bool = False,
    n_ddim_steps: int = 250,
) -> pd.DataFrame:
    """
    Multi-strike DDPM vs GBM vs BS experiment with implied vols.
    """
    T_total = H_steps * dt
    strikes = np.asarray(strikes, dtype=float)

    # RN returns (one panel reused for all strikes)
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
        s0=math.sqrt(sigma ** 2 * dt),
        T=alphas_bar.shape[0],
        device=device,
        use_ddim=use_ddim,
        n_ddim_steps=n_ddim_steps,
    )

    ddpm_prices = []
    ddpm_stderrs = []
    gbm_prices = []
    gbm_stderrs = []
    bs_prices = []

    for K in strikes:
        # DDPM price
        p_ddpm, se_ddpm = ddpm_price_from_returns(
            returns_Q=returns_Q,
            S0=S0,
            K=K,
            r=r,
            T_total=T_total,
            option_type=option_type,
        )
        ddpm_prices.append(p_ddpm)
        ddpm_stderrs.append(se_ddpm)

        # GBM MC price
        p_gbm, se_gbm = gbm_mc_price(
            S0=S0,
            K=K,
            T=T_total,
            r=r,
            sigma=sigma,
            n_paths=n_paths,
            H_steps=H_steps,
            option_type=option_type,
        )
        gbm_prices.append(p_gbm)
        gbm_stderrs.append(se_gbm)

        # BS analytic
        p_bs = black_scholes_price(
            S0=S0,
            K=K,
            T=T_total,
            r=r,
            sigma=sigma,
            option_type=option_type,
        )
        bs_prices.append(p_bs)

    df = pd.DataFrame(
        {
            "strike": strikes,
            "ddpm_price": ddpm_prices,
            "ddpm_stderr": ddpm_stderrs,
            "gbm_price": gbm_prices,
            "gbm_stderr": gbm_stderrs,
            "bs_price": bs_prices,
        }
    )

    # implied vols
    iv_ddpm = []
    iv_gbm = []
    iv_bs = []

    for _, row in df.iterrows():
        K = float(row["strike"])
        iv_ddpm.append(implied_vol_call(row["ddpm_price"], S0=S0, K=K, T=T_total, r=r))
        iv_gbm.append(implied_vol_call(row["gbm_price"], S0=S0, K=K, T=T_total, r=r))
        iv_bs.append(implied_vol_call(row["bs_price"], S0=S0, K=K, T=T_total, r=r))

    df["iv_ddpm"] = iv_ddpm
    df["iv_gbm"] = iv_gbm
    df["iv_bs"] = iv_bs
    return df


def ddpm_multistrike_ablation_shift(
    model: torch.nn.Module,
    S0: float,
    strikes: Sequence[float],
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
    option_type: str = "call",
    use_ddim: bool = False,
    n_ddim_steps: int = 250,
) -> pd.DataFrame:
    """
    Compare DDPM prices with epsilon shift (RN) vs without (physical).
    """
    T_total = H_steps * dt
    strikes = np.asarray(strikes, dtype=float)

    # RN returns (shift ON)
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
        s0=math.sqrt(sigma ** 2 * dt),
        T=alphas_bar.shape[0],
        device=device,
        use_ddim=use_ddim,
        n_ddim_steps=n_ddim_steps,
    )

    # Physical returns (shift OFF) by setting r = mu
    returns_P = sample_returns_Q_std(
        model=model,
        n_paths=n_paths,
        H_steps=H_steps,
        mu=mu,
        r=mu,
        sigma=sigma,
        dt=dt,
        alphas=alphas,
        alphas_bar=alphas_bar,
        betas=betas,
        m_hat=m_hat_P,
        s_hat=s_hat_P,
        s0=math.sqrt(sigma ** 2 * dt),
        T=alphas_bar.shape[0],
        device=device,
        use_ddim=use_ddim,
        n_ddim_steps=n_ddim_steps,
    )

    rows = []
    for K in strikes:
        p_shift, se_shift = ddpm_price_from_returns(
            returns_Q=returns_Q,
            S0=S0,
            K=K,
            r=r,
            T_total=T_total,
            option_type=option_type,
        )

        p_noshift, se_noshift = ddpm_price_from_returns(
            returns_Q=returns_P,
            S0=S0,
            K=K,
            r=r,
            T_total=T_total,
            option_type=option_type,
        )

        rows.append(
            {
                "strike": K,
                "price_shift": p_shift,
                "stderr_shift": se_shift,
                "price_noshift": p_noshift,
                "stderr_noshift": se_noshift,
            }
        )

    return pd.DataFrame(rows)
