import math
from typing import Dict, Tuple, Optional, Sequence

import numpy as np
from scipy.stats import norm, kstest
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

    Args:
        S0: Spot price.
        K: Strike.
        T: Time to maturity in years.
        r: Continuously compounded risk free rate.
        sigma: Volatility.
        option_type: "call" or "put".

    Returns:
        Black–Scholes price.
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

    Uses log returns:
        Y_h ~ N(m_Q, v0)
    with
        m_Q = (r - 0.5 sigma^2) dt
        v0  = sigma^2 dt

    Args:
        S0: Spot.
        K: Strike.
        T: Time to maturity (years).
        r: Risk free rate.
        sigma: Volatility.
        n_paths: Number of Monte Carlo paths.
        H_steps: Number of time steps per path.
        option_type: "call" or "put".
        rng: Optional numpy Generator.

    Returns:
        (price, standard_error)
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = T / H_steps
    mQ = (r - 0.5 * sigma ** 2) * dt
    v0 = sigma ** 2 * dt

    # simulate log price
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

    Args:
        returns_Q: Array of shape (n_paths, H_steps) containing log returns.
        S0: Spot price.

    Returns:
        Array of shape (n_paths,) with terminal prices S_T.
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
    Check the martingale condition e^{-rt} E_Q[S_t] ~ S0 at selected horizons.

    Args:
        returns_Q: Array of shape (n_paths, H_steps) of log returns under Q.
        S0: Spot price.
        r: Risk free rate.
        dt: Time step size.
        checkpoints: Collection of time indices (in steps) to test.

    Returns:
        Dict mapping t_index -> discounted expected price.
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
        out[t] = disc * S_t.mean()

    return out


def terminal_ks_test(
    returns_Q: np.ndarray,
    S0: float,
    r: float,
    sigma: float,
    dt: float,
) -> Tuple[float, float]:
    """
    Kolmogorov–Smirnov test of terminal log price vs the GBM N(0,1) reference.

    Args:
        returns_Q: Array of shape (n_paths, H_steps) of log returns under Q.
        S0: Spot price.
        r: Risk free rate.
        sigma: Volatility.
        dt: Time step size.

    Returns:
        (ks_statistic, p_value)
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
    Monte Carlo price from a given panel of DDPM generated log returns.

    Args:
        returns_Q: Array of shape (n_paths, H_steps) of log returns under Q.
        S0: Spot.
        K: Strike.
        r: Risk free rate.
        T_total: Time to maturity in years.
        option_type: "call" or "put".

    Returns:
        (price, standard_error)
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
    return price, stderr


# ---------------------------------------------------------------------
# High level wrapper: run full DDPM pricing experiment
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

    This:
        1. Samples risk-neutral log returns via DDPM + epsilon shift.
        2. Computes empirical mean and std of returns.
        3. Performs martingale checks.
        4. Runs KS test on terminal log price.
        5. Prices the option via DDPM Monte Carlo.
        6. Computes Black–Scholes benchmark.

    Args:
        model: Trained DDPM score network (in standardized space).
        S0: Spot price.
        K: Strike.
        r: Risk free rate.
        sigma: Volatility.
        mu: Physical drift (used only to define dm_std inside sampling).
        H_steps: Number of time steps in horizon.
        n_paths: Number of Monte Carlo paths.
        alphas, alphas_bar, betas: DDPM schedules (on device).
        m_hat_P, s_hat_P: Empirical mean and std of P-space standardized DDPM outputs.
        dt: Time step size in years.
        device: torch.device.
        use_ddim: If True, use DDIM sampler (currently not implemented here).
        n_ddim_steps: Number of DDIM steps if use_ddim is True.
        checkpoints: Time indices (in steps) for martingale checks.
        option_type: "call" or "put".

    Returns:
        Dict with keys:
            "P_m_hat", "P_s_hat",
            "returns_mean", "returns_std",
            "martingale",
            "ks_stat", "ks_pvalue",
            "ddpm_price", "ddpm_stderr",
            "bs_price", "abs_diff".
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
