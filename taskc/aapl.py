"""
Real-data experiment on the frozen AAPL surface (DECISIONS.md section 26).

The surface is an artifact, not an output: `yfinance.Ticker.option_chain` returns
whatever expiries are live when it is called, so `data/aapl/aapl_option_chain_clean.csv`
cannot be regenerated. The return series IS reproducible, from
`src_real/data/prices_yf.py` on branch v2_aapl with pinned dates.

Conventions, all fixed in section 26 before anything ran:

  * H = 21 TRADING days; quoted dte is CALENDAR days. step = round(dte * 252/365).
  * Only the carry b = ln(F/S0)/T is identified at 7-25 dte, from the put-call-parity
    forward. The discount rate r is pinned at 4 % and used ONLY to discount payoffs;
    at 25 dte it is a common factor of 0.9973 and cannot affect arm comparisons.
  * Martingale columns are built on a `Paths` whose rate is the CARRY, so e^{-bt} S_t
    is the martingale. Vanillas and exotics are discounted explicitly at r.
"""

from dataclasses import dataclass, replace
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

import taskc  # noqa: F401  (sys.path setup for taskb)
from config import Heston, CALIB_TESTFUNS          # taskb
from heston import Paths, simulate                  # taskb
from constraints import build_martingale_columns, ConstraintSet   # taskb
import evaluation as ev                             # taskb

TRADING_PER_CAL = 252.0 / 365.0
R_PINNED = 0.04            # section 26.2: assumption, discounting only
MAX_REL_SPREAD = 0.10      # section 26.3
CALIB_DTE = 25             # 2026-05-15


# --------------------------------------------------------------------------
# the surface
# --------------------------------------------------------------------------

def load_surface(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["expiry"] = pd.to_datetime(d["expiry"])
    d["step"] = (d["dte"] * TRADING_PER_CAL).round().astype(int)
    return d


def spot_of(d: pd.DataFrame) -> float:
    s = d["spot"].unique()
    assert len(s) == 1, f"surface carries {len(s)} spots; expected one snapshot"
    return float(s[0])


def calls_at(d: pd.DataFrame, dte: int, max_rel_spread: float = MAX_REL_SPREAD) -> pd.DataFrame:
    """Section 26.3 filter: calls, two-sided quote, relative spread within bound."""
    g = d[(d["dte"] == dte) & (d["option_type"] == "call")].copy()
    g = g[(g["bid"] > 0) & (g["ask"] > 0)]
    g["rel_spread"] = (g["ask"] - g["bid"]) / g["mid"]
    return g[g["rel_spread"] <= max_rel_spread].sort_values("strike").reset_index(drop=True)


def parity_forward(d: pd.DataFrame, dte: int) -> Tuple[float, float, int]:
    """
    C - P = e^{-rT}(F - K) regressed on K over matched strikes.
    Returns (implied_r, F, n_strikes). Only F is trustworthy at these maturities.
    """
    g = d[(d["dte"] == dte) & (d["bid"] > 0) & (d["ask"] > 0)]
    c = g[g["option_type"] == "call"].set_index("strike")["mid"]
    p = g[g["option_type"] == "put"].set_index("strike")["mid"]
    K = np.array(sorted(set(c.index) & set(p.index)), dtype=float)
    if len(K) < 4:
        raise ValueError(f"dte={dte}: only {len(K)} parity-matched strikes")
    y = (c.loc[K] - p.loc[K]).to_numpy()
    A = np.vstack([np.ones_like(K), -K]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    T = dte / 365.0
    return float(-np.log(b) / T), float(a / b), len(K)


def carry_from_forward(F: float, S0: float, dte: int) -> float:
    """b = ln(F/S0)/T, the only rate quantity the short-dated surface identifies."""
    return float(np.log(F / S0) / (dte / 365.0))


# --------------------------------------------------------------------------
# constraints
# --------------------------------------------------------------------------

def paths_from_returns(Y: np.ndarray, S0: float, carry: float, dt: float = 1 / 252.0) -> Paths:
    """(n, H) log-returns -> Paths whose rate is the CARRY, so e^{-bt}S_t is a martingale."""
    Y = np.asarray(Y, dtype=np.float64)
    S = np.empty((Y.shape[0], Y.shape[1] + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(np.cumsum(Y, axis=1))
    return Paths(S=S, v=None, r=carry, dt=dt, world="aapl", S0=S0)


def vanilla_columns(paths: Paths, calls: pd.DataFrame, r: float = R_PINNED):
    """Discounted call payoffs at the quoted step, with market mids as targets."""
    cols, names, tgt, spread = [], [], [], []
    for _, row in calls.iterrows():
        s = int(row["step"]); K = float(row["strike"])
        T = s * paths.dt
        cols.append(np.exp(-r * T) * np.maximum(paths.S[:, s] - K, 0.0))
        names.append(f"call[dte={int(row['dte'])}d,step={s},K={K:g}]")
        tgt.append(float(row["mid"]))
        spread.append(float(row["ask"] - row["bid"]))
    return cols, names, np.asarray(tgt), np.asarray(spread)


def build_aapl(paths: Paths, calls: pd.DataFrame, r: float = R_PINNED,
               testfuns: Sequence = CALIB_TESTFUNS) -> ConstraintSet:
    """C0 martingale families (on the carry-discounted price) + calibrated vanillas."""
    mcols, mnames = build_martingale_columns(paths, testfuns)
    vcols, vnames, vtgt, _ = vanilla_columns(paths, calls, r)
    G = np.column_stack(list(mcols) + list(vcols))
    c = np.concatenate([np.zeros(len(mcols)), vtgt])
    return ConstraintSet(G=G, c=c, names=list(mnames) + vnames,
                         kinds=["mart"] * len(mcols) + ["vanilla"] * len(vcols))


def heldout_errors(paths: Paths, calls: pd.DataFrame, w: np.ndarray = None,
                   r: float = R_PINNED) -> pd.DataFrame:
    """Model price vs market mid, in bid-ask spread units (section 26.5)."""
    cols, names, tgt, spread = vanilla_columns(paths, calls, r)
    out = []
    for col, nm, t, sp in zip(cols, names, tgt, spread):
        px = float(np.average(col, weights=w)) if w is not None else float(col.mean())
        out.append(dict(name=nm, dte=int(nm.split("dte=")[1].split("d")[0]),
                        strike=float(nm.split("K=")[1].rstrip("]")),
                        model=px, mid=t, spread=sp,
                        err=px - t, err_spreads=(px - t) / sp if sp > 0 else np.nan))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# the parametric arms
# --------------------------------------------------------------------------

def fit_heston_mom(rets: np.ndarray, dt: float = 1 / 252.0, block: int = 21) -> Heston:
    """
    Method of moments on non-overlapping realized variance (section 26.4 arm 2).

      theta   = mean(RV)
      kappa   = -log(corr(RV_k, RV_{k+1})) / (block*dt)
      xi      = sqrt(2 kappa Var(RV) / theta)      [CIR stationary variance]
      rho     = corr(block return, RV change)
    """
    r = np.asarray(rets, dtype=np.float64)
    n = (len(r) // block) * block
    R = r[:n].reshape(-1, block)
    RV = (R ** 2).sum(axis=1) / (block * dt)          # annualized variance per block
    theta = float(RV.mean())
    a1 = float(np.corrcoef(RV[:-1], RV[1:])[0, 1])
    a1 = min(max(a1, 1e-3), 0.999)
    kappa = float(-np.log(a1) / (block * dt))
    # RV of `block` daily squares carries sampling noise ~ 2 theta^2 / block on top of
    # the true variance of v; subtracting it stops xi being inflated by estimation error.
    var_rv = max(RV.var(ddof=1) - 2.0 * theta ** 2 / block, 1e-12)
    xi = float(np.sqrt(max(2.0 * kappa * var_rv / max(theta, 1e-12), 1e-12)))
    # Leverage from CONTEMPORANEOUS block return vs block RV. Daily proxies
    # (corr(r_t, r_{t+1}^2) etc.) are attenuated to -0.05..-0.08 because r^2 is a
    # chi^2_1-noisy estimate of v; averaging 21 days removes most of that noise.
    rho = float(np.corrcoef(R.sum(axis=1), RV)[0, 1])
    rho = min(max(rho, -0.95), -0.01)
    return Heston(S0=100.0, v0=theta, drift=float(r.mean() / dt), r=0.0,
                  kappa=kappa, theta=theta, xi=xi, rho=rho)


def sample_heston_returns(h: Heston, n: int, H: int, seed: int, dt: float = 1 / 252.0) -> np.ndarray:
    from config import SimConfig
    p = simulate(h, SimConfig(n_paths=n, H=H, dt=dt, seed=seed), world="P")
    return np.diff(np.log(p.S), axis=1)


def fit_student_t(rets: np.ndarray):
    from scipy import stats
    df, loc, scale = stats.t.fit(np.asarray(rets, dtype=np.float64))
    return dict(df=float(df), loc=float(loc), scale=float(scale))


def sample_student_t(par: dict, n: int, H: int, seed: int) -> np.ndarray:
    from scipy import stats
    return stats.t.rvs(par["df"], loc=par["loc"], scale=par["scale"],
                       size=(n, H), random_state=np.random.default_rng(seed))


def fit_gaussian(rets: np.ndarray) -> dict:
    r = np.asarray(rets, dtype=np.float64)
    return dict(loc=float(r.mean()), scale=float(r.std(ddof=1)))


def sample_gaussian(par: dict, n: int, H: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return par["loc"] + par["scale"] * rng.standard_normal((n, H))


# --------------------------------------------------------------------------
# exotics
# --------------------------------------------------------------------------

def aapl_exotics(S0: float) -> Dict[str, dict]:
    """Section 26.5: struck off the as-of spot, so they are comparable across arms."""
    return {"asian_call_atm":        dict(kind="asian", K=S0),
            "uo_barrier_call_atm_110": dict(kind="up_and_out", K=S0, B=1.10 * S0),
            "lookback_float_call":   dict(kind="lookback_float")}


def price_exotics(paths: Paths, S0: float, w: np.ndarray = None, r: float = R_PINNED) -> dict:
    """Exotics discounted explicitly at the pinned r, not at the path's carry."""
    T = paths.H * paths.dt
    carry_disc = np.exp(-paths.r * T)            # what exotic_payoff already applied
    out = {}
    for k, sp in aapl_exotics(S0).items():
        pay = ev.exotic_payoff(paths, **sp) / carry_disc * np.exp(-r * T)
        if w is None:
            m = float(pay.mean()); se = float(pay.std(ddof=1) / np.sqrt(len(pay)))
        else:
            m = float(w @ pay); se = float(np.sqrt(np.sum(w ** 2 * (pay - m) ** 2)))
        out[k] = dict(price=m, se=se)
    return out


# --------------------------------------------------------------------------
# interval-constrained entropy projection (DECISIONS.md section 26.8)
# --------------------------------------------------------------------------

def solve_interval(G: np.ndarray, lo: np.ndarray, hi: np.ndarray, maxiter: int = 2000):
    """
    min KL(Q||P)  s.t.  lo_j <= E_Q[g_j] <= hi_j,  Q << P_N.

    Market quotes are intervals, not points: forcing E_Q[payoff] to the mid asserts
    a precision the bid-ask spread denies, and on this surface it is not even
    attainable -- reweighting a lognormal sample cannot hit 21 smiled strikes exactly.

    Dual (lam for the upper bounds, mu for the lower, both >= 0, beta = mu - lam):

        min_{lam, mu >= 0}  log E_P[e^{(mu-lam).g}] + lam.hi - mu.lo

    Equality is the special case lo = hi. Returns (w, beta, kl, ess, info).
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    n, m = G.shape
    logn = np.log(n)
    Gm, Gs = G.mean(0), G.std(0)
    Gs = np.where(Gs > 1e-12, Gs, 1.0)
    Z = (G - Gm) / Gs                                   # standardise, as every other solve does
    lo_s, hi_s = (lo - Gm) / Gs, (hi - Gm) / Gs

    def obj(x):
        lam, mu = x[:m], x[m:]
        beta = mu - lam
        z = Z @ beta
        lse = logsumexp(z) - logn
        w = np.exp(z - z.max()); w /= w.sum()
        g = Z.T @ w
        f = lse + lam @ hi_s - mu @ lo_s
        return f, np.concatenate([-g + hi_s, g - lo_s])

    x0 = np.zeros(2 * m)
    r = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * (2 * m),
                 options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=1e-14, gtol=1e-10, maxcor=50))
    beta_s = r.x[m:] - r.x[:m]
    z = Z @ beta_s
    lse = float(logsumexp(z) - logn)
    w = np.exp(z - z.max()); w /= w.sum()
    kl = float(w @ z - lse)
    ess = float(1.0 / np.sum(w ** 2) / n)
    fit = G.T @ w
    viol = np.maximum(np.maximum(lo - fit, fit - hi), 0.0)
    return w, beta_s / Gs, kl, ess, dict(
        max_violation=float(viol.max()), n_violated=int((viol > 1e-8).sum()),
        n_at_lower=int(np.sum(fit <= lo + 1e-9)), n_at_upper=int(np.sum(fit >= hi - 1e-9)),
        nit=int(r.nit), success=bool(r.success), fitted=fit)
