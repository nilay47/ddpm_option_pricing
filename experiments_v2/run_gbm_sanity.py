import json, math, os
import numpy as np
import torch

from src.price_options import black_scholes_price, ddpm_price_from_returns
from src.experiments import martingale_trajectory

from src_v2.finance.sample_returns_shifted import sample_returns_Q_epsilon_shift


def run_gbm_sanity(
    model,
    S0: float,
    mu: float,
    r: float,
    sigma: float,
    dt: float,
    H_steps: int,
    Ks,
    n_paths: int,
    alphas,
    alphas_bar,
    betas,
    device,
    apply_mean_projection: bool = False,
    out_dir: str = "artifacts_v2/gbm_sanity",
):
    os.makedirs(out_dir, exist_ok=True)

    T_total = H_steps * dt

    returns_Q = sample_returns_Q_epsilon_shift(
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
        s0=sigma * math.sqrt(dt),
        device=device,
        apply_mean_projection=apply_mean_projection,
    )

    # diagnostics
    times, M = martingale_trajectory(returns_Q, S0=S0, r=r, dt=dt)

    results = {
        "n_paths": int(n_paths),
        "H_steps": int(H_steps),
        "dt": float(dt),
        "T_total": float(T_total),
        "apply_mean_projection": bool(apply_mean_projection),
        "returns_mean": float(returns_Q.mean()),
        "returns_std": float(returns_Q.std(ddof=1)),
        "martingale_M_end": float(M[-1]),
        "martingale_max_abs_dev": float(np.max(np.abs(M - S0))),
        "strikes": [],
    }

    rows = []
    for K in Ks:
        p_ddpm, se_ddpm = ddpm_price_from_returns(
            returns_Q=returns_Q,
            S0=S0,
            K=float(K),
            r=r,
            T_total=T_total,
            option_type="call",
        )
        p_bs = black_scholes_price(S0=S0, K=float(K), T=T_total, r=r, sigma=sigma, option_type="call")

        row = {
            "K": float(K),
            "ddpm_price": float(p_ddpm),
            "ddpm_stderr": float(se_ddpm),
            "bs_price": float(p_bs),
            "diff": float(p_ddpm - p_bs),
        }
        rows.append(row)

    results["strikes"] = rows

    # save artifacts
    np.savez(os.path.join(out_dir, "returns_and_martingale.npz"), returns_Q=returns_Q, times=times, M=M)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results
