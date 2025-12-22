# src_real/experiments/aapl_smile_21d.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src_real.schedules import make_alpha_schedule
from src_real.data.blocks import (
    log_returns_from_prices,
    build_return_blocks,
    standardize_blocks_per_dim,
    unstandardize_blocks_per_dim,
)
from src_real.models.score_mlp_vec import ScoreMLPVec
from src_real.diffusion.train import train_ddpm_blocks
from src_real.diffusion.sample import sample_blocks_ddpm
from src_real.finance.pricing_smile import (
    terminal_prices_from_return_blocks,
    price_calls_from_ST,
    smile_table_from_prices,
    bs_smile_table,
)


@dataclass(frozen=True)
class AAPLSmileConfig:
    H: int = 21
    days_per_year: int = 252
    T_diffusion: int = 1000

    # training
    batch_size: int = 512
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    time_emb_dim: int = 64

    # sampling/pricing
    n_mc: int = 50_000
    Ks_over_S0: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)

    # finance params (first pass)
    r: float = 0.04
    sigma_anchor: float = 0.20   # only for BS baseline + optional RN mean shift

    # reproducibility
    seed: int = 42


def run_smile_from_prices(
    prices: np.ndarray,
    S0: Optional[float],
    cfg: AAPLSmileConfig,
    device: torch.device,
    save_ckpt_path: Optional[str] = None,
    load_ckpt_path: Optional[str] = None,
) -> dict:
    """
    End-to-end experiment on a provided price series.

    This version does NOT fetch market data. It:
      - trains DDPM on real return blocks (P)
      - samples blocks (P)
      - (optional) crude RN mean shift using sigma_anchor
      - prices calls across strikes and produces IV smile table
      - produces BS smile table (flat) for comparison

    Args:
        prices: 1D array of prices (daily close)
        S0: spot to use for pricing; if None, uses last price
        cfg: configuration
        device: torch.device
        save_ckpt_path: optional path to save model + stats
        load_ckpt_path: optional path to load model + stats

    Returns:
        dict with:
            "df_ddpm_P", "df_ddpm_Q", "df_bs",
            "train_stats", "rb_stats", "S0", "T_total"
    """
    # seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 1 or len(prices) < cfg.H + 2:
        raise ValueError("prices must be 1D and long enough to form return blocks")

    if S0 is None:
        S0 = float(prices[-1])
    else:
        S0 = float(S0)

    dt = 1.0 / cfg.days_per_year
    T_total = cfg.H * dt

    # 1) returns + blocks
    rets = log_returns_from_prices(prices)  # (T_prices-1,)
    rb = build_return_blocks(returns=rets, H=cfg.H, dt=dt, stride=1)

    Z_train = standardize_blocks_per_dim(rb)  # (N,H)
    x_train = torch.from_numpy(Z_train).float()

    # 2) schedules
    betas, alphas, alphas_bar = make_alpha_schedule(
        T=cfg.T_diffusion,
        device=device,
        schedule="linear",
        beta_max=2e-2,
    )

    # 3) model
    model = ScoreMLPVec(H=cfg.H, hidden_dim=cfg.hidden_dim, time_emb_dim=cfg.time_emb_dim).to(device)

    # 4) load or train
    train_stats = None
    if load_ckpt_path is not None:
        ckpt = torch.load(load_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        # keep rb stats from current run (you may choose to load them too)
    else:
        loader = DataLoader(TensorDataset(x_train), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        train_stats = train_ddpm_blocks(
            model=model,
            train_loader=loader,
            alphas_bar=alphas_bar,
            T=cfg.T_diffusion,
            device=device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            grad_clip=1.0,
        )

        if save_ckpt_path is not None:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "rb_mean_vec": rb.mean_vec,
                    "rb_std_vec": rb.std_vec,
                },
                save_ckpt_path,
            )

    # 5) sample standardized blocks (P-space)
    Z_samp = sample_blocks_ddpm(
        model=model,
        n_samples=cfg.n_mc,
        H=cfg.H,
        alphas=alphas,
        alphas_bar=alphas_bar,
        betas=betas,
        device=device,
        chunk=5000,
    )  # (M,H)

    X_samp_P = unstandardize_blocks_per_dim(Z_samp, rb)  # (M,H) log returns in P-like data space

    # 6) crude RN mean shift (first pass)
    # per-step shift: X_Q = X_P + (mQ - mP_h)
    mP_h = rb.mean_vec.astype(np.float64)  # (H,)
    mQ = (cfg.r - 0.5 * cfg.sigma_anchor**2) * dt
    X_samp_Q = X_samp_P + (mQ - mP_h[None, :])

    # 7) price smiles
    Ks = np.array([k * S0 for k in cfg.Ks_over_S0], dtype=np.float64)

    ST_P = terminal_prices_from_return_blocks(X_samp_P, S0=S0)
    pP, seP = price_calls_from_ST(ST_P, Ks=Ks, r=cfg.r, T=T_total)
    df_ddpm_P = smile_table_from_prices(S0=S0, Ks=Ks, prices=pP, stderrs=seP, r=cfg.r, T=T_total)
    df_ddpm_P["label"] = "DDPM_P"

    ST_Q = terminal_prices_from_return_blocks(X_samp_Q, S0=S0)
    pQ, seQ = price_calls_from_ST(ST_Q, Ks=Ks, r=cfg.r, T=T_total)
    df_ddpm_Q = smile_table_from_prices(S0=S0, Ks=Ks, prices=pQ, stderrs=seQ, r=cfg.r, T=T_total)
    df_ddpm_Q["label"] = "DDPM_Q_mean_shift"

    df_bs = bs_smile_table(S0=S0, Ks=Ks, r=cfg.r, T=T_total, sigma=cfg.sigma_anchor)
    df_bs["label"] = "BS_flat"

    return {
        "df_ddpm_P": df_ddpm_P,
        "df_ddpm_Q": df_ddpm_Q,
        "df_bs": df_bs,
        "train_stats": train_stats.losses if train_stats is not None else None,
        "rb_stats": {
            "global_mean": rb.global_mean,
            "global_std": rb.global_std,
            "mean_vec_first5": rb.mean_vec[:5].tolist(),
            "std_vec_first5": rb.std_vec[:5].tolist(),
            "n_blocks": int(rb.blocks.shape[0]),
        },
        "S0": S0,
        "T_total": T_total,
    }