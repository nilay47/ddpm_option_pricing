"""
h_psi: the learned Doob h-function that amortizes Q* into the frozen sampler.

    h_tau(z) = E_{P_theta}[ L*(X) | Z_tau = z ] = q_tau(z) / p_tau(z)
    grad log q_tau = grad log p_tau + grad log h_tau

Parameterization (DECISIONS.md section 11):

    h_psi(z, tau) = h_min + (1 - h_min) * exp( abar_tau * s_psi(z, tau) )

so h -> 1 as abar -> 0 (tau -> T) structurally, h > h_min always, and the
loss is MSE on h against L* directly (never log L: E[log L | z] != log E[L | z]).
s_psi reuses the sinusoidal time-embedding module from src.schedules.

Everything here is in the network's standardized z-space, which is the space
the sampler runs in; the only raw-space quantity is the target L*(x), computed
once from the de-standardized path by taskc.dual.Tilt.
"""

from dataclasses import dataclass
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import taskc  # noqa: F401
from src.schedules import sinusoidal_time_embedding
from taskc.config import TaskCConfig, CFG
from taskc.ptheta import Schedule


# --------------------------------------------------------------------------

class HNet(nn.Module):
    """s_psi(z, t01) -> scalar; h = h_min + (1-h_min) exp(abar * s)."""

    # numerical guard only: abar*s <= S_MAX so exp never overflows in the first
    # steps of training (max L* on B is 15 -> abar*s ~ 2.7 at convergence).
    S_MAX = 10.0

    def __init__(self, data_dim: int = 21, hidden_dim: int = 256, time_emb_dim: int = 32,
                 h_min: float = 0.05):
        super().__init__()
        self.data_dim, self.time_emb_dim, self.h_min = data_dim, time_emb_dim, float(h_min)
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)      # start at h == 1 everywhere
        nn.init.zeros_(self.net[-1].bias)

    def s(self, z, t01):
        return self.net(torch.cat([z, sinusoidal_time_embedding(t01, dim=self.time_emb_dim)], dim=1)).squeeze(1)

    def forward(self, z, t01, abar_t):
        """h_psi(z, tau). abar_t: (B,) tensor of abar at the same t as t01."""
        a = torch.clamp(abar_t * self.s(z, t01), max=self.S_MAX)
        return self.h_min + (1.0 - self.h_min) * torch.exp(a)

    def log_h(self, z, t01, abar_t):
        return torch.log(self.forward(z, t01, abar_t))

    @property
    def floor_threshold(self) -> float:
        """abar*s below this puts h within 1 % of h_min."""
        return float(np.log(0.01 * self.h_min / (1.0 - self.h_min)))


def grad_log_h(hnet: HNet, z, t01, abar_t):
    """grad_z log h_psi(z, tau) by autograd; z need not require grad on entry."""
    with torch.enable_grad():
        z = z.detach().requires_grad_(True)
        lh = hnet.log_h(z, t01, abar_t).sum()
        g, = torch.autograd.grad(lh, z)
    return g.detach()


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

@dataclass
class HTrainLog:
    epoch_loss: list
    floor_frac: list          # fraction of training samples with h within 1 % of h_min
    clamp_frac: list          # fraction hitting the S_MAX numerical guard
    seconds: float


def train_hnet(hnet: HNet, z0: np.ndarray, L: np.ndarray, sched: Schedule, device="cpu",
               epochs: int = 100, batch_size: int = 512, lr: float = 1e-3, lr_min: float = 1e-5,
               weight_decay: float = 1e-4, ema_decay: float = 0.999, seed: int = 0,
               verbose: bool = True) -> HTrainLog:
    """
    MSE( h_psi(z_tau, tau), L*(x) ) with tau ~ Uniform{0..T-1}, z_tau by forward
    noising of z0 with fresh eps. Full tau range, no reweighting (section 11).
    Cosine lr decay + EMA (the step-1 lesson). Returns the log; `hnet` is left
    holding the EMA weights, raw weights in hnet.raw_state_dict.
    """
    from tqdm.auto import tqdm
    device = torch.device(device)
    hnet.to(device).train()
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    z0_t = torch.from_numpy(np.asarray(z0, dtype=np.float32))
    L_t = torch.from_numpy(np.asarray(L, dtype=np.float32))
    abar = sched.alphas_bar.to(device)
    T = sched.T
    n = z0_t.shape[0]
    steps_per_epoch = (n + batch_size - 1) // batch_size
    opt = torch.optim.AdamW(hnet.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch, eta_min=lr_min)
    ema = {k: v.detach().clone() for k, v in hnet.state_dict().items()}
    thr = hnet.floor_threshold
    log = HTrainLog([], [], [], 0.0)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot, nfl, ncl, cnt = 0.0, 0, 0, 0
        pbar = tqdm(range(steps_per_epoch), desc=f"h epoch {ep+1}/{epochs}", disable=not verbose)
        for i in pbar:
            idx = perm[i * batch_size:(i + 1) * batch_size]
            z = z0_t[idx].to(device); tgt = L_t[idx].to(device)
            t = torch.randint(0, T, (z.shape[0],), device=device)
            ab = abar[t]
            eps = torch.randn_like(z)
            zt = torch.sqrt(ab).view(-1, 1) * z + torch.sqrt(1 - ab).view(-1, 1) * eps
            t01 = (t.float() + 0.5) / T
            a = ab * hnet.s(zt, t01)
            h = hnet.h_min + (1 - hnet.h_min) * torch.exp(torch.clamp(a, max=HNet.S_MAX))
            loss = ((h - tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            with torch.no_grad():
                for k, v in hnet.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)
                nfl += int((a < thr).sum()); ncl += int((a > HNet.S_MAX).sum()); cnt += a.numel()
            tot += loss.item() * z.shape[0]
            pbar.set_description(f"h epoch {ep+1}/{epochs} - MSE {loss.item():.4f} lr {opt.param_groups[0]['lr']:.1e}")
        log.epoch_loss.append(tot / n); log.floor_frac.append(nfl / cnt); log.clamp_frac.append(ncl / cnt)
    log.seconds = time.time() - t0
    hnet.raw_state_dict = {k: v.detach().cpu().clone() for k, v in hnet.state_dict().items()}
    hnet.load_state_dict(ema)
    hnet.eval()
    return log


# --------------------------------------------------------------------------
# diagnostics (run on draw C only)
# --------------------------------------------------------------------------

@torch.no_grad()
def tower_curve(hnet: HNet, z0: np.ndarray, sched: Schedule, ts, device="cpu", seed=1,
                chunk: int = 20_000):
    """E_C[h_tau(Z_tau)] and its SE at each tau in ts (target: 1)."""
    device = torch.device(device)
    z0_t = torch.from_numpy(np.asarray(z0, dtype=np.float32))
    abar = sched.alphas_bar.to(device); T = sched.T
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for t in ts:
        vals = []
        for lo in range(0, z0_t.shape[0], chunk):
            z = z0_t[lo:lo + chunk].to(device)
            eps = torch.randn(z.shape, generator=gen).to(device)
            zt = torch.sqrt(abar[t]) * z + torch.sqrt(1 - abar[t]) * eps
            t01 = torch.full((z.shape[0],), (t + 0.5) / T, device=device)
            vals.append(hnet(zt, t01, abar[t].expand(z.shape[0])).cpu())
        v = torch.cat(vals).numpy()
        out[int(t)] = dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))),
                           frac_near_floor=float((v < hnet.h_min * 1.01).mean()),
                           max=float(v.max()), min=float(v.min()))
    return out


@torch.no_grad()
def h0_vs_L(hnet: HNet, z0: np.ndarray, L: np.ndarray, sched: Schedule, device="cpu", chunk=20_000):
    """h at tau = 0 on clean z0 vs the true L*: correlation and RMSE (diagnostic, not gated)."""
    device = torch.device(device)
    z0_t = torch.from_numpy(np.asarray(z0, dtype=np.float32)); T = sched.T
    ab0 = sched.alphas_bar[0].to(device)
    hs = []
    for lo in range(0, z0_t.shape[0], chunk):
        z = z0_t[lo:lo + chunk].to(device)
        hs.append(hnet(z, torch.full((z.shape[0],), 0.5 / T, device=device), ab0.expand(z.shape[0])).cpu())
    h = torch.cat(hs).numpy()
    return dict(corr=float(np.corrcoef(h, L)[0, 1]), rmse=float(np.sqrt(np.mean((h - L) ** 2))),
                mean_h=float(h.mean()), mean_L=float(L.mean()),
                r2=float(1 - np.mean((h - L) ** 2) / np.var(L)))


def grad_ratio_curve(hnet: HNet, eps_model, z0: np.ndarray, sched: Schedule, ts, device="cpu",
                     n: int = 5_000, seed=2):
    """||grad log h_tau|| / ||grad log p_tau|| averaged over n forward-noised C paths, per tau.
    grad log p_tau = -eps_theta(z_tau, tau) / sqrt(1 - abar_tau)."""
    device = torch.device(device)
    z0_t = torch.from_numpy(np.asarray(z0[:n], dtype=np.float32)).to(device)
    abar = sched.alphas_bar.to(device); T = sched.T
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for t in ts:
        eps = torch.randn(z0_t.shape, generator=gen).to(device)
        zt = torch.sqrt(abar[t]) * z0_t + torch.sqrt(1 - abar[t]) * eps
        t01 = torch.full((n,), (t + 0.5) / T, device=device)
        gh = grad_log_h(hnet, zt, t01, abar[t].expand(n))
        with torch.no_grad():
            gp = -eps_model(zt, t01) / torch.sqrt(1 - abar[t])
        nh, npp = gh.norm(dim=1), gp.norm(dim=1)
        out[int(t)] = dict(ratio_mean=float((nh / npp).mean()), ratio_median=float((nh / npp).median()),
                           grad_h_norm=float(nh.mean()), grad_p_norm=float(npp.mean()),
                           eps_corr_norm=float((torch.sqrt(1 - abar[t]) * nh).mean()))
    return out


# --------------------------------------------------------------------------

def save_hnet(path: str, hnet: HNet, meta: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(dict(state=hnet.state_dict(), raw_state=getattr(hnet, "raw_state_dict", None),
                    data_dim=hnet.data_dim, time_emb_dim=hnet.time_emb_dim, h_min=hnet.h_min,
                    hidden_dim=hnet.net[0].out_features, meta=meta), path)


def load_hnet(path: str, device="cpu") -> HNet:
    d = torch.load(path, map_location="cpu", weights_only=False)
    h = HNet(d["data_dim"], d["hidden_dim"], d["time_emb_dim"], d["h_min"])
    h.load_state_dict(d["state"]); h.meta = d["meta"]
    return h.to(torch.device(device)).eval()
