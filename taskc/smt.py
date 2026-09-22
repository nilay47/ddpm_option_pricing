"""
Step 4: the SMT (corrected) sampler and the amortization unit test.

Sampler: the frozen reverse loop (`taskc.sampler.reverse_ancestral`, same t_start,
same arithmetic) with the eps hook

    eps_hat -> eps_hat + delta,   delta = -sqrt(1 - abar_t) * grad_z log h_psi(z_t, t)

because eps = -sqrt(1 - abar) * score and grad log q = grad log p + grad log h.
The cap is enforced by rejection exactly as for P_theta (DECISIONS.md section 11c);
the rejection rate is reported separately.

Unit test (execution plan, Task C.5): weighted P_theta under L* (draw C, weights
from A's beta*) versus the unweighted SMT draw. If they disagree the bug is in
amortization, not in finance. Sliced Wasserstein needs a null: SW between two
independent P_theta draws (A vs C, unweighted).

Strided (respaced) ancestral sampler: ONLY for the discretization diagnostic.
Its output at K < 999 steps is NOT P_theta and is labelled so.
"""

from dataclasses import dataclass
import time
import numpy as np
import torch

import taskc  # noqa: F401
from taskc.config import TaskCConfig, CFG
from taskc.data import apply_cap
from taskc.hnet import HNet, grad_log_h
from taskc.ptheta import Schedule, DrawResult
from taskc.sampler import reverse_ancestral


def make_eps_correction(hnet: HNet, sched: Schedule):
    abar = sched.alphas_bar

    def delta(y, t01, t):
        a = abar[t].expand(y.shape[0])
        return -torch.sqrt(1.0 - abar[t]) * grad_log_h(hnet, y, t01, a)
    return delta


@torch.no_grad()
def sample_smt(model, hnet: HNet, sched: Schedule, n: int, seed: int, cfg: TaskCConfig = CFG,
               device="cpu", verbose: bool = True, max_redraw_factor: float = 2.0) -> DrawResult:
    """Corrected draw with cap-by-rejection; same seeding/chunking as sample_ptheta."""
    device = torch.device(device)
    model = model.to(device).eval(); hnet = hnet.to(device).eval(); sched = sched.to(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    delta = make_eps_correction(hnet, sched)
    t0 = time.time(); chunks, kept, drawn, rejected = [], 0, 0, 0
    while kept < n:
        if drawn >= max_redraw_factor * n:
            raise RuntimeError(f"SMT rejection rate {rejected/drawn:.3f} exceeds the redraw budget")
        m = min(cfg.sample_chunk, n - kept)
        z = reverse_ancestral(model, (m, cfg.data_dim), sched.alphas, sched.alphas_bar, sched.betas,
                              device, t_start=sched.t_start, eps_correction=delta)
        z = z.detach().cpu().numpy().astype(np.float32)
        z_ok, n_rej = apply_cap(z, cfg.z_cap)
        drawn += m; rejected += n_rej; kept += z_ok.shape[0]; chunks.append(z_ok)
        if verbose:
            print(f"  smt seed={seed}: {kept:,}/{n:,} kept, {rejected} rejected, {time.time()-t0:.0f}s", flush=True)
    z = np.concatenate(chunks, 0)[:n]
    return DrawResult(z=z, seed=seed, n_requested=n, n_drawn=drawn, n_rejected=rejected,
                      seconds=time.time() - t0, device=str(device))


# --------------------------------------------------------------------------
# strided ancestral sampler -- discretization diagnostic only (NOT P_theta)
# --------------------------------------------------------------------------

@torch.no_grad()
def reverse_ancestral_strided(model, shape, sched: Schedule, n_steps: int, device,
                              eps_correction=None):
    """
    Ancestral DDPM on a subsequence of n_steps timesteps from t_start down to 0,
    with the respaced schedule alpha_eff = abar_t / abar_prev (Nichol & Dhariwal).
    The time encoding always uses the full T. Labelled NOT P_theta whenever
    n_steps < t_start + 1.
    """
    device = torch.device(device)
    abar_full = sched.alphas_bar.to(device); T = sched.T
    ts = np.unique(np.round(np.linspace(sched.t_start, 0, n_steps)).astype(int))[::-1]  # descending
    B, D = shape
    y = torch.randn(B, D, device=device)
    for i, t in enumerate(ts):
        t = int(t)
        ab_t = abar_full[t]
        t01 = torch.full((B,), (t + 0.5) / T, device=device)
        eps_hat = model(y, t01)
        if eps_correction is not None:
            eps_hat = eps_hat + eps_correction(y, t01, t)
        if i < len(ts) - 1:
            t_prev = int(ts[i + 1]); ab_prev = abar_full[t_prev]
            alpha_eff = ab_t / ab_prev; beta_eff = 1.0 - alpha_eff
            post_var = torch.clamp((1.0 - ab_prev) / (1.0 - ab_t) * beta_eff, min=1e-12)
            mean = (1.0 / torch.sqrt(alpha_eff)) * (y - (beta_eff / torch.sqrt(1.0 - ab_t)) * eps_hat)
            y = mean + torch.sqrt(post_var) * torch.randn_like(y)
        else:
            # last step to t = 0 exactly as the full sampler's t=0 step
            beta0 = sched.betas.to(device)[0]; alpha0 = sched.alphas.to(device)[0]
            y = (1.0 / torch.sqrt(alpha0)) * (y - (beta0 / torch.sqrt(1.0 - abar_full[0])) * eps_hat) if t == 0 else \
                (y - torch.sqrt(1.0 - ab_t) * eps_hat) / torch.sqrt(ab_t)
    return y


@torch.no_grad()
def sample_strided(model, sched: Schedule, n: int, n_steps: int, seed: int, cfg: TaskCConfig = CFG,
                   device="cpu", hnet: HNet = None) -> DrawResult:
    device = torch.device(device)
    model = model.to(device).eval(); sched = sched.to(device)
    delta = None
    if hnet is not None:
        hnet = hnet.to(device).eval(); delta = make_eps_correction(hnet, sched)
    torch.manual_seed(seed)
    t0 = time.time(); chunks, kept, drawn, rejected = [], 0, 0, 0
    while kept < n:
        if drawn >= 2 * n:
            raise RuntimeError("strided sampler rejection budget exceeded")
        m = min(cfg.sample_chunk, n - kept)
        z = reverse_ancestral_strided(model, (m, cfg.data_dim), sched, n_steps, device, delta)
        z = z.detach().cpu().numpy().astype(np.float32)
        z_ok, n_rej = apply_cap(z, cfg.z_cap)
        drawn += m; rejected += n_rej; kept += z_ok.shape[0]; chunks.append(z_ok)
    return DrawResult(z=np.concatenate(chunks, 0)[:n], seed=seed, n_requested=n, n_drawn=drawn,
                      n_rejected=rejected, seconds=time.time() - t0, device=str(device))


# --------------------------------------------------------------------------
# sliced Wasserstein
# --------------------------------------------------------------------------

def sliced_wasserstein(X: np.ndarray, Y: np.ndarray, n_proj: int = 256, seed: int = 0,
                       wX: np.ndarray = None) -> float:
    """
    SW_2 between samples X and Y in R^D (equal-weight, or X weighted by wX).
    Weighted 1-D Wasserstein via the weighted quantile function.
    """
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    P = rng.standard_normal((n_proj, D)); P /= np.linalg.norm(P, axis=1, keepdims=True)
    q = np.linspace(0.0005, 0.9995, 2000)
    tot = 0.0
    for k in range(n_proj):
        px, py = X @ P[k], Y @ P[k]
        if wX is None:
            qx = np.quantile(px, q)
        else:
            o = np.argsort(px); cw = np.cumsum(wX[o]); cw /= cw[-1]
            qx = px[o][np.searchsorted(cw, q, side="left").clip(0, len(px) - 1)]
        qy = np.quantile(py, q)
        tot += np.mean((qx - qy) ** 2)
    return float(np.sqrt(tot / n_proj))
