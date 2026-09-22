"""
Regenerate the Gaussian (GBM) appendix tables 6, 7, 9, 10 and the 2-D table 13
with NO end-of-process mean adjustment and a single de-standardization with the
SOURCE mean. src/ and src_v2/ are not modified: the published 1-dim ScoreMLP,
cosine schedule, eps-shift constants (rn_noise_shift_std) and reverse-step
arithmetic are reused through taskc.sampler.reverse_ancestral (bitwise the
src_v2 loop, plus the eps hook).

Rows:
  no shift       P_theta sampled as trained
  exact SMT      analytic eps-shift  eps -> eps - sqrt(abar(1-abar)) d,  d = (r-mu)dt/s0
  estimated SMT  the ICLR pipeline in one dimension: martingale constraint
                 E_Q[e^Y] = e^{r dt} -> convex dual on P_theta samples -> L* ->
                 h_psi regression -> corrected sampler. No analytic target anywhere.
  retrained Q    the same DDPM trained on N(mQ, s0^2) samples, sampled unshifted
  lambda sweep   exact shift scaled: r_used = mu + lambda (r - mu)

    ~/.venv/bin/python3 -m taskc.gauss_appendix --device mps [--quick]
"""
import argparse, json, math, os, time
from dataclasses import dataclass, asdict
import numpy as np
import torch
from scipy.stats import norm, ks_2samp, wasserstein_distance
from scipy.optimize import minimize

import taskc  # noqa
from src.schedules import make_alpha_schedule
from src.ddpm_model import ScoreMLP as ScoreMLP1D
from src_v2.diffusion.score_mlp import ScoreMLP as ScoreMLPND
from src_v2.finance.risk_neutral import rn_noise_shift_std
from src.price_options import black_scholes_price
from taskc.sampler import reverse_ancestral, last_unclipped_step
from taskc.hnet import HNet, train_hnet, grad_log_h


# --------------------------------------------------------------------------
# configuration (stated in full; DECISIONS.md section 16)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GaussCfg:
    S0: float = 100.0
    mu: float = 0.12
    r: float = 0.05
    sigma: float = 0.20
    dt: float = 1.0 / 252.0
    H: int = 21
    N_train: int = 50_000
    T: int = 1000
    cosine_s: float = 0.008
    hidden: int = 256
    time_emb: int = 64
    epochs: int = 450
    batch: int = 512
    lr: float = 1e-3
    lr_min: float = 1e-5
    wd: float = 1e-4
    ema: float = 0.999
    z_cap: float = 8.0
    seed_data_P: int = 42
    seed_data_Q: int = 43
    seed_init_P: int = 0
    seed_init_Q: int = 1
    n_paths: int = 100_000          # tables 6, 7, 10 (H = 21)
    n_paths_mat: int = 20_000       # table 9 (H up to 126)
    n_lambda: int = 50_000          # table 7 rows other than lambda = 0, 1
    strikes: tuple = (80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0)
    maturities: tuple = (5, 10, 21, 42, 63, 84, 126)
    lambdas: tuple = (-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    seed_draw: int = 2026
    chunk: int = 200_000

    @property
    def s0(self): return self.sigma * math.sqrt(self.dt)
    @property
    def mP(self): return (self.mu - 0.5 * self.sigma ** 2) * self.dt
    @property
    def mQ(self): return (self.r - 0.5 * self.sigma ** 2) * self.dt
    @property
    def d_std(self): return (self.mQ - self.mP) / self.s0     # standardized mean shift


# --------------------------------------------------------------------------
# training (published loop + cosine decay + EMA, as in taskc.ptheta)
# --------------------------------------------------------------------------

def train_1d(z: np.ndarray, sched, cfg: GaussCfg, device, init_seed: int, model=None, verbose=False):
    from tqdm.auto import tqdm
    torch.manual_seed(init_seed)
    model = (ScoreMLP1D(hidden_dim=cfg.hidden, time_emb_dim=cfg.time_emb) if model is None else model).to(device).train()
    z = np.asarray(z, dtype=np.float32); z = z.reshape(len(z), -1)
    x = torch.from_numpy(z)
    abar = sched[2].to(device); T = cfg.T
    g = torch.Generator().manual_seed(init_seed)
    n = x.shape[0]; spe = (n + cfg.batch - 1) // cfg.batch
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * spe, eta_min=cfg.lr_min)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(cfg.epochs):
        perm = torch.randperm(n, generator=g)
        for i in tqdm(range(spe), disable=not verbose, desc=f"ep {ep+1}"):
            b = x[perm[i * cfg.batch:(i + 1) * cfg.batch]].to(device)
            t = torch.randint(0, T, (b.shape[0],), device=device)
            ab = abar[t].view(-1, 1); noise = torch.randn_like(b)
            xt = torch.sqrt(ab) * b + torch.sqrt(1 - ab) * noise
            loss = torch.nn.functional.mse_loss(model(xt, (t.float() + 0.5) / T), noise)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point: ema[k].mul_(cfg.ema).add_(v, alpha=1 - cfg.ema)
    model.load_state_dict(ema); return model.eval()


# --------------------------------------------------------------------------
# sampling: the published loop through reverse_ancestral; de-standardize with mP
# --------------------------------------------------------------------------

@torch.no_grad()
def draw_z(model, sched, n: int, dim: int, seed: int, device, cfg: GaussCfg, eps_correction=None, t_start=None):
    betas, alphas, abar = sched
    torch.manual_seed(seed)
    out, kept, drawn, rej = [], 0, 0, 0
    while kept < n:
        m = min(cfg.chunk, n - kept)
        z = reverse_ancestral(model, (m, dim), alphas, abar, betas, torch.device(device), t_start=t_start,
                              eps_correction=eps_correction).cpu().numpy()
        ok = np.abs(z).max(axis=1) <= cfg.z_cap
        rej += int((~ok).sum()); drawn += m; z = z[ok]; kept += z.shape[0]; out.append(z)
        if drawn > 2 * n: raise RuntimeError("rejection budget exceeded")
    return np.concatenate(out)[:n], rej


def exact_shift_hook(sched, cfg: GaussCfg, lam: float, device):
    """eps -> eps - noise_shift_std[t], with the published constant, scaled by lambda."""
    r_used = cfg.mu + lam * (cfg.r - cfg.mu)
    shift = rn_noise_shift_std(mu=cfg.mu, r=r_used, dt=cfg.dt, s0=cfg.s0, alphas_bar=sched[2], device=torch.device(device))
    return lambda y, t01, t: -shift[t].view(1, 1).expand_as(y)


@torch.no_grad()
def draw_z_lambda_batched(model, sched, lams, n_each: int, seed: int, device, cfg: GaussCfg, t_start=None):
    """
    One shared reverse pass for every exact-shift configuration at once: the batch
    carries a per-sample lambda, and the hook applies -lambda_i * noise_shift_std[t].
    lambda = 0 is the no-shift row, lambda = 1 the exact-SMT row. Returns {lam: z (n_each,)}.
    Rejection (cap 8) is applied per row; every row gets exactly n_each samples.
    """
    betas, alphas, abar = sched
    shift1 = rn_noise_shift_std(mu=cfg.mu, r=cfg.r, dt=cfg.dt, s0=cfg.s0, alphas_bar=abar, device=torch.device(device))  # lambda = 1
    lams_t = torch.tensor(lams, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    L = len(lams); per_chunk = max(1, cfg.chunk // L)
    out = {lam: [] for lam in lams}; kept = {lam: 0 for lam in lams}; rej = {lam: 0 for lam in lams}; drawn = 0
    while min(kept.values()) < n_each:
        need = [lam for lam in lams if kept[lam] < n_each]
        m = min(per_chunk, n_each)
        lam_vec = torch.cat([torch.full((m,), lam, device=device) for lam in need])
        hook = lambda y, t01, t: -(lam_vec * shift1[t]).view(-1, 1)
        z = reverse_ancestral(model, (lam_vec.shape[0], 1), alphas, abar, betas, torch.device(device), t_start=t_start, eps_correction=hook).cpu().numpy()[:, 0]
        drawn += lam_vec.shape[0]
        for k, lam in enumerate(need):
            zz = z[k * m:(k + 1) * m]; ok = np.abs(zz) <= cfg.z_cap
            rej[lam] += int((~ok).sum()); zz = zz[ok]; out[lam].append(zz); kept[lam] += len(zz)
        if drawn > 3 * L * n_each: raise RuntimeError("rejection budget exceeded")
    return {lam: np.concatenate(out[lam])[:n_each] for lam in lams}, rej


def evaluate_config(z: np.ndarray, cfg: GaussCfg, rng, n_h21: int = None, n_mat: int = None):
    """From one i.i.d. standardized pool z (1-dim): the H=21 metrics (tables 6/7/10) and the
    maturity curve (table 9). Pools overlap between the two views (same i.i.d. law)."""
    n_h21 = cfg.n_paths if n_h21 is None else n_h21; n_mat = cfg.n_paths_mat if n_mat is None else n_mat
    Y = cfg.s0 * z + cfg.mP                                    # de-standardize ONCE with the SOURCE mean
    Hmax = max(cfg.maturities)
    assert len(Y) >= max(n_h21 * cfg.H, n_mat * Hmax), "pool too small"
    Y21 = Y[:n_h21 * cfg.H].reshape(n_h21, cfg.H); S21 = paths_from_returns(Y21, cfg.S0)
    rows, rmse, mae = price_table(S21, cfg, cfg.H); ts = terminal_stats(S21[:, cfg.H], cfg, cfg.H, rng)
    h21 = dict(mart_dev=martingale_dev(S21, cfg.r, cfg.dt), rmse=rmse, mae=mae, W1=ts["W1"], KS=ts["KS"],
               mean_return=float(Y.mean()), bias_sd_vs_mP=float((Y.mean() - cfg.mP) / cfg.s0), bias_sd_vs_mQ=float((Y.mean() - cfg.mQ) / cfg.s0),
               sd_return=float(Y.std()), prices=rows, terminal=ts,
               mart_profile=((S21 * np.exp(-cfg.r * cfg.dt * np.arange(cfg.H + 1))).mean(0) - cfg.S0).tolist())
    Ymat = Y[:n_mat * Hmax].reshape(n_mat, Hmax); Smat = paths_from_returns(Ymat, cfg.S0)
    mat = {}
    for H in cfg.maturities:
        pay = math.exp(-cfg.r * H * cfg.dt) * np.maximum(Smat[:, H] - cfg.S0, 0)
        mat[H] = dict(atm=float(pay.mean()), se=float(pay.std(ddof=1) / math.sqrt(n_mat)), bs=black_scholes_price(cfg.S0, cfg.S0, H * cfg.dt, cfg.r, cfg.sigma, "call"))
    return dict(h21=h21, maturity=mat)


def pool_size(cfg: GaussCfg) -> int:
    return max(cfg.n_paths * cfg.H, cfg.n_paths_mat * max(cfg.maturities))


def tables_markdown(res: dict, cfg: GaussCfg) -> str:
    """Render tables 6, 7, 9, 10, 13 from the results dict."""
    L = []
    d = res["derived"]
    L.append(f"Configuration: mu={cfg.mu} r={cfg.r} sigma={cfg.sigma} dt=1/252 H={cfg.H} N_train={cfg.N_train} | s0={d['s0']:.6f} mP={d['mP']:.4e} mQ={d['mQ']:.4e} shift d={d['d_std']:+.4f} sd | device {res['device']} | draws: {cfg.n_paths} x 21 (T6/7/10), {cfg.n_paths_mat} x 126 (T9)\n")
    L.append("**Table 6 — Gaussian results** (martingale deviation max_h |E[e^{-rh dt} S_h] - S0|; RMSE/MAE vs Black-Scholes over K=80..120; W1/KS of S_21 vs lognormal Q)\n")
    L.append("| Method | mean-return bias (sd) | Mart. Dev. | RMSE | MAE | W1 | KS |\n|---|---|---|---|---|---|---|")
    for name in ("No shift", "Exact SMT", "Estimated SMT", "Retrained Q"):
        r = res["table6"].get(name)
        if r is None: L.append(f"| {name} | — | — | — | — | — | — |"); continue
        bias = r["bias_sd_vs_mP"] if name == "No shift" else r["bias_sd_vs_mQ"]
        L.append(f"| {name} | {bias:+.4f} | {r['mart_dev']:.4f} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['W1']:.4f} | {r['KS']:.4f} |")
    L.append("\n**Table 7 — lambda sweep** (r_used = mu + lambda (r - mu); lambda=0 is the no-shift draw, lambda=1 the exact-SMT draw)\n")
    L.append("| lambda | Mart. Dev. | RMSE | W1 | n |\n|---|---|---|---|---|")
    for k in sorted(res.get("table7", {}), key=float):
        r = res["table7"][k]; L.append(f"| {float(k):+.2f} | {r['mart_dev']:.4f} | {r['rmse']:.4f} | {r['W1']:.4f} | {r['n']} |")
    L.append("\n**Table 9 — ATM call vs maturity H** (price; abs error vs BS)\n")
    names = [n for n in ("No shift", "Exact SMT", "Estimated SMT", "Retrained Q") if n in res["table9"]]
    if names:
        L.append("| H | " + " | ".join(names) + " | BS |\n|---|" + "---|" * (len(names) + 1))
        for H in cfg.maturities:
            row = f"| {H} | " + " | ".join(f"{res['table9'][n][str(H)]['atm']:.3f} ({abs(res['table9'][n][str(H)]['atm']-res['table9'][n][str(H)]['bs']):.3f})" for n in names)
            L.append(row + f" | {res['table9'][names[0]][str(H)]['bs']:.3f} |")
    else:
        L.append("(not yet computed)")
    L.append("\n**Table 10 — terminal S_T at H=21**\n")
    L.append("| Method | E[S_T] | Std(S_T) | KS vs theory | E[e^{-rT} S_T] |\n|---|---|---|---|---|")
    for name in ("No shift", "Exact SMT", "Estimated SMT", "Retrained Q"):
        r = res["table10"].get(name)
        if r: L.append(f"| {name} | {r['E_ST']:.2f} | {r['sd_ST']:.2f} | {r['KS']:.4f} | {r['E_disc']:.2f} |")
    r = next(iter(res["table10"].values())) if res["table10"] else None
    if r: L.append(f"| Theory Q | {r['mean_theory']:.2f} | {r['sd_theory']:.2f} | 0 | 100.00 |")
    if res.get("table13"):
        t = res["table13"]; L.append("\n**Table 13 — 2-D Gaussian transport** (mP=(0,0) -> mQ=(2,1), s0=(0.5,0.5): a 4 sd / 2 sd shift)\n")
        L.append("| | mean x1 | mean x2 | std x1 | std x2 |\n|---|---|---|---|---|")
        for k in ("No shift", "SMT", "Q_ref", "target"):
            if k in t: L.append(f"| {k} | {t[k]['mean'][0]:.3f} | {t[k]['mean'][1]:.3f} | {t[k]['std'][0]:.3f} | {t[k]['std'][1]:.3f} |")
        if "SMT" in t: L.append(f"\nSMT reaches {t['SMT']['mean'][0]/2*100:.0f} % / {t['SMT']['mean'][1]/1*100:.0f} % of the displacement; std is {t['SMT']['std'][0]/0.5*100:.0f} % / {t['SMT']['std'][1]/0.5*100:.0f} % of target.")
    return "\n".join(L)


# --------------------------------------------------------------------------
# finance metrics
# --------------------------------------------------------------------------

def paths_from_returns(Y: np.ndarray, S0: float) -> np.ndarray:
    S = np.empty((Y.shape[0], Y.shape[1] + 1)); S[:, 0] = S0; S[:, 1:] = S0 * np.exp(np.cumsum(Y, axis=1)); return S


def martingale_dev(S: np.ndarray, r: float, dt: float) -> float:
    disc = np.exp(-r * dt * np.arange(S.shape[1]))
    return float(np.abs((S * disc).mean(0) - S[0, 0]).max())


def price_table(S: np.ndarray, cfg: GaussCfg, H: int):
    T = H * cfg.dt; disc = math.exp(-cfg.r * T); ST = S[:, H]
    rows = []
    for K in cfg.strikes:
        pay = disc * np.maximum(ST - K, 0.0)
        rows.append((K, float(pay.mean()), float(pay.std(ddof=1) / math.sqrt(len(pay))),
                     black_scholes_price(cfg.S0, K, T, cfg.r, cfg.sigma, "call")))
    err = np.array([p - bs for _, p, _, bs in rows])
    return rows, float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def terminal_stats(ST: np.ndarray, cfg: GaussCfg, H: int, rng):
    T = H * cfg.dt
    mean_th = cfg.S0 * math.exp(cfg.r * T); sd_th = mean_th * math.sqrt(math.exp(cfg.sigma ** 2 * T) - 1)
    # reference lognormal sample under Q for KS / W1 (same size)
    ref = cfg.S0 * np.exp(rng.normal((cfg.r - 0.5 * cfg.sigma ** 2) * T, cfg.sigma * math.sqrt(T), size=len(ST)))
    logST = np.log(ST / cfg.S0)
    ks_theory = float(ks_2samp(logST, np.log(ref / cfg.S0)).statistic)
    return dict(E_ST=float(ST.mean()), sd_ST=float(ST.std(ddof=1)), E_disc=float(math.exp(-cfg.r * T) * ST.mean()),
                KS=ks_theory, W1=float(wasserstein_distance(ST, ref)), mean_theory=mean_th, sd_theory=sd_th)


# --------------------------------------------------------------------------
# estimated SMT: the pipeline in one dimension
# --------------------------------------------------------------------------

def dual_1d(y: np.ndarray, cfg: GaussCfg):
    """min_beta log mean exp(beta g) - beta c,  g(y) = e^y - e^{r dt}, c = 0  (discrete martingale)."""
    g = (np.exp(np.asarray(y, dtype=np.float64)) - math.exp(cfg.r * cfg.dt)); m, s = float(g.mean()), float(g.std()); gs = (g - m) / s; cs = (0.0 - m) / s
    def obj(b):
        z = gs * b[0]; zm = z.max(); w = np.exp(z - zm); lse = math.log(w.mean()) + zm
        return lse - b[0] * cs, np.array([(gs * w).sum() / w.sum() - cs])
    res = minimize(obj, np.zeros(1), jac=True, method="L-BFGS-B", options=dict(ftol=1e-14, gtol=1e-12))
    b_std = float(res.x[0]); b_raw = b_std / s
    z = (g - m) * b_raw; lse = float(math.log(np.mean(np.exp(z - z.max()))) + z.max())
    w = np.exp(z - z.max()); w /= w.sum()
    return dict(beta_raw=float(b_raw), beta_std=float(b_std), m=float(m), lse=float(lse), ess=float(1 / np.sum(w ** 2) / len(w)),
                fitted=float((w * g).sum()), converged=bool(res.success))


def logL_1d(y, dual, cfg: GaussCfg):
    """log L*(y) = beta_raw ((e^y - e^{r dt}) - m_A) - lse_A, m_A/lse_A from the solve draw."""
    return ((np.exp(y) - math.exp(cfg.r * cfg.dt)) - dual["m"]) * dual["beta_raw"] - dual["lse"]


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default=None); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="artifacts_gauss")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    cfg = GaussCfg()
    if args.quick:
        cfg = GaussCfg(N_train=20_000, epochs=40, n_paths=5_000, n_paths_mat=2_000, n_lambda=2_000, chunk=50_000)
    os.makedirs(args.out, exist_ok=True)
    t_all = time.time()
    print(f"== config: {asdict(cfg)}\n   s0={cfg.s0:.6f} mP={cfg.mP:.6e} mQ={cfg.mQ:.6e} d_std={cfg.d_std:+.5f}  device={device}", flush=True)
    sched = make_alpha_schedule(T=cfg.T, device=torch.device(device), s=cfg.cosine_s)
    t_start = last_unclipped_step(sched[0])
    rng = np.random.default_rng(cfg.seed_draw)

    # ---- data + models ------------------------------------------------------
    yP = np.random.default_rng(cfg.seed_data_P).normal(cfg.mP, cfg.s0, cfg.N_train)
    yQ = np.random.default_rng(cfg.seed_data_Q).normal(cfg.mQ, cfg.s0, cfg.N_train)
    zP, zQ = (yP - cfg.mP) / cfg.s0, (yQ - cfg.mP) / cfg.s0          # ONE standardizer (source), for both models
    t0 = time.time(); modelP = train_1d(zP, sched, cfg, device, cfg.seed_init_P); tP = time.time() - t0
    t0 = time.time(); modelQ = train_1d(zQ, sched, cfg, device, cfg.seed_init_Q); tQ = time.time() - t0
    print(f"trained P_theta in {tP/60:.1f} min, retrained-Q in {tQ/60:.1f} min ({cfg.epochs} epochs, {cfg.N_train:,} samples each)", flush=True)
    torch.save(dict(P=modelP.state_dict(), Q=modelQ.state_dict(), cfg=asdict(cfg), device=device), os.path.join(args.out, "models.pt"))

    # ---- estimated SMT: dual on P_theta draw A, h_psi on draw B ------------------
    nA = 1_000_000 if not args.quick else 50_000
    zA, _ = draw_z(modelP, sched, nA, 1, cfg.seed_draw + 1, device, cfg, t_start=t_start)
    dual = dual_1d(cfg.s0 * zA[:, 0] + cfg.mP, cfg)
    print(f"dual (1 constraint E[e^Y]=e^(r dt)) on {nA:,} P_theta draws: beta_raw={dual['beta_raw']:.4f} ESS={dual['ess']*100:.2f}% fitted={dual['fitted']:.2e} conv={dual['converged']}", flush=True)
    nB = 200_000 if not args.quick else 20_000
    zB, _ = draw_z(modelP, sched, nB, 1, cfg.seed_draw + 2, device, cfg, t_start=t_start)
    LB = np.exp(logL_1d(cfg.s0 * zB[:, 0] + cfg.mP, dual, cfg))
    from taskc.ptheta import Schedule
    S = Schedule(sched[0], sched[1], sched[2], t_start)
    hnet = HNet(1, 256, 32, 0.05)
    hlog = train_hnet(hnet, zB.astype(np.float32), LB, S, device=device, epochs=100 if not args.quick else 5, verbose=False)
    print(f"h_psi trained ({hlog.seconds/60:.1f} min): E_B[L*]={LB.mean():.4f} min={LB.min():.3f} max={LB.max():.3f}; final MSE {hlog.epoch_loss[-1]:.2e}; floor active {max(hlog.floor_frac):.1e}", flush=True)
    abar_dev = sched[2]
    def est_hook(y, t01, t):
        return -torch.sqrt(1 - abar_dev[t]) * grad_log_h(hnet, y, t01, abar_dev[t].expand(y.shape[0]))
    json.dump(dict(dual=dual, hpsi=dict(E_L_B=float(LB.mean()), min=float(LB.min()), max=float(LB.max()), mse=hlog.epoch_loss, seconds=hlog.seconds)),
              open(os.path.join(args.out, "estimated_smt.json"), "w"), indent=1)

    # ---- draws for tables 6/7/10 (H = 21), pooled 126 columns for table 9 ----------
    def returns(model, n, H, seed, hook=None):
        z, rej = draw_z(model, sched, n * H, 1, seed, device, cfg, eps_correction=hook, t_start=t_start)
        return (cfg.s0 * z + cfg.mP).reshape(n, H), rej      # de-standardize ONCE with the SOURCE mean

    methods = {
        "No shift":      (modelP, None),
        "Exact SMT":     (modelP, exact_shift_hook(sched, cfg, 1.0, device)),
        "Estimated SMT": (modelP, est_hook),
        "Retrained Q":   (modelQ, None),
    }
    Hmax = max(cfg.maturities)
    res6, res9, res10, pools = {}, {}, {}, {}
    for i, (name, (model, hook)) in enumerate(methods.items()):
        t0 = time.time()
        Y21, rej21 = returns(model, cfg.n_paths, cfg.H, cfg.seed_draw + 10 + i, hook)
        S21 = paths_from_returns(Y21, cfg.S0)
        rows, rmse, mae = price_table(S21, cfg, cfg.H)
        ts = terminal_stats(S21[:, cfg.H], cfg, cfg.H, rng)
        res6[name] = dict(mart_dev=martingale_dev(S21, cfg.r, cfg.dt), rmse=rmse, mae=mae, W1=ts["W1"], KS=ts["KS"],
                          mean_return=float(Y21.mean()), sd_return=float(Y21.std()), rejected=rej21, prices=rows)
        res10[name] = ts
        Ymat, _ = returns(model, cfg.n_paths_mat, Hmax, cfg.seed_draw + 20 + i, hook)
        Smat = paths_from_returns(Ymat, cfg.S0)
        res9[name] = {H: dict(atm=float(np.mean(math.exp(-cfg.r * H * cfg.dt) * np.maximum(Smat[:, H] - cfg.S0, 0))),
                              se=float(np.std(math.exp(-cfg.r * H * cfg.dt) * np.maximum(Smat[:, H] - cfg.S0, 0), ddof=1) / math.sqrt(cfg.n_paths_mat)),
                              bs=black_scholes_price(cfg.S0, cfg.S0, H * cfg.dt, cfg.r, cfg.sigma, "call")) for H in cfg.maturities}
        pools[name] = Y21
        print(f"  {name:14s} mean ret {Y21.mean():.3e} (mP {cfg.mP:.3e}, mQ {cfg.mQ:.3e})  mart {res6[name]['mart_dev']:.4f}  rmse {rmse:.4f}  mae {mae:.4f}  W1 {ts['W1']:.4f}  KS {ts['KS']:.4f}  rej {rej21}  {time.time()-t0:.0f}s", flush=True)

    # ---- table 7: lambda sweep (lambda = 0 reuses the no-shift draw exactly; lambda = 1 the exact-SMT draw)
    res7 = {}
    for lam in cfg.lambdas:
        if lam == 0.0:
            r6 = res6["No shift"]; res7[lam] = dict(mart_dev=r6["mart_dev"], rmse=r6["rmse"], mae=r6["mae"], W1=r6["W1"], n=cfg.n_paths, source="No shift row"); continue
        if lam == 1.0:
            r6 = res6["Exact SMT"]; res7[lam] = dict(mart_dev=r6["mart_dev"], rmse=r6["rmse"], mae=r6["mae"], W1=r6["W1"], n=cfg.n_paths, source="Exact SMT row"); continue
        Y, rej = returns(modelP, cfg.n_lambda, cfg.H, cfg.seed_draw + 100 + int(lam * 100), exact_shift_hook(sched, cfg, lam, device))
        Sl = paths_from_returns(Y, cfg.S0); _, rmse, mae = price_table(Sl, cfg, cfg.H); ts = terminal_stats(Sl[:, cfg.H], cfg, cfg.H, rng)
        res7[lam] = dict(mart_dev=martingale_dev(Sl, cfg.r, cfg.dt), rmse=rmse, mae=mae, W1=ts["W1"], n=cfg.n_lambda, rejected=rej, source="run")
        print(f"  lambda={lam:+.2f}: mart {res7[lam]['mart_dev']:.4f} rmse {rmse:.4f} W1 {ts['W1']:.4f}", flush=True)

    # ---- table 13: 2-D Gaussian transport, same recipe, no adjustment -----------------
    mP2, mQ2, s2 = np.array([0.0, 0.0]), np.array([2.0, 1.0]), np.array([0.5, 0.5])
    N2 = cfg.N_train; x2 = np.random.default_rng(7).normal(mP2, s2, (N2, 2)); z2 = (x2 - mP2) / s2
    torch.manual_seed(0); m2 = ScoreMLPND(data_dim=2, hidden_dim=cfg.hidden, time_emb_dim=cfg.time_emb)
    m2 = train_1d(z2, sched, cfg, device, 0, model=m2)
    d2 = torch.tensor((mQ2 - mP2) / s2, dtype=torch.float32, device=device)
    abar = sched[2]
    hook2 = lambda y, t01, t: -(torch.sqrt(abar[t] * (1 - abar[t])) * d2).view(1, 2).expand_as(y)
    n2 = cfg.n_paths if not args.quick else 5_000
    zs = {"No shift": draw_z(m2, sched, n2, 2, 501, device, cfg, t_start=t_start)[0], "SMT": draw_z(m2, sched, n2, 2, 502, device, cfg, hook2, t_start)[0]}
    Qref = np.random.default_rng(9).normal(mQ2, s2, (n2, 2))
    res13 = {k: dict(mean=(v * s2 + mP2).mean(0).tolist(), std=(v * s2 + mP2).std(0).tolist(),
                     W1_x1=float(wasserstein_distance((v * s2 + mP2)[:, 0], Qref[:, 0])), W1_x2=float(wasserstein_distance((v * s2 + mP2)[:, 1], Qref[:, 1])))
             for k, v in zs.items()}
    res13["Q_ref"] = dict(mean=Qref.mean(0).tolist(), std=Qref.std(0).tolist()); res13["target"] = dict(mean=mQ2.tolist(), std=s2.tolist(), shift_sd=(mQ2 / s2).tolist())
    print(f"  2D: SMT mean {res13['SMT']['mean']} std {res13['SMT']['std']} (target {mQ2.tolist()}, {s2.tolist()}); no-shift mean {res13['No shift']['mean']}", flush=True)

    out = dict(config=asdict(cfg), derived=dict(s0=cfg.s0, mP=cfg.mP, mQ=cfg.mQ, d_std=cfg.d_std, t_start=t_start),
               device=device, seeds=dict(data_P=cfg.seed_data_P, data_Q=cfg.seed_data_Q, init_P=cfg.seed_init_P, init_Q=cfg.seed_init_Q, draw_base=cfg.seed_draw),
               train_seconds=dict(P=tP, Q=tQ), table6=res6, table7={str(k): v for k, v in res7.items()},
               table9={m: {str(H): v for H, v in d.items()} for m, d in res9.items()}, table10=res10, table13=res13,
               estimated_smt=dict(dual=dual, E_L_B=float(LB.mean())), total_seconds=time.time() - t_all,
               notes="no mean adjustment anywhere; de-standardization Y = s0 z + mP once; cap 8 by rejection; sampler from t_start; estimated SMT = 1-D projection pipeline (non-circular)")
    json.dump(out, open(os.path.join(args.out, "gauss_appendix.json"), "w"), indent=1)
    print(f"\nDONE {(time.time()-t_all)/60:.1f} min -> {args.out}/gauss_appendix.json", flush=True)


if __name__ == "__main__":
    main()
