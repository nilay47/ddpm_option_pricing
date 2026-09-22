"""
P_theta: the 21-dim path DDPM.

Reuses, unchanged:
    src.schedules.make_alpha_schedule        cosine betas, alphas, alphas_bar
    src_v2.diffusion.score_mlp.ScoreMLP      time-conditioned MLP, data_dim=21
    src.train_ddpm.train_ddpm                eps-prediction training loop
    taskc.sampler.reverse_ancestral          the sample_ddpm_x0 loop, plus
                                             t_start and an eps hook

What is new is only the bookkeeping around them: a checkpoint that carries the
standardizer and config, and the frozen sampler wrapper that (i) seeds the draw,
(ii) chunks it, (iii) applies the outlier rule, (iv) hands back z. P_theta is
*defined* as the law this wrapper produces (DECISIONS.md section 3).
"""

from dataclasses import dataclass, asdict
import os
import time
import numpy as np
import torch

import taskc  # noqa: F401
from src.schedules import make_alpha_schedule
from src.train_ddpm import train_ddpm
from src_v2.diffusion.score_mlp import ScoreMLP
from taskc.sampler import reverse_ancestral, last_unclipped_step
from taskc.config import TaskCConfig, CFG
from taskc.data import PathStandardizer, apply_cap


# --------------------------------------------------------------------------
# schedule + model
# --------------------------------------------------------------------------

@dataclass
class Schedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_bar: torch.Tensor
    t_start: int                 # first reverse step of the frozen sampler

    @property
    def T(self) -> int:
        return int(self.alphas_bar.shape[0])

    def to(self, device) -> "Schedule":
        return Schedule(self.betas.to(device), self.alphas.to(device),
                        self.alphas_bar.to(device), self.t_start)


def make_schedule(cfg: TaskCConfig = CFG, device="cpu") -> Schedule:
    betas, alphas, alphas_bar = make_alpha_schedule(T=cfg.T, device=torch.device(device), s=cfg.cosine_s)
    t_start = last_unclipped_step(betas, cfg.beta_clip) if cfg.skip_clipped_steps else cfg.T - 1
    return Schedule(betas, alphas, alphas_bar, t_start)


def build_model(cfg: TaskCConfig = CFG) -> ScoreMLP:
    torch.manual_seed(cfg.init_seed)
    return ScoreMLP(data_dim=cfg.data_dim, hidden_dim=cfg.hidden_dim, time_emb_dim=cfg.time_emb_dim)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train_ptheta(model: ScoreMLP, loader, sched: Schedule, cfg: TaskCConfig = CFG,
                 device="cpu", epochs: int = None) -> ScoreMLP:
    """Thin pass-through to the published training loop (reused as the same code, not as
    validated output); `epochs` overridable for smoke tests. With cfg.snr_weighting=True
    (rung 2 of the ladder) dispatches to the Min-SNR-gamma weighted loop instead."""
    device = torch.device(device)
    if cfg.lr_decay or cfg.ema:
        return train_ptheta_decay_ema(model, loader, sched, cfg, device=device, epochs=epochs)
    if cfg.snr_weighting:
        return train_ptheta_snr_weighted(model, loader, sched, cfg, device=device, epochs=epochs,
                                         gamma=cfg.snr_gamma)
    return train_ddpm(
        model=model, train_loader=loader, alphas_bar=sched.alphas_bar.to(device),
        T=sched.T, device=device,
        epochs=cfg.epochs if epochs is None else epochs,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )


def train_ptheta_snr_weighted(model: ScoreMLP, loader, sched: Schedule, cfg: TaskCConfig = CFG,
                              device="cpu", epochs: int = None, gamma: float = 5.0,
                              verbose: bool = True) -> ScoreMLP:
    """
    Rung 2 of the escalation ladder (DECISIONS.md section 7). OFF by default.

    Same eps-prediction objective as `src.train_ddpm.train_ddpm`, with the
    per-sample loss reweighted by  w(t) = min(SNR_t, gamma) / SNR_t,
    SNR_t = abar_t / (1 - abar_t). Uniform-t training under-weights the
    high-noise steps because their absolute eps-loss is tiny, although those
    are the steps that set the global position of a sample (and the ones the
    frozen sampler visits first). This is the standard remedy, not a bespoke
    workaround: Min-SNR-gamma weighting, Hang et al., "Efficient Diffusion
    Training via Min-SNR Weighting Strategy", ICCV 2023 (arXiv 2303.09556);
    perception-prioritized weighting, Choi et al., "Perception Prioritized
    Training of Diffusion Models", CVPR 2022 (arXiv 2204.00227). In eps-space
    Min-SNR-gamma reduces to dividing the eps-loss by max(SNR_t, gamma)/gamma
    i.e. w(t) = min(1, gamma / SNR_t): full weight at high noise
    (SNR <= gamma), down-weighted at low noise where the eps target is mostly
    irreducible.
    """
    from tqdm.auto import tqdm
    device = torch.device(device)
    model.to(device).train()
    abar = sched.alphas_bar.to(device)
    snr = abar / (1.0 - abar)
    w_t = torch.clamp(gamma / snr, max=1.0)                 # (T,)
    T = sched.T
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for epoch in range(cfg.epochs if epochs is None else epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", disable=not verbose)
        for (batch,) in pbar:
            batch = batch.to(device)
            t = torch.randint(0, T, (batch.size(0),), device=device).long()
            ab = abar[t].view(-1, 1)
            noise = torch.randn_like(batch)
            x_t = torch.sqrt(ab) * batch + torch.sqrt(1 - ab) * noise
            pred = model(x_t, (t.float() + 0.5) / T)
            per = ((pred - noise) ** 2).mean(dim=1)
            loss = (w_t[t] * per).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_description(f"Epoch {epoch+1} - wLoss: {loss.item():.4f}")
    return model


def train_ptheta_decay_ema(model: ScoreMLP, loader, sched: Schedule, cfg: TaskCConfig = CFG,
                           device="cpu", epochs: int = None, verbose: bool = True) -> ScoreMLP:
    """
    The `src.train_ddpm.train_ddpm` loop (uniform t, eps-MSE, AdamW) with two
    training-schedule changes and nothing else:
      * cosine lr decay from cfg.lr to cfg.lr_min over all steps;
      * an exponential moving average of the weights (cfg.ema_decay), which is
        what gets returned and sampled from.
    Both are standard DDPM practice (Ho et al. 2020 use EMA 0.9999; Nichol &
    Dhariwal 2021 decay the lr). They target the constant-lr last-iterate noise
    floor diagnosed in docs/05_LEVEL_BIAS_DIAGNOSIS.md, which epochs and loss
    weighting could not move. Returns the EMA model (eval mode); the raw last
    iterate is attached as `model.raw_state_dict` for the record.
    """
    from tqdm.auto import tqdm
    device = torch.device(device)
    model.to(device).train()
    abar = sched.alphas_bar.to(device)
    T = sched.T
    n_epochs = cfg.epochs if epochs is None else epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = n_epochs * len(loader)
    if cfg.lr_decay:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=cfg.lr_min)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    d = cfg.ema_decay
    step = 0
    for epoch in range(n_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}", disable=not verbose)
        for (batch,) in pbar:
            batch = batch.to(device)
            t = torch.randint(0, T, (batch.size(0),), device=device).long()
            ab = abar[t].view(-1, 1)
            noise = torch.randn_like(batch)
            x_t = torch.sqrt(ab) * batch + torch.sqrt(1 - ab) * noise
            pred = model(x_t, (t.float() + 0.5) / T)
            loss = torch.nn.functional.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            if cfg.lr_decay:
                sch.step()
            step += 1
            if cfg.ema:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(d).add_(v, alpha=1 - d)
                        else:
                            ema[k].copy_(v)
            pbar.set_description(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item():.4f} "
                                 f"lr {opt.param_groups[0]['lr']:.1e}")
    raw = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if cfg.ema:
        model.load_state_dict(ema)
    model.raw_state_dict = raw
    return model.eval()


# --------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------

def save_checkpoint(path: str, model: ScoreMLP, std: PathStandardizer, cfg: TaskCConfig,
                    extra: dict = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "standardizer": std.state_dict(),
        "config": {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v)
                   for k, v in asdict(cfg).items()},
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device="cpu"):
    """Returns (model, standardizer, config_dict, extra). Model is in eval mode."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    c = payload["config"]
    model = ScoreMLP(data_dim=c["data_dim"], hidden_dim=c["hidden_dim"], time_emb_dim=c["time_emb_dim"])
    model.load_state_dict(payload["model_state"])
    model.to(torch.device(device)).eval()
    std = PathStandardizer.from_state_dict(payload["standardizer"])
    return model, std, c, payload["extra"]


# --------------------------------------------------------------------------
# the frozen sampler
# --------------------------------------------------------------------------

@dataclass
class DrawResult:
    z: np.ndarray            # (n, H) float32 standardized paths, capped
    seed: int
    n_requested: int
    n_drawn: int             # total paths generated including rejected
    n_rejected: int
    seconds: float
    device: str = "cpu"      # draws are seed-identified, not bitwise across devices

    @property
    def reject_rate(self) -> float:
        return self.n_rejected / max(self.n_drawn, 1)


@torch.no_grad()
def sample_ptheta(model: ScoreMLP, sched: Schedule, n: int, seed: int,
                  cfg: TaskCConfig = CFG, device="cpu", data_dim: int = None,
                  verbose: bool = True, max_redraw_factor: float = 2.0) -> DrawResult:
    """
    Draw n paths from P_theta with the frozen ancestral sampler
    (reverse_ancestral from t_start = sched.t_start, no eps correction).

    The draw is seeded through torch's global RNG on the target device — the
    published reverse loop calls torch.randn directly, and we do not touch it.
    Seeds identify a draw (A/B/C); they are not a claim of bitwise
    reproducibility across CPU/GPU.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    sched = sched.to(device)
    D = cfg.data_dim if data_dim is None else data_dim

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    chunks, kept, drawn, rejected = [], 0, 0, 0
    while kept < n:
        if drawn >= max_redraw_factor * n:
            raise RuntimeError(f"P_theta rejection rate {rejected/drawn:.3f} exceeds the "
                               f"redraw budget ({max_redraw_factor}x); the generator is "
                               "pathological, not the cap. Stop and inspect.")
        m = min(cfg.sample_chunk, n - kept)
        z = reverse_ancestral(model=model, shape=(m, D), alphas=sched.alphas,
                              alphas_bar=sched.alphas_bar, betas=sched.betas, device=device,
                              t_start=sched.t_start, eps_correction=None)
        z = z.detach().cpu().numpy().astype(np.float32)
        z_ok, n_rej = apply_cap(z, cfg.z_cap)
        drawn += m
        rejected += n_rej
        kept += z_ok.shape[0]
        chunks.append(z_ok)
        if verbose:
            print(f"  draw seed={seed}: {kept:,}/{n:,} kept, {rejected} rejected, "
                  f"{time.time()-t0:.0f}s", flush=True)
    z = np.concatenate(chunks, axis=0)[:n]
    return DrawResult(z=z, seed=seed, n_requested=n, n_drawn=drawn, n_rejected=rejected,
                      seconds=time.time() - t0, device=str(device))


def draw_path(cfg: TaskCConfig, name: str) -> str:
    return os.path.join(cfg.artifact_dir, f"ptheta_draw_{name}.npz")


def save_draw(cfg: TaskCConfig, name: str, res: DrawResult):
    os.makedirs(cfg.artifact_dir, exist_ok=True)
    np.savez_compressed(draw_path(cfg, name), z=res.z, seed=res.seed, n_requested=res.n_requested,
                        n_drawn=res.n_drawn, n_rejected=res.n_rejected, seconds=res.seconds,
                        device=np.array(res.device), t_start=cfg.T - 1 if not cfg.skip_clipped_steps else -1)


def load_draw(cfg: TaskCConfig, name: str) -> DrawResult:
    d = np.load(draw_path(cfg, name))
    return DrawResult(z=d["z"], seed=int(d["seed"]), n_requested=int(d["n_requested"]),
                      n_drawn=int(d["n_drawn"]), n_rejected=int(d["n_rejected"]),
                      seconds=float(d["seconds"]),
                      device=str(d["device"]) if "device" in d.files else "cpu")
