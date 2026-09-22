"""
Task C configuration. Every number here is pre-registered in DECISIONS.md;
change the file and the log there together.
"""

import os
from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

import taskc  # noqa: F401  (sys.path setup for taskb)
from config import P_PARAMS, SIM_P, SimConfig, Heston  # taskb


@dataclass(frozen=True)
class Draw:
    seed: int
    n: int


@dataclass(frozen=True)
class TaskCConfig:
    # --- market / path geometry (inherited from taskb) ---------------------
    H: int = SIM_P.H                       # 21 daily steps
    dt: float = SIM_P.dt                   # 1/252
    S0: float = P_PARAMS.S0                # 100
    r: float = P_PARAMS.r                  # 0.05 discount rate

    # --- simulator the prior is trained on (Task D overrides this per prior) ---
    heston: Heston = P_PARAMS

    # --- training data: the same Heston-P sample Task B projected against ---
    train_seed: int = SIM_P.seed           # 20260920
    n_train: int = SIM_P.n_paths           # 100_000
    # independent Heston-P reference sample for the step-2 gate
    ref_seed: int = 20260922
    n_ref: int = 100_000

    # --- diffusion schedule (src.schedules, unchanged) ---------------------
    T: int = 1000
    cosine_s: float = 0.008

    # --- model (src_v2.diffusion.score_mlp.ScoreMLP, unchanged) -----------
    data_dim: int = 21
    hidden_dim: int = 512
    time_emb_dim: int = 64
    init_seed: int = 0

    # --- training loop (src.train_ddpm.train_ddpm, unchanged) --------------
    epochs: int = 150
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # rung 2 of the escalation ladder (DECISIONS.md section 7): Min-SNR-gamma
    # t-reweighting (Hang et al. 2023). OFF unless rung 1 has failed and the
    # change is logged in DECISIONS.md section 9.
    snr_weighting: bool = False
    snr_gamma: float = 5.0
    # training-schedule change approved 2026-09-20 (DECISIONS.md section 9): cosine
    # lr decay lr -> lr_min over the run, and an EMA of the weights used for
    # sampling. Addresses the constant-lr last-iterate noise floor that produced
    # the per-coordinate level bias. OFF by default so rungs 0-2 stay reproducible.
    lr_decay: bool = False
    lr_min: float = 1e-5
    ema: bool = False
    ema_decay: float = 0.999

    # --- outlier rule (DECISIONS.md §4) -----------------------------------
    z_cap: float = 8.0

    # --- frozen sampler -----------------------------------------------------
    # Start the ancestral chain at the last step whose beta is below the
    # cosine-schedule clip (0.999): t_start = 998 at T=1000. The clipped step
    # t=999 amplifies eps error by 1/sqrt(alpha) = 31.6 and derails 21-dim
    # paths (DECISIONS.md section 3 and change log).
    skip_clipped_steps: bool = True
    beta_clip: float = 0.999
    sample_chunk: int = 10_000

    # --- the three disjoint P_theta draws (DECISIONS.md §5) ----------------
    draws: Dict[str, Draw] = field(default_factory=lambda: {
        "A": Draw(seed=1001, n=100_000),   # gate, screen, dual solve
        "B": Draw(seed=1002, n=100_000),   # h_psi targets
        "C": Draw(seed=1003, n=100_000),   # evaluation
    })
    # Task D: the dual is solved on a 1e6 draw (DECISIONS.md section 14.1)
    solve_draw: Draw = Draw(seed=1101, n=1_000_000)

    # --- artifacts ------------------------------------------------------------
    artifact_dir: str = "artifacts_taskc"
    ckpt_name: str = "ptheta_mlp21.pt"

    def run_dir(self, tag: str) -> str:
        """Artifact sub-directory for one rung of the escalation ladder."""
        return os.path.join(self.artifact_dir, tag)

    def train_sim(self) -> SimConfig:
        return SimConfig(n_paths=self.n_train, H=self.H, dt=self.dt, seed=self.train_seed)

    def ref_sim(self) -> SimConfig:
        return SimConfig(n_paths=self.n_ref, H=self.H, dt=self.dt, seed=self.ref_seed)


CFG = TaskCConfig()

# The recipe the frozen P_theta was trained with (DECISIONS.md section 2, amended
# 2026-09-20, and section 9). CFG keeps lr_decay/ema off so rungs 0-2 stay
# reproducible, which means a bare `replace(CFG, ...)` silently trains the
# constant-lr model that FAILS the step-2 gate. Anything that trains a model for
# the paper must start from FROZEN, not from CFG.
FROZEN = replace_cfg = TaskCConfig(epochs=450, lr_decay=True, ema=True)


def frozen(**overrides) -> TaskCConfig:
    """FROZEN with overrides; refuses to silently drop the recipe."""
    cfg = replace(FROZEN, **overrides)
    assert cfg.lr_decay and cfg.ema and cfg.epochs == 450, (
        "the frozen recipe is 450 epochs with cosine lr decay and EMA; "
        "override it only with a DECISIONS.md log entry")
    return cfg
