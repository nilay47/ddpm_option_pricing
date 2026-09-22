"""
Shape / plumbing smoke tests for taskc step 1. No training to convergence, no
full pipeline: a T=50 schedule, a tiny model, a few hundred paths.

Run:  ~/.venv/bin/python3 -m pytest taskc/tests -q
  or: ~/.venv/bin/python3 taskc/tests/test_shapes.py
"""

import os
import sys
import tempfile
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import taskc  # noqa: F401,E402
from taskc.config import CFG, Draw  # noqa: E402
from taskc.data import (PathStandardizer, apply_cap, build_training_set, make_loader,  # noqa: E402
                        reference_paths)
from taskc.ptheta import (make_schedule, build_model, train_ptheta, sample_ptheta,  # noqa: E402
                          save_checkpoint, load_checkpoint)
from constraints import build, feasibility_screen  # noqa: E402  (taskb)
from config import CALIB_TESTFUNS, VANILLA_C3, HELDOUT_TESTFUNS, HELDOUT_VANILLAS  # noqa: E402
import evaluation as ev  # noqa: E402

SMALL = replace(CFG, n_train=3_000, n_ref=2_000, T=50, hidden_dim=32, time_emb_dim=8,
                epochs=1, sample_chunk=128,
                draws={"A": Draw(seed=1, n=300)})


def test_standardizer_roundtrip_and_paths():
    rng = np.random.default_rng(0)
    Y = rng.normal(3e-4, 0.0126, size=(500, 21))
    std = PathStandardizer.fit(Y, S0=100.0, r=0.05, dt=1 / 252)
    z = std.to_z(Y)
    assert abs(z.mean()) < 1e-12 and abs(z.std() - 1) < 1e-12
    assert np.allclose(std.to_Y(z), Y)
    p = std.to_paths(z)
    assert p.S.shape == (500, 22) and p.v is None and p.S0 == 100.0
    assert np.all(p.S[:, 0] == 100.0)
    assert np.allclose(p.returns(), Y)            # de-standardization is exact
    # the taskb constraint builder must accept generated paths as-is
    cs = build(p, CALIB_TESTFUNS, VANILLA_C3)
    assert cs.G.shape == (500, 53) and cs.c.shape == (53,)
    ho = build(p, HELDOUT_TESTFUNS, HELDOUT_VANILLAS)
    assert ho.G.shape == (500, 24)
    scr = feasibility_screen(cs)
    assert scr["margin"].shape == (53,)
    ex = ev.price_exotics(p)
    assert set(ex) == {"asian_call_K100", "uo_barrier_call_K100_B110", "lookback_float_call"}
    # state_dict roundtrip
    std2 = PathStandardizer.from_state_dict(std.state_dict())
    assert std2 == std


def test_cap():
    z = np.zeros((5, 21)); z[2, 7] = 9.0; z[4, 0] = -8.5
    kept, n_rej = apply_cap(z, 8.0)
    assert kept.shape == (3, 21) and n_rej == 2


def test_training_set_and_reference():
    ts = build_training_set(SMALL)
    assert ts.z.shape[1] == 21 and ts.z.dtype == np.float32
    assert ts.z.shape[0] + ts.n_rejected == SMALL.n_train
    assert ts.n_rejected == 0                     # cap binds on no real data
    assert ts.max_abs_z < SMALL.z_cap
    assert abs(ts.z.mean()) < 0.05 and abs(ts.z.std() - 1) < 0.05
    ref = reference_paths(SMALL)
    assert ref.S.shape == (SMALL.n_ref, 22) and ref.v is None


class ExactGaussianEps(torch.nn.Module):
    """E[eps | x_t] = sqrt(1 - abar_t) x_t for x_0 ~ N(0, I): the exact noise
    predictor for standardized Gaussian data. Lets the sampler be exercised
    without a trained net (an untrained net blows up through the clipped
    last beta, which is what the redraw guard is for)."""
    def __init__(self, sched):
        super().__init__()
        self.abar = sched.alphas_bar.clone()
    def forward(self, x, t01):
        t = torch.round(t01 * self.abar.shape[0] - 0.5).long().clamp(0, self.abar.shape[0] - 1)
        return torch.sqrt(1.0 - self.abar[t]).view(-1, 1) * x


def test_model_train_sample_checkpoint():
    ts = build_training_set(SMALL)
    sched = make_schedule(SMALL)
    assert sched.T == 50 and sched.alphas_bar[-1] < sched.alphas_bar[0]
    model = build_model(SMALL)
    x = torch.randn(7, 21); t = torch.rand(7)
    assert model(x, t).shape == (7, 21)
    loader = make_loader(ts.z, batch_size=256)
    model = train_ptheta(model, loader, sched, SMALL, epochs=1)      # runs, returns
    assert isinstance(model, type(build_model(SMALL)))
    # exercise the frozen sampler with the exact Gaussian predictor
    exact = ExactGaussianEps(sched)
    res = sample_ptheta(exact, sched, n=300, seed=1, cfg=SMALL, verbose=False)
    assert abs(res.z.mean()) < 0.15 and abs(res.z.std() - 1) < 0.15
    assert res.z.shape == (300, 21) and res.z.dtype == np.float32
    assert res.n_drawn >= 300 and res.n_drawn - res.n_rejected >= 300
    assert np.abs(res.z).max() <= SMALL.z_cap
    # same seed -> same draw on the same device
    res2 = sample_ptheta(exact, sched, n=300, seed=1, cfg=SMALL, verbose=False)
    assert np.array_equal(res.z, res2.z)
    # z -> Paths -> constraints on generated data
    p = ts.std.to_paths(res.z)
    assert build(p, CALIB_TESTFUNS, VANILLA_C3).G.shape == (300, 53)
    # checkpoint roundtrip
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.pt")
        save_checkpoint(path, model, ts.std, SMALL, extra={"note": "smoke"})
        m2, std2, c, extra = load_checkpoint(path)
        assert std2 == ts.std and c["hidden_dim"] == 32 and extra["note"] == "smoke"
        with torch.no_grad():
            assert torch.allclose(m2(x, t), model(x, t))


def test_reverse_loop_matches_sample_ddpm_x0():
    """With t_start = T-1 and no hook, taskc.sampler.reverse_ancestral must be
    bitwise the published src_v2 sampler; with the default t_start it starts
    at the last unclipped beta."""
    from src_v2.diffusion.sampler import sample_ddpm_x0
    from taskc.sampler import reverse_ancestral, last_unclipped_step
    cfgT = replace(SMALL, T=1000)
    sched = make_schedule(cfgT)
    assert sched.t_start == 998 == last_unclipped_step(sched.betas)
    assert float(sched.betas[999]) >= 0.999 and float(sched.betas[998]) < 0.999
    exact = ExactGaussianEps(sched)
    torch.manual_seed(5)
    a = sample_ddpm_x0(exact, (64, 21), sched.alphas, sched.alphas_bar, sched.betas, torch.device("cpu"))
    torch.manual_seed(5)
    b = reverse_ancestral(exact, (64, 21), sched.alphas, sched.alphas_bar, sched.betas,
                          torch.device("cpu"), t_start=sched.T - 1)
    assert torch.equal(a, b)
    # the hook is applied additively to eps_hat
    torch.manual_seed(5)
    c = reverse_ancestral(exact, (64, 21), sched.alphas, sched.alphas_bar, sched.betas,
                          torch.device("cpu"), t_start=sched.T - 1,
                          eps_correction=lambda y, t01, t: torch.zeros_like(y))
    assert torch.equal(a, c)
    # skip_clipped_steps=False restores the original start
    assert make_schedule(replace(cfgT, skip_clipped_steps=False)).t_start == 999


def test_pathological_generator_is_caught():
    class Explode(torch.nn.Module):
        def forward(self, x, t):
            return torch.full_like(x, -50.0)     # drives every path past the cap
    sched = make_schedule(SMALL)
    try:
        sample_ptheta(Explode(), sched, n=200, seed=0, cfg=SMALL, verbose=False)
    except RuntimeError as e:
        assert "pathological" in str(e)
    else:
        raise AssertionError("redraw budget guard did not fire")


if __name__ == "__main__":
    for fn in [test_standardizer_roundtrip_and_paths, test_cap, test_training_set_and_reference,
               test_model_train_sample_checkpoint, test_reverse_loop_matches_sample_ddpm_x0,
               test_pathological_generator_is_caught]:
        fn(); print(f"ok  {fn.__name__}")
