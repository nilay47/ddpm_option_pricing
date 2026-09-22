# Frozen P_theta

`ptheta_mlp21.pt` is the 21-dim path DDPM used throughout the paper: the model the
step-2 gate passed and every downstream result (dual, h_psi, amortization, overlay
baseline) is built on. Committed so a Colab session can load the *same* model
rather than retraining one.

Contents: 8 float32 tensors (580,117 parameters), the global standardizer
(m, s, S0, r, dt), the training config, and the training metadata. Nothing else.

Recipe (see `metadata.json`, and `taskc.config.FROZEN`): 450 epochs, AdamW
lr 1e-3 -> 1e-5 cosine, weight decay 1e-4, batch 512, EMA 0.999 (the EMA weights
are what is stored and sampled), cosine beta schedule T=1000, sampler from
t_start = 998, outlier cap 8 by rejection. Trained on 1e5 Heston-P paths
(seed 20260920), H=21, dt=1/252.

Load with `taskc.ptheta.load_checkpoint`; the standardizer it returns must equal
the one `taskc.data.build_training_set` fits, and the notebooks assert that.
