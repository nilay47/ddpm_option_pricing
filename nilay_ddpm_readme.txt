DDPM Risk Neutral Option Pricing
This document describes the code accompanying the draft
“A Generative Diffusion Framework for Risk Neutral Derivative Pricing.”
The repository includes the DDPM training pipeline, the risk neutral epsilon shift,
and all experiments used in the paper including martingale tests, KS tests,
European option pricing, and arithmetic Asian option pricing.
Repository Structure
ddpm_option_pricing/
• src/
 • ddpm_model.py - ScoreNet or UNet architecture for DDPM
 • schedules.py - Beta schedules and time embeddings
 • train_ddpm.py - DDPM training loop under the P measure
 • sample_ddpm.py - Reverse diffusion with the risk neutral epsilon shift
 • price_options.py - European option pricing using DDPM or GBM
 • price_asian_options.py - Discrete arithmetic Asian option pricing
 • experiments.py — High level experiment utilities
How to Run Locally
Open and run the notebook named “nilay_ddpm_experiments.ipynb”.
The notebook will:
• Train or load the DDPM on historical log returns
• Apply the risk neutral epsilon shift
• Generate simulated price paths
• Run martingale and KS tests
• Price European and Asian options under the learned risk neutral dynamics
How to Run in Google Colab
Upload the notebook directly to Colab and run it.
Since the repository is public, no authentication or cloning steps are required.
Notes
The implementation follows the derivations in the accompanying paper, including:
• Forward SDE to Fokker Planck to reverse SDE
• Fisher identity for score estimation
• Risk neutral drift correction using the epsilon shift
European pricing is benchmarked against Black Scholes and GBM Monte Carlo.
Asian pricing uses discrete arithmetic averages with a GBM Monte Carlo benchmark.