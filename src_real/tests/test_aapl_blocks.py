import numpy as np
import torch

from src_real.data.prices_yf import load_prices_yfinance
from src_real.data.blocks import build_return_blocks  # if your function name differs, tell me

def main():
    ticker = "AAPL"
    start = "2023-06-01"
    end   = "2024-06-01"
    H = 21

    df = load_prices_yfinance(ticker=ticker, start=start, end=end)
    close = df["Close"].to_numpy(dtype=np.float64)

    # log returns (daily)
    rets = np.diff(np.log(close))  # shape (N-1,)

    blocks = make_return_blocks(rets, H=H)  # shape (n_blocks, H)
    print("blocks shape:", blocks.shape)

    print("blocks overall mean:", float(blocks.mean()))
    print("blocks overall std :", float(blocks.std(ddof=1)))
    print("blocks min/max     :", float(blocks.min()), float(blocks.max()))
    print("per-dim mean first 5:", blocks.mean(axis=0)[:5])
    print("per-dim std  first 5:", blocks.std(axis=0, ddof=1)[:5])

    # standardize using train stats
    m_hat = blocks.mean()
    s_hat = blocks.std(ddof=1)
    z = (blocks - m_hat) / (s_hat if s_hat > 1e-12 else 1.0)

    print("Z mean/std:", float(z.mean()), float(z.std(ddof=1)))
    print("Z min/max :", float(z.min()), float(z.max()))

if __name__ == "__main__":
    main()