# src_real/data/prices_yf.py

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

def load_prices_yfinance(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    price_col: str = "Adj Close",
) -> np.ndarray:
    """
    Download daily prices via yfinance.

    Args:
        ticker: e.g. "AAPL"
        start: "YYYY-MM-DD"
        end: "YYYY-MM-DD" or None
        price_col: "Adj Close" or "Close"

    Returns:
        1D numpy array of float64 prices
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError("yfinance not installed. Run: pip install yfinance") from e

    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for {ticker} from {start} to {end}")

    # yfinance sometimes returns a column MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if price_col not in df.columns:
        raise ValueError(f"Missing column '{price_col}'. Got columns: {list(df.columns)}")

    prices = df[price_col].astype(float).to_numpy()
    prices = prices[~np.isnan(prices)]
    if len(prices) < 2:
        raise ValueError("Not enough prices after cleaning")

    return prices