"""Cross-sectional factor screen.

Ranks the universe on four factors, converts each to a cross-sectional z-score,
and blends them into a single composite. Scores are also computed
sector-neutrally, so a sector-wide rally does not decide the leaderboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    FACTOR_WEIGHTS,
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP,
    RSI_WINDOW,
)

# Winsorise z-scores so one extreme name cannot dominate the composite.
Z_CLIP = 3.0


def _zscore(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    s = series.astype(float)
    sd = s.std()
    z = (s - s.mean()) / sd if sd and sd > 0 else pd.Series(0.0, index=s.index)
    z = z.clip(-Z_CLIP, Z_CLIP)
    return z if higher_is_better else -z


def momentum_12_1(prices: pd.Series) -> float:
    """12-month return skipping the last month — the standard momentum factor.

    The skip removes the short-term reversal effect that contaminates a raw
    12-month lookback.
    """
    p = prices.dropna()
    if len(p) < MOMENTUM_LOOKBACK + 1:
        return np.nan
    start = p.iloc[-(MOMENTUM_LOOKBACK + 1)]
    end = p.iloc[-(MOMENTUM_SKIP + 1)]
    return float(end / start - 1) if start > 0 else np.nan


def build_factor_table(frames: dict[str, pd.DataFrame], metrics: pd.DataFrame) -> pd.DataFrame:
    """Factor exposures, z-scores and the blended composite score."""
    factors = pd.DataFrame(index=metrics.index)
    factors["Sector"] = metrics["Sector"]
    factors["Momentum 12-1"] = [momentum_12_1(frames[t]["Close"]) for t in factors.index]
    factors["Sharpe"] = metrics["Sharpe"]
    factors["Volatility"] = metrics["Volatility"]
    factors["Max Drawdown"] = metrics["Max Drawdown"]
    factors = factors.dropna(subset=["Momentum 12-1"])

    z = pd.DataFrame(index=factors.index)
    z["momentum"] = _zscore(factors["Momentum 12-1"])
    z["risk_adjusted"] = _zscore(factors["Sharpe"])
    z["low_volatility"] = _zscore(factors["Volatility"], higher_is_better=False)
    # Drawdowns are negative, so a shallower (larger) value is better.
    z["drawdown_resilience"] = _zscore(factors["Max Drawdown"])

    for factor, zs in z.items():
        factors[f"z {factor.replace('_', ' ').title()}"] = zs

    factors["Composite Score"] = sum(z[f] * w for f, w in FACTOR_WEIGHTS.items())
    factors["Rank"] = factors["Composite Score"].rank(ascending=False, method="min").astype(int)

    # Sector-neutral variant: rank within sector, so the screen is not just a
    # bet on whichever sector happened to lead.
    factors["Sector Score"] = factors.groupby("Sector")["Composite Score"].transform(
        lambda s: (s - s.mean()) / s.std() if s.std() and s.std() > 0 else s * 0
    )
    factors["Sector Rank"] = factors.groupby("Sector")["Composite Score"].rank(
        ascending=False, method="min"
    ).astype(int)

    return factors.sort_values("Composite Score", ascending=False)


def current_signals(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Latest technical state for every symbol — the screen's 'today' view."""
    rows = []
    for ticker, df in frames.items():
        last = df.iloc[-1]
        window = df.tail(60)
        rows.append({
            "Ticker": ticker,
            "Close": float(last["Close"]),
            "Trend": last["Trend"],
            f"RSI {RSI_WINDOW}": float(last[f"RSI_{RSI_WINDOW}"]),
            "RSI Zone": str(last["RSI_Zone"]),
            "MACD Hist": float(last["MACD_Hist"]),
            "Bollinger": last["BB_Breakout"],
            "% From SMA200": float(last["Pct_From_SMA200"]),
            "Drawdown": float(last["Drawdown"]) / 100,
            "Volume vs 20d": float(last["Volume_Ratio"]),
            "Crosses (60d)": int(window["Golden_Cross"].sum() + window["Death_Cross"].sum()),
        })
    return pd.DataFrame(rows).set_index("Ticker")


def sector_rollup(metrics: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted sector aggregates of the headline metrics."""
    agg = metrics.groupby("Sector").agg(
        Holdings=("CAGR", "size"),
        CAGR=("CAGR", "mean"),
        Volatility=("Volatility", "mean"),
        Sharpe=("Sharpe", "mean"),
        Beta=("Beta", "mean"),
        **{"Max Drawdown": ("Max Drawdown", "mean")},
        **{"Total Return": ("Total Return", "mean")},
        **{"Hit Rate": ("Hit Rate", "mean")},
    )
    return agg.sort_values("CAGR", ascending=False)
