"""Cleaning and validation for raw OHLCV frames.

Every symbol is reindexed onto a shared NYSE-session calendar so that
cross-sectional maths (correlation, portfolio weights, sector aggregates) can
assume aligned dates. Each symbol also emits a quality record, which the
pipeline rolls up into a data-quality report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# A day whose absolute return exceeds this is flagged for review, not removed —
# real markets do gap, and silently deleting the tails distorts risk metrics.
OUTLIER_SIGMA = 8.0


def _session_calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Union of every symbol's trading days, restricted to weekdays."""
    idx = pd.DatetimeIndex([])
    for df in frames.values():
        idx = idx.union(pd.DatetimeIndex(df.index))
    return idx[idx.dayofweek < 5].sort_values()


def clean(df: pd.DataFrame, ticker: str, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Clean one symbol and return ``(frame, quality_record)``."""
    raw_rows = len(df)
    df = df[[c for c in OHLCV if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index)

    duplicates = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Prices must be strictly positive; a zero or negative print is bad data.
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] <= 0, col] = np.nan
    non_positive = int(df[price_cols].isna().sum().sum())

    # Align to the shared calendar, then carry the last print across holidays.
    listed = calendar[(calendar >= df.index.min()) & (calendar <= df.index.max())]
    df = df.reindex(listed)
    gaps = int(df["Close"].isna().sum())
    df[price_cols] = df[price_cols].ffill()

    if "Volume" in df.columns:
        # A missing session has no volume — 0 is the honest value, not a carry-forward.
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64")

    df = df.dropna(subset=["Close"])

    # Flag (do not drop) extreme moves so they stay auditable.
    ret = df["Close"].pct_change()
    sigma = ret.std()
    outliers = int((ret.abs() > OUTLIER_SIGMA * sigma).sum()) if sigma and sigma > 0 else 0

    coverage = len(df) / len(listed) if len(listed) else 0.0
    record = {
        "Ticker": ticker,
        "Raw Rows": raw_rows,
        "Clean Rows": len(df),
        "Start": df.index.min().date() if len(df) else None,
        "End": df.index.max().date() if len(df) else None,
        "Duplicate Dates": duplicates,
        "Non-Positive Prices": non_positive,
        "Calendar Gaps Filled": gaps,
        "Return Outliers (>8σ)": outliers,
        "Coverage (%)": round(coverage * 100, 2),
    }
    return df, record


def clean_all(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Clean every symbol. Returns ``({ticker: frame}, quality_report)``."""
    print("\n" + "─" * 72)
    print("  2 · CLEAN & VALIDATE")
    print("─" * 72)

    calendar = _session_calendar(raw)
    cleaned: dict[str, pd.DataFrame] = {}
    records: list[dict] = []

    for ticker, df in raw.items():
        frame, record = clean(df, ticker, calendar)
        if len(frame) < 100:
            print(f"    {ticker}: only {len(frame)} usable rows — excluded")
            continue
        cleaned[ticker] = frame
        records.append(record)

    quality = pd.DataFrame(records).set_index("Ticker")
    print(f"  {len(cleaned)} symbols · {len(calendar):,} sessions "
          f"({calendar.min().date()} → {calendar.max().date()})")
    print(f"  Gaps filled: {int(quality['Calendar Gaps Filled'].sum()):,}  ·  "
          f"duplicates dropped: {int(quality['Duplicate Dates'].sum())}  ·  "
          f"8σ moves flagged: {int(quality['Return Outliers (>8σ)'].sum())}")
    print(f"  Mean calendar coverage: {quality['Coverage (%)'].mean():.2f}%")
    return cleaned, quality


def price_matrix(frames: dict[str, pd.DataFrame], field: str = "Close") -> pd.DataFrame:
    """Wide matrix of one field: rows = dates, columns = tickers."""
    return pd.DataFrame({t: df[field] for t, df in frames.items()}).sort_index()


def return_matrix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide matrix of simple daily returns (decimals)."""
    return price_matrix(frames).pct_change().dropna(how="all")
