"""Technical indicators computed per symbol.

Every function takes and returns a DataFrame so the set can be composed in any
order; ``enrich`` applies the full stack used by the pipeline and dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    ATR_WINDOW,
    BB_STD,
    BB_WINDOW,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MA_200,
    MA_LONG,
    MA_SHORT,
    RSI_WINDOW,
    TRADING_DAYS,
    VOL_WINDOW,
)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"]).diff()
    df["Cumulative_Return"] = (1 + df["Return"].fillna(0)).cumprod() - 1
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    for window in (MA_SHORT, MA_LONG, MA_200):
        df[f"SMA_{window}"] = close.rolling(window).mean()
    for span in (MA_SHORT, MA_LONG):
        df[f"EMA_{span}"] = close.ewm(span=span, adjust=False).mean()
    # Distance from the long-term trend, a cleaner signal than the raw level.
    df["Pct_From_SMA200"] = (close / df[f"SMA_{MA_200}"] - 1) * 100
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = BB_WINDOW, n_std: float = BB_STD) -> pd.DataFrame:
    df = df.copy()
    mid = df["Close"].rolling(window).mean()
    sd = df["Close"].rolling(window).std()
    df["BB_Middle"] = mid
    df["BB_Upper"] = mid + n_std * sd
    df["BB_Lower"] = mid - n_std * sd
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / mid * 100
    # 0 = at the lower band, 1 = at the upper band.
    span = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Position"] = ((df["Close"] - df["BB_Lower"]) / span.replace(0, np.nan)).clip(-0.5, 1.5)
    return df


def add_rsi(df: pd.DataFrame, window: int = RSI_WINDOW) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing: com = window - 1 makes ewm equivalent to his average.
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"RSI_{window}"] = (100 - 100 / (1 + rs)).fillna(100 * (avg_gain > 0))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fast = df["Close"].ewm(span=MACD_FAST, adjust=False).mean()
    slow = df["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"] = fast - slow
    df["MACD_Signal"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.DataFrame:
    """Average True Range — volatility in price units, gap-aware."""
    df = df.copy()
    prev_close = df["Close"].shift()
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df[f"ATR_{window}"] = true_range.ewm(com=window - 1, adjust=False).mean()
    df["ATR_Pct"] = df[f"ATR_{window}"] / df["Close"] * 100
    return df


def add_volatility(df: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    df = df.copy()
    df[f"Volatility_{window}d"] = (
        df["Return"].rolling(window).std() * np.sqrt(TRADING_DAYS) * 100
    )
    return df


def add_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    peak = df["Close"].cummax()
    df["Drawdown"] = (df["Close"] / peak - 1) * 100
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]
    df["Dollar_Volume"] = df["Close"] * df["Volume"]
    # On-Balance Volume: cumulative volume signed by the day's direction.
    df["OBV"] = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()
    return df


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean/Categorical trading-signal columns used by the screener."""
    df = df.copy()
    short, long = df[f"SMA_{MA_LONG}"], df[f"SMA_{MA_200}"]
    # Undefined until both averages have warmed up — keep those days out of
    # the cross logic so the start of the sample is not read as a signal.
    warm = short.notna() & long.notna()
    above = (short > long) & warm
    prev = above.shift(1, fill_value=False).astype(bool)
    df["Golden_Cross"] = above & ~prev & warm
    df["Death_Cross"] = ~above & prev & warm
    df["Trend"] = np.where(~warm, "Warm-up", np.where(above, "Uptrend", "Downtrend"))

    rsi = df[f"RSI_{RSI_WINDOW}"]
    df["RSI_Zone"] = pd.cut(
        rsi, bins=[-0.1, 30, 45, 55, 70, 100.1],
        labels=["Oversold", "Weak", "Neutral", "Strong", "Overbought"],
    )
    df["BB_Breakout"] = np.where(
        df["Close"] > df["BB_Upper"], "Above Upper",
        np.where(df["Close"] < df["BB_Lower"], "Below Lower", "In Band"),
    )
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full indicator stack to one symbol."""
    for step in (
        add_returns, add_moving_averages, add_bollinger_bands, add_rsi,
        add_macd, add_atr, add_volatility, add_drawdown, add_volume_features,
        add_signals,
    ):
        df = step(df)
    return df


def enrich_all(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    print("\n" + "─" * 72)
    print("  3 · INDICATORS")
    print("─" * 72)
    enriched = {t: enrich(df) for t, df in frames.items()}
    sample = next(iter(enriched.values()))
    n_indicators = len(sample.columns) - 5
    print(f"  {n_indicators} indicator columns × {len(enriched)} symbols "
          f"= {n_indicators * len(enriched):,} derived series")
    print("  SMA/EMA · Bollinger · RSI · MACD · ATR · OBV · rolling vol · drawdown · signals")
    return enriched
