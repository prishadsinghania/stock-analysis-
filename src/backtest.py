"""Rule-based strategy backtests, scored against buy-and-hold.

Signals are generated on day *t* and traded at the day *t+1* close, so no
decision uses information that was not available when it was made. Turnover is
charged at ``COST_BPS`` per side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import COST_BPS, MA_200, MA_LONG, RSI_WINDOW, TRADING_DAYS


def _apply_positions(df: pd.DataFrame, position: pd.Series, cost_bps: float) -> pd.Series:
    """Turn a 0/1 exposure series into a net daily return stream."""
    # Lag by one day: a signal observed at today's close is traded tomorrow.
    held = position.shift(1).fillna(0)
    gross = held * df["Return"].fillna(0)
    trades = held.diff().abs().fillna(0)
    return gross - trades * cost_bps / 10_000


def sma_crossover(df: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.Series:
    """Long while SMA-50 is above SMA-200, flat otherwise."""
    position = (df[f"SMA_{MA_LONG}"] > df[f"SMA_{MA_200}"]).astype(float)
    position[df[f"SMA_{MA_200}"].isna()] = 0.0
    return _apply_positions(df, position, cost_bps)


def rsi_mean_reversion(df: pd.DataFrame, entry: float = 30, exit_: float = 55,
                       cost_bps: float = COST_BPS) -> pd.Series:
    """Buy oversold, sell once momentum normalises — a contrarian rule."""
    rsi = df[f"RSI_{RSI_WINDOW}"]
    position = pd.Series(np.nan, index=df.index)
    position[rsi < entry] = 1.0
    position[rsi > exit_] = 0.0
    position = position.ffill().fillna(0)
    return _apply_positions(df, position, cost_bps)


def trend_plus_momentum(df: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.Series:
    """Long only when the long-term trend and the MACD both agree."""
    position = (
        (df["Close"] > df[f"SMA_{MA_200}"]) & (df["MACD_Hist"] > 0)
    ).astype(float)
    position[df[f"SMA_{MA_200}"].isna()] = 0.0
    return _apply_positions(df, position, cost_bps)


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return df["Return"].fillna(0)


STRATEGIES = {
    "Buy & Hold": buy_and_hold,
    "SMA 50/200 Crossover": sma_crossover,
    "RSI Mean Reversion": rsi_mean_reversion,
    "Trend + MACD": trend_plus_momentum,
}


def _exposure(df: pd.DataFrame, name: str) -> float:
    """Share of days the strategy was invested — context for its return."""
    if name == "Buy & Hold":
        return 1.0
    if name == "SMA 50/200 Crossover":
        pos = (df[f"SMA_{MA_LONG}"] > df[f"SMA_{MA_200}"]).astype(float)
    elif name == "Trend + MACD":
        pos = ((df["Close"] > df[f"SMA_{MA_200}"]) & (df["MACD_Hist"] > 0)).astype(float)
    else:
        rsi = df[f"RSI_{RSI_WINDOW}"]
        pos = pd.Series(np.nan, index=df.index)
        pos[rsi < 30] = 1.0
        pos[rsi > 55] = 0.0
        pos = pos.ffill().fillna(0)
    return float(pos.mean())


def run_strategies(df: pd.DataFrame, ticker: str, risk_free: float = 0.0,
                   cost_bps: float = COST_BPS) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Score every rule on one symbol. Returns ``(scorecard, equity_curves)``."""
    from src.metrics import cagr, hit_rate, max_drawdown, sharpe_ratio, sortino_ratio, volatility

    curves: dict[str, pd.Series] = {}
    rows = []
    for name, fn in STRATEGIES.items():
        series = fn(df) if name == "Buy & Hold" else fn(df, cost_bps=cost_bps)
        curves[name] = series
        trades = 0
        if name != "Buy & Hold":
            exposure = _exposure(df, name)
            pos = (series != 0).astype(int)
            trades = int(pos.diff().abs().fillna(0).sum())
        else:
            exposure = 1.0
        rows.append({
            "Strategy": name,
            "Total Return": float((1 + series).prod() - 1),
            "CAGR": cagr(series),
            "Volatility": volatility(series),
            "Sharpe": sharpe_ratio(series, risk_free),
            "Sortino": sortino_ratio(series, risk_free),
            "Max Drawdown": max_drawdown(series),
            "Hit Rate": hit_rate(series),
            "Time Invested": exposure,
            "Trades": trades,
        })

    scorecard = pd.DataFrame(rows).set_index("Strategy")
    scorecard.attrs["ticker"] = ticker
    return scorecard, curves


def universe_strategy_scan(frames: dict[str, pd.DataFrame], risk_free: float = 0.0,
                           cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Run every rule across every symbol and average the outcome per rule.

    A rule that only works on one name is a coincidence; this is the check.
    """
    from src.metrics import cagr, max_drawdown, sharpe_ratio

    records = []
    for ticker, df in frames.items():
        for name, fn in STRATEGIES.items():
            series = fn(df) if name == "Buy & Hold" else fn(df, cost_bps=cost_bps)
            records.append({
                "Ticker": ticker,
                "Strategy": name,
                "CAGR": cagr(series),
                "Sharpe": sharpe_ratio(series, risk_free),
                "Max Drawdown": max_drawdown(series),
            })

    long = pd.DataFrame(records)
    hold = long[long["Strategy"] == "Buy & Hold"].set_index("Ticker")["Sharpe"]
    long["Beats Buy & Hold"] = long.apply(
        lambda r: r["Sharpe"] > hold.get(r["Ticker"], np.inf), axis=1
    )

    summary = long.groupby("Strategy").agg(
        **{
            "Mean CAGR": ("CAGR", "mean"),
            "Median CAGR": ("CAGR", "median"),
            "Mean Sharpe": ("Sharpe", "mean"),
            "Mean Max Drawdown": ("Max Drawdown", "mean"),
            "Win Rate vs Hold": ("Beats Buy & Hold", "mean"),
        }
    )
    return summary.sort_values("Mean Sharpe", ascending=False)
