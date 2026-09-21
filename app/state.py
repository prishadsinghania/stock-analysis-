"""Shared dashboard state: the analytics bundle and cached filtered views.

The filter row (sector + time window) scopes every chart on the page, so the
slice is computed once per filter combination and memoised rather than being
recomputed per figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from src import screener
from src.config import sector_of
from src.metrics import build_metrics_table
from src.pipeline import Analytics, build_analytics

ANALYTICS: Analytics | None = None

# Preset windows offered in the filter row, in years back from the last session.
WINDOWS: dict[str, float | None] = {
    "1Y": 1, "3Y": 3, "5Y": 5, "Max": None,
}
DEFAULT_WINDOW = "Max"


def load(refresh: bool = False) -> Analytics:
    global ANALYTICS
    ANALYTICS = build_analytics(refresh=refresh)
    return ANALYTICS


def analytics() -> Analytics:
    if ANALYTICS is None:
        return load()
    return ANALYTICS


@dataclass
class View:
    """The universe as scoped by the current filters."""

    frames: dict[str, pd.DataFrame]
    returns: pd.DataFrame
    benchmark: pd.DataFrame
    metrics: pd.DataFrame
    factors: pd.DataFrame
    signals: pd.DataFrame
    sectors: pd.DataFrame
    label: str
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def tickers(self) -> list[str]:
        return list(self.frames)

    @property
    def benchmark_returns(self) -> pd.Series:
        return self.benchmark["Return"].dropna()

    @property
    def sessions(self) -> int:
        return len(self.returns)


def window_bounds(window: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a = analytics()
    end = a.end
    years = WINDOWS.get(window, None)
    start = a.start if years is None else max(a.start, end - pd.DateOffset(years=years))
    return pd.Timestamp(start), pd.Timestamp(end)


@lru_cache(maxsize=48)
def _build_view(window: str, sector_key: str) -> View:
    a = analytics()
    start, end = window_bounds(window)
    sectors = tuple(s for s in sector_key.split("|") if s)

    tickers = [t for t in a.tickers if not sectors or sector_of(t) in sectors]
    frames = {t: a.frames[t].loc[start:end] for t in tickers}
    frames = {t: df for t, df in frames.items() if len(df) > 30}
    benchmark = a.benchmark.loc[start:end]
    bench_returns = benchmark["Return"].dropna()

    metrics = build_metrics_table(frames, benchmark_returns=bench_returns, risk_free=a.risk_free)
    returns = pd.DataFrame({t: df["Return"] for t, df in frames.items()}).dropna(how="all")

    return View(
        frames=frames,
        returns=returns,
        benchmark=benchmark,
        metrics=metrics,
        factors=screener.build_factor_table(frames, metrics),
        signals=screener.current_signals(frames),
        sectors=screener.sector_rollup(metrics),
        label=window,
        start=start,
        end=end,
    )


def get_view(window: str | None, sectors: list[str] | None) -> View:
    """Memoised view for the current filter state."""
    window = window or DEFAULT_WINDOW
    key = "|".join(sorted(sectors)) if sectors else ""
    return _build_view(window, key)
