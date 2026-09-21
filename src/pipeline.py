"""End-to-end pipeline: ingest → clean → enrich → score → optimise.

Both the CLI (`main.py`) and the dashboard build their state from
``build_analytics()``, so the two always show the same numbers. Results are
memoised to a pickle so a dashboard restart does not re-download the universe.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field

import pandas as pd

from src import eda, screener
from src.analyzer import enrich_all
from src.backtest import universe_strategy_scan
from src.config import BENCHMARK, CACHE_DIR, END_DATE, START_DATE
from src.data_cleaner import clean_all, return_matrix
from src.data_fetcher import fetch_prices, fetch_risk_free_rate
from src.metrics import build_metrics_table
from src.portfolio import compare_strategies, weight_frame

BUNDLE_PATH = os.path.join(CACHE_DIR, "analytics.pkl")
BUNDLE_TTL_HOURS = 12


@dataclass
class Analytics:
    """Everything the reports and dashboard read from."""

    frames: dict[str, pd.DataFrame]              # enriched OHLCV per symbol
    benchmark: pd.DataFrame                      # enriched benchmark frame
    returns: pd.DataFrame                        # wide daily-return matrix
    metrics: pd.DataFrame                        # 38 metrics per symbol
    factors: pd.DataFrame                        # factor exposures + composite score
    signals: pd.DataFrame                        # latest technical state
    sectors: pd.DataFrame                        # sector roll-up
    quality: pd.DataFrame                        # data-quality audit
    portfolios: pd.DataFrame                     # strategy scorecard
    portfolio_curves: dict[str, pd.Series]       # strategy return streams
    weights: pd.DataFrame                        # weights per construction method
    strategy_scan: pd.DataFrame                  # rule backtests across the universe
    headlines: list[str]                         # EDA findings in plain language
    risk_free: float
    built_at: float = field(default_factory=time.time)

    @property
    def tickers(self) -> list[str]:
        return list(self.frames)

    @property
    def benchmark_returns(self) -> pd.Series:
        return self.benchmark["Return"].dropna()

    @property
    def start(self):
        return self.returns.index.min()

    @property
    def end(self):
        return self.returns.index.max()

    @property
    def sessions(self) -> int:
        return len(self.returns)

    @property
    def observations(self) -> int:
        """Price observations behind the analysis — the dataset's headline size."""
        return int(sum(len(df) for df in self.frames.values()))


def _rule(label: str) -> None:
    print("\n" + "─" * 72)
    print(f"  {label}")
    print("─" * 72)


def build_analytics(*, use_cache: bool = True, refresh: bool = False) -> Analytics:
    """Run the full pipeline (or load a recent cached run)."""
    if use_cache and not refresh and os.path.exists(BUNDLE_PATH):
        age_hours = (time.time() - os.path.getmtime(BUNDLE_PATH)) / 3600
        if age_hours < BUNDLE_TTL_HOURS:
            try:
                with open(BUNDLE_PATH, "rb") as fh:
                    bundle = pickle.load(fh)
                print(f"  Loaded cached analytics ({age_hours:.1f}h old) — "
                      f"{len(bundle.frames)} symbols")
                return bundle
            except Exception:
                print("  Cached analytics unreadable; rebuilding.")

    started = time.time()
    print("\n" + "═" * 72)
    print("  EQUITY ANALYTICS PIPELINE")
    print(f"  Window {START_DATE} → {END_DATE}   ·   benchmark {BENCHMARK}")
    print("═" * 72)

    raw = fetch_prices()
    risk_free = fetch_risk_free_rate()
    cleaned, quality = clean_all(raw)
    enriched = enrich_all(cleaned)

    benchmark = enriched.pop(BENCHMARK)
    frames = enriched
    returns = return_matrix(frames)
    bench_returns = benchmark["Return"].dropna()

    _rule("4 · METRICS & FACTOR SCREEN")
    metrics = build_metrics_table(frames, benchmark_returns=bench_returns, risk_free=risk_free)
    factors = screener.build_factor_table(frames, metrics)
    signals = screener.current_signals(frames)
    sectors = screener.sector_rollup(metrics)
    print(f"  {metrics.shape[1]} metrics × {metrics.shape[0]} symbols "
          f"= {metrics.shape[0] * metrics.shape[1]:,} data points")
    print(f"  Top composite score: {factors.index[0]} ({factors['Composite Score'].iloc[0]:.2f})")

    _rule("5 · PORTFOLIO CONSTRUCTION")
    portfolios, curves = compare_strategies(returns, bench_returns, risk_free)
    weights = weight_frame(returns, risk_free)
    best = portfolios["Sharpe"].drop("Benchmark (SPY)", errors="ignore").idxmax()
    print(f"  4 construction methods backtested with quarterly rebalancing")
    print(f"  Best risk-adjusted: {best} (Sharpe {portfolios.loc[best, 'Sharpe']:.2f} "
          f"vs {portfolios.loc['Benchmark (SPY)', 'Sharpe']:.2f} for {BENCHMARK})")

    _rule("6 · STRATEGY SCAN & EDA")
    strategy_scan = universe_strategy_scan(frames, risk_free)
    headlines = eda.eda_headlines(returns, bench_returns)
    print(f"  4 rules × {len(frames)} symbols = {4 * len(frames)} backtests")
    for line in headlines:
        print(f"    • {line}")

    bundle = Analytics(
        frames=frames, benchmark=benchmark, returns=returns, metrics=metrics,
        factors=factors, signals=signals, sectors=sectors, quality=quality,
        portfolios=portfolios, portfolio_curves=curves, weights=weights,
        strategy_scan=strategy_scan, headlines=headlines, risk_free=risk_free,
    )

    try:
        with open(BUNDLE_PATH, "wb") as fh:
            pickle.dump(bundle, fh)
    except Exception as exc:
        print(f"  (analytics cache not written: {type(exc).__name__})")

    print(f"\n  Pipeline complete in {time.time() - started:.1f}s")
    return bundle
