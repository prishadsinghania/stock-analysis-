"""Bulk OHLCV ingestion from Yahoo Finance with an on-disk cache.

The universe is ~60 symbols, so the fetcher batches requests, retries the
symbols a batch drops, and falls back to the parquet/CSV cache when the API is
unreachable. That makes the pipeline runnable offline once it has been run once.
"""

from __future__ import annotations

import os
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from src.config import (
    BENCHMARK,
    CACHE_DIR,
    END_DATE,
    RISK_FREE_FALLBACK,
    RISK_FREE_TICKER,
    START_DATE,
    TICKERS,
)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
BATCH_SIZE = 12
MAX_RETRIES = 3

# yfinance keeps a shared sqlite timezone cache; concurrent writers trip a
# "database is locked" error, so pin it to the project cache directory.
os.environ.setdefault("YFINANCE_CACHE_DIR", CACHE_DIR)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker.replace('^', '_')}.csv")


def _read_cache(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    cols = [c for c in OHLCV if c in df.columns]
    return df[cols] if cols else None


def _write_cache(ticker: str, df: pd.DataFrame) -> None:
    df.to_csv(_cache_path(ticker))


def _extract(panel: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Pull a single symbol's OHLCV frame out of a multi-symbol yfinance panel."""
    try:
        if isinstance(panel.columns, pd.MultiIndex):
            available = panel.columns.get_level_values(1)
            if ticker not in set(available):
                return None
            df = panel.xs(ticker, axis=1, level=1)
        else:
            df = panel
        df = df[[c for c in OHLCV if c in df.columns]].dropna(how="all")
    except (KeyError, IndexError):
        return None
    return df if len(df) > 20 else None


def _download(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )


def fetch_prices(
    tickers: list[str] | None = None,
    start: str = START_DATE,
    end: str = END_DATE,
    *,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for every ticker. Returns ``{ticker: DataFrame}``."""
    tickers = list(tickers or TICKERS)
    wanted = tickers + [BENCHMARK] if BENCHMARK not in tickers else list(tickers)

    print(f"\n  Ingesting {len(wanted)} symbols  {start} → {end}")
    data: dict[str, pd.DataFrame] = {}
    pending = list(wanted)

    for attempt in range(1, MAX_RETRIES + 1):
        if not pending:
            break
        if attempt > 1:
            print(f"    retry {attempt - 1}: {len(pending)} symbol(s) — {', '.join(pending[:8])}")
            time.sleep(1.5 * attempt)

        still_missing: list[str] = []
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i : i + BATCH_SIZE]
            try:
                panel = _download(batch, start, end)
            except Exception as exc:  # network/API failure — fall through to retry
                print(f"    batch failed ({type(exc).__name__}); will retry")
                still_missing.extend(batch)
                continue

            for ticker in batch:
                df = _extract(panel, ticker)
                if df is None:
                    still_missing.append(ticker)
                else:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    data[ticker] = df
                    _write_cache(ticker, df)
        pending = still_missing

    # Anything the API never returned falls back to the cache.
    if pending and use_cache:
        for ticker in list(pending):
            cached = _read_cache(ticker)
            if cached is not None:
                data[ticker] = cached
                pending.remove(ticker)
                print(f"    {ticker}: served from cache ({len(cached)} rows)")

    fetched = len(data)
    print(f"  Ingested {fetched}/{len(wanted)} symbols", end="")
    if pending:
        print(f" — unavailable: {', '.join(pending)}")
    else:
        print()
    return data


def fetch_risk_free_rate(start: str = START_DATE, end: str = END_DATE) -> float:
    """Average 13-week T-bill yield over the window, as a decimal (0.03 = 3%)."""
    try:
        import yfinance as yf

        series = yf.download(
            RISK_FREE_TICKER, start=start, end=end,
            auto_adjust=True, progress=False, threads=False,
        )
        close = series["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) > 50:
            rate = float(close.mean()) / 100.0
            _write_cache(RISK_FREE_TICKER, series[[c for c in OHLCV if c in series.columns]])
            print(f"  Risk-free rate: {rate:.2%} (mean {RISK_FREE_TICKER} over window)")
            return rate
    except Exception:
        pass

    cached = _read_cache(RISK_FREE_TICKER)
    if cached is not None and "Close" in cached and len(cached) > 50:
        rate = float(cached["Close"].mean()) / 100.0
        print(f"  Risk-free rate: {rate:.2%} (cached {RISK_FREE_TICKER})")
        return rate

    print(f"  Risk-free rate: {RISK_FREE_FALLBACK:.2%} (fallback)")
    return RISK_FREE_FALLBACK


def fetch_all(tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    print("\n" + "─" * 72)
    print("  1 · INGEST")
    print("─" * 72)
    return fetch_prices(tickers)
