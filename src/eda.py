"""Exploratory data analysis over the cross-section.

These routines answer the questions that come before any model: what does the
return distribution actually look like, does the calendar matter, how stable
are correlations, and how did each regime treat each sector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from src.config import TRADING_DAYS, sector_of

# Named windows used for the regime table. Each is a well-documented market
# episode, so per-regime behaviour is interpretable rather than arbitrary.
REGIMES: dict[str, tuple[str, str]] = {
    "Pre-COVID bull": ("2019-01-01", "2020-02-19"),
    "COVID crash": ("2020-02-20", "2020-03-23"),
    "Stimulus recovery": ("2020-03-24", "2021-12-31"),
    "2022 rate shock": ("2022-01-01", "2022-10-12"),
    "AI-led rally": ("2022-10-13", "2024-12-31"),
    "Recent": ("2025-01-01", "2030-12-31"),
}


# ── Distribution ─────────────────────────────────────────────────────────────
def distribution_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol moments plus a formal normality test."""
    rows = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        jb = stats.jarque_bera(r)
        rows.append({
            "Ticker": ticker,
            "Sector": sector_of(ticker),
            "Mean (bps/day)": r.mean() * 10_000,
            "Std (bps/day)": r.std() * 10_000,
            "Skew": r.skew(),
            "Excess Kurtosis": r.kurt(),
            "Jarque-Bera p": jb.pvalue,
            "Normal at 5%": jb.pvalue > 0.05,
            "Days beyond ±3σ": int((r.abs() > 3 * r.std()).sum()),
            "Expected beyond ±3σ": round(len(r) * 0.0027, 1),
        })
    return pd.DataFrame(rows).set_index("Ticker")


def fat_tail_summary(returns: pd.DataFrame) -> dict:
    """How far the cross-section departs from the normal assumption."""
    table = distribution_table(returns)
    return {
        "symbols": len(table),
        "normal_at_5pct": int(table["Normal at 5%"].sum()),
        "mean_excess_kurtosis": float(table["Excess Kurtosis"].mean()),
        "mean_negative_skew": float((table["Skew"] < 0).mean()),
        "observed_3sigma_days": int(table["Days beyond ±3σ"].sum()),
        "expected_3sigma_days": float(table["Expected beyond ±3σ"].sum()),
    }


# ── Calendar effects ─────────────────────────────────────────────────────────
def monthly_seasonality(returns: pd.DataFrame) -> pd.DataFrame:
    """Average monthly return per calendar month, equal-weighted across names."""
    monthly = (1 + returns).resample("ME").prod() - 1
    by_month = monthly.groupby(monthly.index.month).mean()
    by_month.index = pd.Index(
        [pd.Timestamp(2000, m, 1).strftime("%b") for m in by_month.index], name="Month"
    )
    return by_month


def day_of_week_effect(returns: pd.DataFrame) -> pd.DataFrame:
    """Mean daily return by weekday, with a t-test against zero."""
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    equal_weighted = returns.mean(axis=1)
    rows = []
    for day, label in enumerate(names):
        sample = equal_weighted[equal_weighted.index.dayofweek == day]
        if sample.empty:
            continue
        t_stat, p_value = stats.ttest_1samp(sample, 0.0)
        rows.append({
            "Weekday": label,
            "Mean (bps)": sample.mean() * 10_000,
            "Positive Share": (sample > 0).mean(),
            "Observations": len(sample),
            "t-stat": t_stat,
            "p-value": p_value,
            "Significant at 5%": p_value < 0.05,
        })
    return pd.DataFrame(rows).set_index("Weekday")


def monthly_heatmap(returns: pd.Series) -> pd.DataFrame:
    """Year × month grid of returns for one series."""
    monthly = (1 + returns.dropna()).resample("ME").prod() - 1
    grid = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.strftime("%b"),
        "Return": monthly.values,
    })
    order = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]
    pivot = grid.pivot_table(index="Year", columns="Month", values="Return")
    return pivot.reindex(columns=[m for m in order if m in pivot.columns])


# ── Correlation structure ────────────────────────────────────────────────────
def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def sector_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Correlation between equal-weighted sector return series."""
    sectors = pd.DataFrame({
        sector: returns[[c for c in returns.columns if sector_of(c) == sector]].mean(axis=1)
        for sector in sorted({sector_of(c) for c in returns.columns})
    })
    return sectors.corr()


def cluster_order(corr: pd.DataFrame) -> list[str]:
    """Order symbols by hierarchical clustering so blocks surface in a heatmap."""
    if len(corr) < 3:
        return list(corr.columns)
    distance = np.clip(1 - corr.values, 0, 2)
    np.fill_diagonal(distance, 0.0)
    linkage = hierarchy.linkage(squareform(distance, checks=False), method="average")
    return [corr.columns[i] for i in hierarchy.leaves_list(linkage)]


def rolling_correlation_to_benchmark(returns: pd.DataFrame, benchmark: pd.Series,
                                     window: int = 90) -> pd.Series:
    """Average pairwise correlation to the benchmark — a risk-on/risk-off gauge."""
    aligned = returns.join(benchmark.rename("_bench"), how="inner")
    bench = aligned.pop("_bench")
    return aligned.rolling(window).corr(bench).mean(axis=1).dropna()


def average_pairwise_correlation(returns: pd.DataFrame, window: int = 90) -> pd.Series:
    """Mean off-diagonal correlation over time — diversification availability."""
    out = {}
    index = returns.index
    for end in range(window, len(index), 5):   # every 5 sessions keeps this cheap
        block = returns.iloc[end - window : end]
        corr = block.corr().values
        n = corr.shape[0]
        out[index[end]] = (corr.sum() - n) / (n * (n - 1))
    return pd.Series(out).dropna()


# ── Regimes ──────────────────────────────────────────────────────────────────
def regime_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised return and volatility per sector within each market regime."""
    rows = []
    for regime, (start, end) in REGIMES.items():
        window = returns.loc[start:end]
        if window.empty:
            continue
        for sector in sorted({sector_of(c) for c in returns.columns}):
            cols = [c for c in returns.columns if sector_of(c) == sector]
            series = window[cols].mean(axis=1)
            total = float((1 + series).prod() - 1)
            rows.append({
                "Regime": regime,
                "Sector": sector,
                "Total Return": total,
                "Annualised Vol": float(series.std() * np.sqrt(TRADING_DAYS)),
                "Days": len(series),
            })
    return pd.DataFrame(rows)


def regime_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Sector × regime grid of total returns."""
    long = regime_table(returns)
    pivot = long.pivot(index="Sector", columns="Regime", values="Total Return")
    return pivot.reindex(columns=[r for r in REGIMES if r in pivot.columns])


# ── Liquidity & dispersion ───────────────────────────────────────────────────
def dispersion_over_time(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Cross-sectional spread of returns — how much stock picking could matter."""
    rolling = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
    return pd.DataFrame({
        "Median": rolling.median(axis=1),
        "P10": rolling.quantile(0.10, axis=1),
        "P90": rolling.quantile(0.90, axis=1),
        "Spread": rolling.quantile(0.90, axis=1) - rolling.quantile(0.10, axis=1),
    }).dropna()


def eda_headlines(returns: pd.DataFrame, benchmark: pd.Series) -> list[str]:
    """Plain-language findings the dashboard and report both display."""
    tails = fat_tail_summary(returns)
    dow = day_of_week_effect(returns)
    months = monthly_seasonality(returns).mean(axis=1)
    corr = correlation_matrix(returns)
    off_diag = corr.values[np.triu_indices_from(corr.values, k=1)]
    sectors = sector_correlation(returns)
    sector_off = sectors.values[np.triu_indices_from(sectors.values, k=1)]

    best_month, worst_month = months.idxmax(), months.idxmin()
    significant_days = dow.index[dow["Significant at 5%"]].tolist()

    return [
        f"{tails['symbols'] - tails['normal_at_5pct']} of {tails['symbols']} symbols reject "
        f"normality at 5% (Jarque-Bera); mean excess kurtosis is "
        f"{tails['mean_excess_kurtosis']:.1f} versus 0 for a normal distribution.",

        f"±3σ days occur {tails['observed_3sigma_days'] / tails['expected_3sigma_days']:.1f}× more often "
        f"than a normal distribution predicts ({tails['observed_3sigma_days']:,} observed vs "
        f"{tails['expected_3sigma_days']:.0f} expected) — tail risk is understated by variance alone.",

        f"Average pairwise correlation across the universe is {off_diag.mean():.2f}, and "
        f"{sector_off.mean():.2f} between sectors — diversification within US large caps is limited.",

        f"Seasonality is weak but visible: {best_month} is the strongest calendar month "
        f"({months.max() * 100:+.2f}% average) and {worst_month} the weakest "
        f"({months.min() * 100:+.2f}%).",

        (f"Day-of-week effects are statistically significant for {', '.join(significant_days)} at 5%."
         if significant_days else
         "No weekday shows a statistically significant mean return at 5% — the day-of-week effect is noise here."),
    ]
