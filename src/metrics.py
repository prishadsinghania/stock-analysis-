"""Risk and return metrics, including benchmark-relative statistics.

All functions take simple daily returns as decimals (0.01 = +1%) and return
plain floats, so they compose into the cross-sectional summary table that the
screener, reports and dashboard all read from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import TRADING_DAYS, VAR_LEVELS


# ── Return ───────────────────────────────────────────────────────────────────
def total_return(returns: pd.Series) -> float:
    return float((1 + returns.dropna()).prod() - 1)


def years_of(returns: pd.Series) -> float:
    idx = returns.dropna().index
    if len(idx) < 2:
        return 0.0
    return (idx[-1] - idx[0]).days / 365.25


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate, annualised on the actual elapsed window."""
    years = years_of(returns)
    growth = 1 + total_return(returns)
    if years <= 0 or growth <= 0:
        return 0.0
    return float(growth ** (1 / years) - 1)


# ── Risk ─────────────────────────────────────────────────────────────────────
def volatility(returns: pd.Series) -> float:
    return float(returns.dropna().std() * np.sqrt(TRADING_DAYS))


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    """Annualised std-dev of returns below a minimum acceptable return."""
    excess = returns.dropna() - mar / TRADING_DAYS
    below = excess[excess < 0]
    if below.empty:
        return 0.0
    return float(np.sqrt((below ** 2).mean()) * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    curve = (1 + returns.dropna()).cumprod()
    return float((curve / curve.cummax() - 1).min())


def drawdown_profile(returns: pd.Series) -> dict:
    """Depth, length and recovery of the worst peak-to-trough episode."""
    curve = (1 + returns.dropna()).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1
    if dd.empty:
        return {"max_drawdown": 0.0, "drawdown_days": 0, "recovery_days": None, "current_drawdown": 0.0}

    trough = dd.idxmin()
    peak_date = curve.loc[:trough].idxmax()
    recovered = curve.loc[trough:][curve.loc[trough:] >= curve.loc[peak_date]]
    recovery_date = recovered.index[0] if len(recovered) else None

    return {
        "max_drawdown": float(dd.min()),
        "drawdown_days": int((trough - peak_date).days),
        "recovery_days": int((recovery_date - trough).days) if recovery_date is not None else None,
        "current_drawdown": float(dd.iloc[-1]),
    }


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Historical one-day VaR: the loss exceeded (1-level) of the time."""
    r = returns.dropna()
    return float(np.percentile(r, (1 - level) * 100)) if len(r) else 0.0


def conditional_var(returns: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall — the mean loss on days worse than VaR."""
    r = returns.dropna()
    if r.empty:
        return 0.0
    cutoff = np.percentile(r, (1 - level) * 100)
    tail = r[r <= cutoff]
    return float(tail.mean()) if len(tail) else float(cutoff)


# ── Risk-adjusted ────────────────────────────────────────────────────────────
def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    vol = volatility(returns)
    return float((cagr(returns) - risk_free) / vol) if vol > 0 else 0.0


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    dd = downside_deviation(returns, risk_free)
    return float((cagr(returns) - risk_free) / dd) if dd > 0 else 0.0


def calmar_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    mdd = abs(max_drawdown(returns))
    return float((cagr(returns) - risk_free) / mdd) if mdd > 0 else 0.0


# ── Benchmark-relative ───────────────────────────────────────────────────────
def _align(returns: pd.Series, benchmark: pd.Series) -> tuple[pd.Series, pd.Series]:
    joined = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    return joined.iloc[:, 0], joined.iloc[:, 1]


def beta_alpha(returns: pd.Series, benchmark: pd.Series, risk_free: float = 0.0) -> dict:
    """OLS of excess asset returns on excess benchmark returns (CAPM)."""
    r, b = _align(returns, benchmark)
    if len(r) < 30:
        return {"beta": np.nan, "alpha": np.nan, "r_squared": np.nan, "correlation": np.nan}

    daily_rf = risk_free / TRADING_DAYS
    slope, intercept, r_value, _, _ = stats.linregress(b - daily_rf, r - daily_rf)
    return {
        "beta": float(slope),
        "alpha": float(intercept * TRADING_DAYS),   # annualised Jensen's alpha
        "r_squared": float(r_value ** 2),
        "correlation": float(r_value),
    }


def tracking_error(returns: pd.Series, benchmark: pd.Series) -> float:
    r, b = _align(returns, benchmark)
    return float((r - b).std() * np.sqrt(TRADING_DAYS))


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    te = tracking_error(returns, benchmark)
    if te <= 0:
        return 0.0
    r, b = _align(returns, benchmark)
    return float((cagr(r) - cagr(b)) / te)


def capture_ratios(returns: pd.Series, benchmark: pd.Series) -> dict:
    """Share of benchmark gains captured on up days vs losses on down days."""
    r, b = _align(returns, benchmark)
    up, down = b > 0, b < 0
    upside = float(r[up].mean() / b[up].mean()) if up.any() and b[up].mean() != 0 else np.nan
    downside = float(r[down].mean() / b[down].mean()) if down.any() and b[down].mean() != 0 else np.nan
    return {"up_capture": upside, "down_capture": downside}


# ── Distribution ─────────────────────────────────────────────────────────────
def distribution_stats(returns: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) < 8:
        return {"skew": np.nan, "excess_kurtosis": np.nan, "jarque_bera_p": np.nan, "is_normal": False}
    jb_p = float(stats.jarque_bera(r).pvalue)
    return {
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurt()),
        "jarque_bera_p": jb_p,
        "is_normal": bool(jb_p > 0.05),
    }


def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    return float((r > 0).mean()) if len(r) else 0.0


def best_worst_month(returns: pd.Series) -> dict:
    monthly = (1 + returns.dropna()).resample("ME").prod() - 1
    if monthly.empty:
        return {"best_month": np.nan, "worst_month": np.nan, "positive_months": np.nan}
    return {
        "best_month": float(monthly.max()),
        "worst_month": float(monthly.min()),
        "positive_months": float((monthly > 0).mean()),
    }


# ── Cross-sectional summary ──────────────────────────────────────────────────
def summarize(
    returns: pd.Series,
    ticker: str,
    *,
    benchmark: pd.Series | None = None,
    risk_free: float = 0.0,
    prices: pd.Series | None = None,
    dollar_volume: pd.Series | None = None,
) -> dict:
    """One row of the master metrics table for a single symbol."""
    r = returns.dropna()
    row: dict = {"Ticker": ticker}

    if prices is not None and len(prices.dropna()):
        p = prices.dropna()
        row["Start Price"] = float(p.iloc[0])
        row["End Price"] = float(p.iloc[-1])
        row["52W High"] = float(p.tail(TRADING_DAYS).max())
        row["52W Low"] = float(p.tail(TRADING_DAYS).min())

    row.update({
        "Total Return": total_return(r),
        "CAGR": cagr(r),
        "Volatility": volatility(r),
        "Downside Dev": downside_deviation(r, risk_free),
        "Sharpe": sharpe_ratio(r, risk_free),
        "Sortino": sortino_ratio(r, risk_free),
        "Calmar": calmar_ratio(r, risk_free),
        "Hit Rate": hit_rate(r),
    })
    row.update({k.replace("_", " ").title(): v for k, v in drawdown_profile(r).items()})
    row.update({k.replace("_", " ").title(): v for k, v in best_worst_month(r).items()})
    row.update({k.replace("_", " ").title(): v for k, v in distribution_stats(r).items()})

    for level in VAR_LEVELS:
        tag = int(level * 100)
        row[f"VaR {tag}"] = value_at_risk(r, level)
        row[f"CVaR {tag}"] = conditional_var(r, level)

    if benchmark is not None:
        row.update({k.replace("_", " ").title(): v for k, v in beta_alpha(r, benchmark, risk_free).items()})
        row.update({k.replace("_", " ").title(): v for k, v in capture_ratios(r, benchmark).items()})
        row["Tracking Error"] = tracking_error(r, benchmark)
        row["Information Ratio"] = information_ratio(r, benchmark)

    if dollar_volume is not None and len(dollar_volume.dropna()):
        row["Avg Dollar Volume"] = float(dollar_volume.dropna().tail(TRADING_DAYS).mean())

    return row


def build_metrics_table(
    frames: dict[str, pd.DataFrame],
    *,
    benchmark_returns: pd.Series | None = None,
    risk_free: float = 0.0,
) -> pd.DataFrame:
    """Master metrics table — one row per symbol, ~30 columns."""
    from src.config import name_of, sector_of

    rows = []
    for ticker, df in frames.items():
        rows.append(summarize(
            df["Return"], ticker,
            benchmark=benchmark_returns,
            risk_free=risk_free,
            prices=df["Close"],
            dollar_volume=df.get("Dollar_Volume"),
        ))

    table = pd.DataFrame(rows).set_index("Ticker")
    table.insert(0, "Sector", [sector_of(t) for t in table.index])
    table.insert(0, "Company", [name_of(t) for t in table.index])
    return table.sort_values("CAGR", ascending=False)
