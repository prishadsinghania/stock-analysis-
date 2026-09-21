"""Portfolio construction, optimisation and attribution.

Mean-variance optimisation over the cleaned return matrix: minimum-variance and
maximum-Sharpe portfolios solved with SLSQP, a Monte-Carlo efficient frontier
for context, and a risk-contribution decomposition that shows where the
portfolio's variance actually comes from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import TRADING_DAYS


def annualised_moments(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised mean-return vector and covariance matrix."""
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    return mu, cov


def portfolio_stats(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame,
                    risk_free: float = 0.0) -> dict:
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ cov.values @ w))
    return {
        "return": ret,
        "volatility": vol,
        "sharpe": (ret - risk_free) / vol if vol > 0 else 0.0,
    }


def _solve(objective, n_assets: int, max_weight: float) -> np.ndarray:
    """SLSQP over the long-only simplex with a per-name cap."""
    bounds = [(0.0, max_weight)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    start = np.repeat(1.0 / n_assets, n_assets)
    result = minimize(objective, start, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 800, "ftol": 1e-10})
    weights = result.x if result.success else start
    weights = np.clip(weights, 0, None)
    return weights / weights.sum()


def min_variance(returns: pd.DataFrame, max_weight: float = 0.15) -> pd.Series:
    mu, cov = annualised_moments(returns)
    c = cov.values
    w = _solve(lambda w: w @ c @ w, len(mu), max_weight)
    return pd.Series(w, index=returns.columns)


def max_sharpe(returns: pd.DataFrame, risk_free: float = 0.0, max_weight: float = 0.15) -> pd.Series:
    mu, cov = annualised_moments(returns)
    m, c = mu.values, cov.values

    def negative_sharpe(w):
        vol = np.sqrt(w @ c @ w)
        return 1e6 if vol <= 0 else -((w @ m - risk_free) / vol)

    w = _solve(negative_sharpe, len(mu), max_weight)
    return pd.Series(w, index=returns.columns)


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    n = returns.shape[1]
    return pd.Series(np.repeat(1.0 / n, n), index=returns.columns)


def inverse_volatility(returns: pd.DataFrame) -> pd.Series:
    """Risk-parity approximation: weight inversely to each name's volatility."""
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    inv = 1.0 / vol.replace(0, np.nan)
    return (inv / inv.sum()).fillna(0)


def efficient_frontier(returns: pd.DataFrame, risk_free: float = 0.0,
                       n_points: int = 40, max_weight: float = 0.15) -> pd.DataFrame:
    """Trace the frontier by minimising variance at a series of target returns."""
    mu, cov = annualised_moments(returns)
    m, c = mu.values, cov.values
    n = len(mu)
    bounds = [(0.0, max_weight)] * n
    lo = float(portfolio_stats(min_variance(returns, max_weight).values, mu, cov)["return"])
    hi = float(np.sort(m)[-max(1, int(np.ceil(1 / max_weight))):].mean())

    points = []
    for target in np.linspace(lo, hi, n_points):
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ m - t},
        ]
        res = minimize(lambda w: w @ c @ w, np.repeat(1 / n, n), method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-9})
        if res.success:
            stats = portfolio_stats(res.x, mu, cov, risk_free)
            points.append({"Volatility": stats["volatility"],
                           "Return": stats["return"],
                           "Sharpe": stats["sharpe"]})
    return pd.DataFrame(points).drop_duplicates(subset="Return")


def monte_carlo_cloud(returns: pd.DataFrame, n_portfolios: int = 4000,
                      risk_free: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Random long-only portfolios — the scatter the frontier is the edge of."""
    rng = np.random.default_rng(seed)
    mu, cov = annualised_moments(returns)
    m, c = mu.values, cov.values
    n = len(mu)

    weights = rng.dirichlet(np.ones(n), size=n_portfolios)
    rets = weights @ m
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, c, weights))
    return pd.DataFrame({
        "Return": rets,
        "Volatility": vols,
        "Sharpe": np.divide(rets - risk_free, vols, out=np.zeros_like(rets), where=vols > 0),
    })


def risk_contributions(weights: pd.Series, returns: pd.DataFrame) -> pd.DataFrame:
    """Decompose portfolio variance into each holding's marginal contribution."""
    _, cov = annualised_moments(returns[weights.index])
    w = weights.values
    c = cov.values
    port_vol = float(np.sqrt(w @ c @ w))
    if port_vol <= 0:
        return pd.DataFrame(index=weights.index)

    marginal = c @ w / port_vol
    contribution = w * marginal
    return pd.DataFrame({
        "Weight": weights,
        "Marginal Risk": marginal,
        "Risk Contribution": contribution,
        "Risk Share": contribution / port_vol,
    }).sort_values("Risk Share", ascending=False)


def backtest_weights(returns: pd.DataFrame, weights: pd.Series,
                     rebalance: str = "QE", cost_bps: float = 5.0) -> pd.Series:
    """Daily portfolio returns for fixed target weights, rebalanced periodically.

    Between rebalances the holdings drift with the market; at each rebalance the
    drifted weights are reset to target and the turnover is charged at
    ``cost_bps`` per unit traded.
    """
    cols = [c for c in weights.index if c in returns.columns]
    rets = returns[cols].dropna(how="all").fillna(0)
    target = weights[cols].values
    target = target / target.sum()

    rebalance_dates = set(rets.resample(rebalance).last().index)
    current = target.copy()
    out = []

    for date, row in rets.iterrows():
        gross = float(current @ row.values)
        # Drift: holdings grow with their own return, then renormalise.
        grown = current * (1 + row.values)
        total = grown.sum()
        current = grown / total if total > 0 else target.copy()

        cost = 0.0
        if date in rebalance_dates:
            turnover = float(np.abs(current - target).sum())
            cost = turnover * cost_bps / 10_000
            current = target.copy()

        out.append(gross - cost)

    return pd.Series(out, index=rets.index, name="Portfolio")


def compare_strategies(returns: pd.DataFrame, benchmark: pd.Series,
                       risk_free: float = 0.0, max_weight: float = 0.15,
                       cost_bps: float = 5.0) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Build four candidate portfolios and score them against the benchmark."""
    from src.metrics import summarize

    builders = {
        "Equal Weight": equal_weight(returns),
        "Inverse Volatility": inverse_volatility(returns),
        "Minimum Variance": min_variance(returns, max_weight),
        "Maximum Sharpe": max_sharpe(returns, risk_free, max_weight),
    }

    curves: dict[str, pd.Series] = {}
    rows = []
    for name, weights in builders.items():
        series = backtest_weights(returns, weights, cost_bps=cost_bps)
        curves[name] = series
        row = summarize(series, name, benchmark=benchmark, risk_free=risk_free)
        row["Holdings > 1%"] = int((weights > 0.01).sum())
        row["Max Weight"] = float(weights.max())
        rows.append(row)

    bench_row = summarize(benchmark, "Benchmark (SPY)", benchmark=benchmark, risk_free=risk_free)
    bench_row["Holdings > 1%"] = np.nan
    bench_row["Max Weight"] = np.nan
    rows.append(bench_row)
    curves["Benchmark (SPY)"] = benchmark

    table = pd.DataFrame(rows).set_index("Ticker")
    table.index.name = "Strategy"
    return table, curves


def weight_frame(returns: pd.DataFrame, risk_free: float = 0.0,
                 max_weight: float = 0.15) -> pd.DataFrame:
    """Side-by-side weights for every construction method."""
    return pd.DataFrame({
        "Equal Weight": equal_weight(returns),
        "Inverse Volatility": inverse_volatility(returns),
        "Minimum Variance": min_variance(returns, max_weight),
        "Maximum Sharpe": max_sharpe(returns, risk_free, max_weight),
    })
