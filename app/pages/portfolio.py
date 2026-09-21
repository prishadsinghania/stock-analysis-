"""Portfolio lab — mean-variance construction over the filtered universe."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from app import figures
from app.components import (
    card,
    data_table,
    graph,
    grid,
    note,
    num,
    pct,
    section,
    stat_tile,
    tile_row,
)
from app.state import View, analytics, get_view
from src import portfolio as pf

TITLE = "Portfolio lab"
SUBTITLE = "Mean-variance construction, backtested against the benchmark"

SCORECARD_COLUMNS = [
    "CAGR", "Volatility", "Sharpe", "Sortino", "Max Drawdown",
    "Beta", "Alpha", "Information Ratio", "Holdings > 1%", "Max Weight",
]


@lru_cache(maxsize=24)
def _optimise(window: str, sector_key: str, max_weight: float):
    """Solve the portfolio problem once per filter state — SLSQP is not free."""
    view = get_view(window, [s for s in sector_key.split("|") if s] or None)
    a = analytics()
    returns, bench = view.returns, view.benchmark_returns

    scorecard, curves = pf.compare_strategies(
        returns, bench, a.risk_free, max_weight=max_weight)
    weights = pf.weight_frame(returns, a.risk_free, max_weight=max_weight)
    frontier = pf.efficient_frontier(returns, a.risk_free, n_points=30, max_weight=max_weight)
    cloud = pf.monte_carlo_cloud(returns, 3000, a.risk_free)

    mu, cov = pf.annualised_moments(returns)
    points = {
        name: pf.portfolio_stats(weights[name].values, mu, cov, a.risk_free)
        for name in weights.columns
    }
    contributions = pf.risk_contributions(weights["Maximum Sharpe"], returns)
    return scorecard, curves, weights, frontier, cloud, points, contributions


def render(view: View, mode: str, max_weight: float = 0.15):
    a = analytics()
    sector_key = "|".join(sorted({view.metrics["Sector"].iloc[i] for i in range(len(view.metrics))})) \
        if len(view.metrics["Sector"].unique()) < 11 else ""
    scorecard, curves, weights, frontier, cloud, points, contributions = _optimise(
        view.label, sector_key, max_weight)

    strategies = scorecard.drop(index="Benchmark (SPY)", errors="ignore")
    best = strategies["Sharpe"].idxmax()
    bench = scorecard.loc["Benchmark (SPY)"]

    tiles = tile_row([
        stat_tile(
            f"Best risk-adjusted portfolio · {best}",
            num(strategies.loc[best, "Sharpe"]),
            delta=f"{strategies.loc[best, 'Sharpe'] - bench['Sharpe']:+.2f} Sharpe vs SPY",
            delta_direction=1 if strategies.loc[best, "Sharpe"] >= bench["Sharpe"] else -1,
            note=f"CAGR {pct(strategies.loc[best, 'CAGR'])} · "
                 f"volatility {pct(strategies.loc[best, 'Volatility'])} · "
                 f"max drawdown {pct(strategies.loc[best, 'Max Drawdown'])}",
            hero=True,
        ),
        stat_tile("Benchmark Sharpe", num(bench["Sharpe"]),
                  note=f"SPY · CAGR {pct(bench['CAGR'])}"),
        stat_tile("Equal-weight Sharpe", num(strategies.loc["Equal Weight", "Sharpe"]),
                  note=f"{len(view.tickers)} holdings, no optimisation"),
        stat_tile("Minimum-variance volatility", pct(strategies.loc["Minimum Variance", "Volatility"]),
                  note=f"vs {pct(bench['Volatility'])} for SPY"),
    ])

    return [
        tiles,
        section("Growth of $1", [
            card(
                "Four construction methods against the benchmark",
                graph(figures.strategy_curves(curves, mode, height=420)),
                subtitle="Quarterly rebalancing, 5 bps charged on turnover at each rebalance",
                table=data_table(
                    pd.DataFrame({k: (1 + v.fillna(0)).cumprod() for k, v in curves.items()})
                      .tail(120).iloc[::-1].round(3),
                    table_id="tbl-pf-curves", index_label="Date", page_size=10),
            ),
            card(
                "Scorecard",
                data_table(scorecard[SCORECARD_COLUMNS], table_id="tbl-pf-score",
                           index_label="Strategy", page_size=6),
                subtitle="Same window and benchmark as every other page",
            ),
        ]),
        section("The efficient frontier", [
            card(
                "Risk and return of every achievable mix",
                graph(figures.efficient_frontier_chart(frontier, cloud, points, mode, height=480)),
                subtitle=f"3,000 random long-only portfolios shaded by Sharpe, with the "
                         f"frontier solved by SLSQP under a {max_weight:.0%} per-name cap",
            ),
            note("The maximum-Sharpe portfolio is fitted on the same window it is measured "
                 "on, so its result is in-sample and optimistic. Equal weight and "
                 "inverse volatility use no return forecast and are the honest comparison; "
                 "the frontier is shown as a description of the opportunity set, not a "
                 "recommendation."),
        ]),
        section("Allocation", [
            grid([
                card(
                    "Weights by construction method",
                    graph(figures.weight_comparison(weights, mode, top_n=14, height=420)),
                    subtitle="Top 14 names by largest weight in any single method",
                    table=data_table((weights * 100).round(2).sort_values(
                        "Maximum Sharpe", ascending=False),
                        table_id="tbl-pf-weights", index_label="Ticker", page_size=12),
                ),
                card(
                    "Where the risk actually sits",
                    graph(figures.ranked_bar(
                        (contributions["Risk Share"].head(12) * 100).sort_values(),
                        mode, title="Share of portfolio volatility", height=420)),
                    subtitle="Maximum-Sharpe portfolio — weight and risk contribution "
                             "are not the same thing",
                    table=data_table(contributions.round(4), table_id="tbl-pf-risk",
                                     index_label="Ticker", page_size=12),
                ),
            ]),
        ]),
    ]
