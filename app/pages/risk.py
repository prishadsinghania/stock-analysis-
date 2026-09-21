"""Risk and correlation — how the universe behaves together and under stress."""

from __future__ import annotations

import numpy as np
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
from app.state import View
from src import eda

TITLE = "Risk & correlation"
SUBTITLE = "Tail risk, co-movement and behaviour by regime"


def render(view: View, mode: str):
    metrics = view.metrics
    returns = view.returns

    corr = returns.corr()
    off_diagonal = corr.values[np.triu_indices_from(corr.values, k=1)]
    sector_corr = eda.sector_correlation(returns)

    tiles = tile_row([
        stat_tile("Average pairwise correlation", num(float(off_diagonal.mean())),
                  note=f"Across {len(corr)} holdings · range "
                       f"{off_diagonal.min():.2f} to {off_diagonal.max():.2f}", hero=True),
        stat_tile("Worst single drawdown", pct(metrics["Max Drawdown"].min()),
                  note=str(metrics["Max Drawdown"].idxmin())),
        stat_tile("Median VaR 95 (1-day)", pct(metrics["VaR 95"].median()),
                  note="Loss exceeded on 1 day in 20"),
        stat_tile("Median expected shortfall", pct(metrics["CVaR 95"].median()),
                  note="Average loss on those days"),
        stat_tile("Beta range",
                  f"{metrics['Beta'].min():.2f} – {metrics['Beta'].max():.2f}",
                  note=f"{metrics['Beta'].idxmin()} most defensive · "
                       f"{metrics['Beta'].idxmax()} most benchmark-sensitive"),
    ])

    ordered = eda.cluster_order(corr)
    clustered = corr.loc[ordered, ordered]

    rolling_corr = eda.average_pairwise_correlation(returns, window=90)
    regimes = eda.regime_matrix(returns) * 100
    dispersion = eda.dispersion_over_time(returns)

    tail_table = metrics[[
        "Sector", "Volatility", "Max Drawdown", "VaR 95", "CVaR 95",
        "VaR 99", "CVaR 99", "Skew", "Excess Kurtosis", "Down Capture",
    ]].sort_values("CVaR 99")

    return [
        tiles,
        section("Co-movement", [
            card(
                "Correlation matrix, hierarchically clustered",
                graph(figures.heatmap(clustered, mode, height=760, show_text=False,
                                      decimals=2, colorbar_title="ρ",
                                      scale="sequential")),
                subtitle="Rows and columns are ordered by average-linkage clustering, so "
                         "blocks of names that move together sit next to each other",
                table=data_table(clustered.round(3), table_id="tbl-risk-corr",
                                 index_label="Ticker", page_size=12),
            ),
            grid([
                card(
                    "Sector correlation",
                    graph(figures.heatmap(sector_corr, mode, height=460, decimals=2,
                                          colorbar_title="ρ", scale="sequential")),
                    subtitle="Correlation between equal-weighted sector return series",
                    table=data_table(sector_corr.round(3), table_id="tbl-risk-sector-corr",
                                     index_label="Sector", page_size=11),
                ),
                card(
                    "Average pairwise correlation over time",
                    graph(figures.rolling_line(rolling_corr, mode, label="90-day correlation",
                                               y_title="Mean pairwise ρ", height=460)),
                    subtitle="Correlation rises in selloffs — diversification is thinnest "
                             "exactly when it is needed most",
                ),
            ]),
        ]),
        section("Behaviour by market regime", [
            card(
                "Sector total return within each regime",
                graph(figures.heatmap(regimes, mode, height=460, value_suffix="%",
                                      decimals=0, colorbar_title="Return %",
                                      scale="returns")),
                subtitle="Six documented market episodes from the pre-COVID bull run to today",
                table=data_table(regimes.round(1), table_id="tbl-risk-regime",
                                 index_label="Sector", page_size=11),
            ),
            note("Regime windows are fixed calendar periods chosen from well-documented "
                 "market events, not fitted to the data."),
        ]),
        section("Dispersion and tail risk", [
            card(
                "Cross-sectional return dispersion",
                graph(figures.dispersion_band(dispersion, mode, height=320)),
                subtitle="Rolling one-month return of the median holding, with the "
                         "10th-to-90th percentile band — the room stock selection has to work in",
            ),
            card(
                "Tail-risk table",
                data_table(tail_table, table_id="tbl-risk-tails",
                           index_label="Ticker", page_size=15),
                subtitle="Sorted by 99% expected shortfall — the average loss on the "
                         "worst 1% of days",
            ),
            note("Value at Risk here is historical, not parametric: it reads the actual "
                 "return distribution rather than assuming a normal one. Given the measured "
                 "excess kurtosis, a normal assumption would understate these losses."),
        ]),
    ]
