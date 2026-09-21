"""Exploration — the distributional and calendar questions that precede modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import figures
from app.components import (
    card,
    data_table,
    graph,
    grid,
    insight_list,
    note,
    num,
    pct,
    section,
    stat_tile,
    tile_row,
)
from app.state import View
from src import eda
from src.config import sector_of

TITLE = "Exploration"
SUBTITLE = "Distribution shape, calendar effects and what they rule out"


def render(view: View, mode: str):
    returns = view.returns
    tails = eda.fat_tail_summary(returns)
    distribution = eda.distribution_table(returns)
    weekday = eda.day_of_week_effect(returns)
    seasonality = eda.monthly_seasonality(returns)
    equal_weighted = returns.mean(axis=1)

    tiles = tile_row([
        stat_tile(
            "Symbols whose returns are not normally distributed",
            f"{tails['symbols'] - tails['normal_at_5pct']} of {tails['symbols']}",
            note=f"Jarque-Bera at 5% · mean excess kurtosis "
                 f"{tails['mean_excess_kurtosis']:.1f} against 0 for a normal distribution",
            hero=True,
        ),
        stat_tile("±3σ days observed", f"{tails['observed_3sigma_days']:,}",
                  delta=f"{tails['observed_3sigma_days'] / tails['expected_3sigma_days']:.1f}× expected",
                  delta_direction=-1,
                  note=f"A normal distribution predicts {tails['expected_3sigma_days']:.0f}"),
        stat_tile("Negatively skewed names", f"{tails['mean_negative_skew']:.0%}",
                  note="Large moves are more often down than up"),
        stat_tile("Observations analysed", f"{returns.size:,}",
                  note=f"{len(returns):,} sessions × {returns.shape[1]} holdings"),
    ])

    monthly_mean = seasonality.mean(axis=1) * 100
    # Aggregate to sector level: a 60-row grid with a label in every cell is
    # unreadable, and the question here is about sectors anyway.
    seasonality_grid = (
        seasonality.T.groupby(seasonality.columns.map(sector_of)).mean() * 100
    )

    return [
        tiles,
        section("Distribution shape", [
            grid([
                card(
                    "Equal-weighted universe returns versus a normal fit",
                    graph(figures.return_distribution(equal_weighted, mode, height=340)),
                    subtitle="The peak is taller and the tails are fatter than the normal "
                             "curve — variance alone understates how bad the bad days are",
                ),
                card(
                    "Skew and excess kurtosis by holding",
                    graph(_moment_scatter(distribution, mode)),
                    subtitle="Every holding sits above zero excess kurtosis; most are "
                             "negatively skewed",
                    table=data_table(distribution.round(3), table_id="tbl-eda-dist",
                                     index_label="Ticker", page_size=12),
                ),
            ]),
        ]),
        section("Calendar effects", [
            grid([
                card(
                    "Average return by calendar month",
                    graph(figures.diverging_bar(monthly_mean, mode, suffix="%",
                                                height=380, decimals=2)),
                    subtitle="Equal-weighted across holdings, averaged over every year "
                             "in the window",
                ),
                card(
                    "Day-of-week effect",
                    data_table(
                        weekday[["Mean (bps)", "t-stat", "p-value", "Significant at 5%"]].round(4),
                        table_id="tbl-eda-dow", index_label="Weekday", page_size=5),
                    subtitle="Mean daily return with a one-sample t-test against zero — "
                             "the p-value column is the point",
                ),
            ], columns="1.2fr 1fr"),
            card(
                "Month by sector",
                graph(figures.heatmap(seasonality_grid, mode, height=420,
                                      value_suffix="%", decimals=1,
                                      colorbar_title="Mean return %",
                                      scale="returns")),
                subtitle="Equal-weighted mean monthly return of each sector, by calendar month",
                table=data_table(seasonality_grid.round(2), table_id="tbl-eda-season",
                                 index_label="Sector", page_size=11),
            ),
            note("Calendar effects here are descriptive. With five weekdays tested at the "
                 "5% level, one apparent result in twenty is expected by chance alone, so "
                 "a single significant weekday would not be evidence of a tradable effect."),
        ]),
        section("Findings", [
            insight_list(eda.eda_headlines(returns, view.benchmark_returns)),
        ]),
    ]


def _moment_scatter(distribution: pd.DataFrame, mode: str):
    """Skew against excess kurtosis — one series, so one colour."""
    import plotly.graph_objects as go

    from app.theme import MARKER_SIZE, layout, rgba, series_color, tokens

    t = tokens(mode)
    color = series_color(mode, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=distribution["Skew"], y=distribution["Excess Kurtosis"], mode="markers",
        marker=dict(size=MARKER_SIZE, color=rgba(color, 0.55),
                    line=dict(width=2, color=t["surface"])),
        customdata=np.stack([distribution.index, distribution["Sector"]], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]}"
                       "<br>Skew %{x:.2f}<br>Excess kurtosis %{y:.1f}<extra></extra>"),
        showlegend=False,
    ))
    fig.update_layout(**layout(mode, height=340, showlegend=False))
    fig.add_vline(x=0, line=dict(color=t["axis"], width=1))
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1))
    fig.update_xaxes(title_text="Skew")
    fig.update_yaxes(title_text="Excess kurtosis")

    extreme = distribution["Excess Kurtosis"].idxmax()
    fig.add_annotation(
        x=distribution.loc[extreme, "Skew"], y=distribution.loc[extreme, "Excess Kurtosis"],
        text=extreme, showarrow=False, yshift=14,
        font=dict(size=11, color=t["text_secondary"]),
    )
    return fig
