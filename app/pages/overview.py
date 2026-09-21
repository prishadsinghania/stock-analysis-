"""Market overview — the landing view: one hero number, then the leaderboards."""

from __future__ import annotations

import pandas as pd

from app import figures
from app.components import (
    card,
    compact,
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
from app.state import View, analytics

TITLE = "Market overview"
SUBTITLE = "Where the universe stands, and which names lead it"


def _sparkline_series(view: View) -> pd.Series:
    """Twelve monthly points of the equal-weighted universe, for the tiles."""
    monthly = (1 + view.returns.mean(axis=1)).resample("ME").prod() - 1
    return monthly.tail(12)


def render(view: View, mode: str):
    a = analytics()
    metrics = view.metrics
    equal_weighted = view.returns.mean(axis=1)
    total = float((1 + equal_weighted).prod() - 1)
    bench_total = float((1 + view.benchmark_returns).prod() - 1)
    spark = _sparkline_series(view)

    beat_benchmark = int((metrics["Total Return"] > bench_total).sum())
    positive = int((metrics["Total Return"] > 0).sum())
    uptrend = int((view.signals["Trend"] == "Uptrend").sum())

    hero = tile_row([
        stat_tile(
            "Equal-weighted universe return",
            pct(total, decimals=1, signed=True),
            delta=f"{(total - bench_total) * 100:+.1f} pts vs SPY",
            delta_direction=1 if total >= bench_total else -1,
            note=f"{view.label} window · {len(metrics)} holdings · {view.sessions:,} sessions",
            figure=figures.sparkline(spark, mode, positive=total >= 0),
            hero=True,
        ),
        stat_tile("Names beating SPY", f"{beat_benchmark}",
                  note=f"of {len(metrics)} · {beat_benchmark / max(len(metrics), 1):.0%} of the universe"),
        stat_tile("Names with positive return", f"{positive}",
                  note=f"{positive / max(len(metrics), 1):.0%} of the universe"),
        stat_tile("In a confirmed uptrend", f"{uptrend}",
                  note="SMA-50 above SMA-200 as of the last session"),
    ], columns=None)

    scoreboard = tile_row([
        stat_tile("Median CAGR", pct(metrics["CAGR"].median()),
                  note=f"SPY benchmark {pct(_bench_cagr(view))}"),
        stat_tile("Median volatility", pct(metrics["Volatility"].median()),
                  note="Annualised, from daily returns"),
        stat_tile("Median Sharpe", num(metrics["Sharpe"].median()),
                  note=f"Risk-free {a.risk_free:.2%} (13-week T-bill)"),
        stat_tile("Median max drawdown", pct(metrics["Max Drawdown"].median()),
                  note="Worst peak-to-trough in window"),
        stat_tile("Median beta", num(metrics["Beta"].median()),
                  note="Versus SPY"),
        stat_tile("Avg daily dollar volume", compact(metrics["Avg Dollar Volume"].median(), prefix="$"),
                  note="Median name, trailing year"),
    ], columns=None)

    top = metrics.nlargest(10, "CAGR")["CAGR"] * 100
    bottom = metrics.nsmallest(10, "CAGR")["CAGR"] * 100
    leaders = pd.concat([top, bottom]).sort_values()

    sector_cagr = view.sectors["CAGR"] * 100

    leaderboard_table = metrics[[
        "Company", "Sector", "Total Return", "CAGR", "Volatility",
        "Sharpe", "Max Drawdown", "Beta",
    ]].copy()

    return [
        hero,
        section(
            "Universe scorecard",
            [scoreboard],
            description=f"{view.start:%d %b %Y} – {view.end:%d %b %Y}, "
                        f"{len(metrics)} US large caps across {metrics['Sector'].nunique()} GICS sectors.",
        ),
        section("Leaders and laggards", [
            grid([
                card(
                    "Top and bottom 10 by CAGR",
                    graph(figures.diverging_bar(leaders, mode, suffix="%", height=520)),
                    subtitle="Annualised growth rate over the selected window",
                    table=data_table(
                        metrics[["Company", "Sector", "CAGR", "Sharpe"]].sort_values("CAGR", ascending=False),
                        table_id="tbl-cagr", index_label="Ticker", page_size=12),
                ),
                card(
                    "Sector performance",
                    graph(figures.ranked_bar(sector_cagr, mode, title="Mean CAGR", height=340)),
                    subtitle="Equal-weighted mean CAGR of each sector's members",
                    table=data_table(view.sectors, table_id="tbl-sector", index_label="Sector", page_size=11),
                ),
            ], columns="1.15fr 1fr"),
        ]),
        section("Return dispersion", [
            card(
                "CAGR distribution within each sector",
                graph(figures.sector_dispersion(metrics, mode, height=440)),
                subtitle="Every dot is one holding — the spread inside a sector is usually "
                         "wider than the gap between sectors",
                table=data_table(
                    metrics[["Sector", "CAGR", "Volatility", "Sharpe"]].sort_values(["Sector", "CAGR"]),
                    table_id="tbl-dispersion", index_label="Ticker", page_size=12),
            ),
        ]),
        section("Full leaderboard", [
            card(
                f"All {len(metrics)} holdings",
                data_table(leaderboard_table, table_id="tbl-leaderboard",
                           index_label="Ticker", page_size=20,
                           sort_by=[{"column_id": "CAGR", "direction": "desc"}]),
                subtitle="Sortable and filterable — type in a column's filter box to narrow the list",
            ),
        ]),
        section("Findings", [
            insight_list(a.headlines),
            note("Metrics are computed from adjusted daily closes. Returns are price "
                 "returns including the dividend adjustment Yahoo Finance applies; they "
                 "exclude taxes and trading costs."),
        ]),
    ]


def _bench_cagr(view: View) -> float:
    from src.metrics import cagr

    return cagr(view.benchmark_returns)
