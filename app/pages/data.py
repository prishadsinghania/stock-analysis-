"""Data quality — the audit trail behind every other page."""

from __future__ import annotations

import pandas as pd
from dash import dcc, html

from app.components import card, data_table, note, pct, section, stat_tile, tile_row
from app.state import View, analytics
from src.config import BENCHMARK, END_DATE, START_DATE

TITLE = "Data quality"
SUBTITLE = "Provenance, validation checks and the full metric table"


def render(view: View, mode: str):
    a = analytics()
    quality = a.quality

    tiles = tile_row([
        stat_tile("Price observations", f"{a.observations:,}",
                  note=f"{len(a.tickers)} holdings × "
                       f"{a.observations // max(len(a.tickers), 1):,} sessions "
                       f"(benchmark held separately)",
                  hero=True),
        stat_tile("Calendar coverage", f"{quality['Coverage (%)'].mean():.2f}%",
                  note="Share of expected sessions present after alignment"),
        stat_tile("Gaps forward-filled", f"{int(quality['Calendar Gaps Filled'].sum()):,}",
                  note="Holidays and halts, carried from the last print"),
        stat_tile("Duplicate dates removed", f"{int(quality['Duplicate Dates'].sum()):,}",
                  note="Kept the first record for each session"),
        stat_tile("Moves beyond 8σ", f"{int(quality['Return Outliers (>8σ)'].sum()):,}",
                  note=f"Flagged for review, deliberately not removed · "
                       f"{int(quality['Non-Positive Prices'].sum()):,} invalid prints nulled"),
    ])

    return [
        tiles,
        section("How the data is built", [
            card("Pipeline", _pipeline_steps(),
                 subtitle=f"{START_DATE} → {END_DATE} · benchmark {BENCHMARK} · "
                          f"risk-free {a.risk_free:.2%} from the 13-week T-bill"),
        ]),
        section("Per-symbol validation", [
            card(
                "Data-quality audit",
                data_table(quality, table_id="tbl-data-quality",
                           index_label="Ticker", page_size=15),
                subtitle="Every symbol is checked for duplicates, invalid prices, "
                         "calendar gaps and extreme moves before it reaches the analysis",
            ),
            note("Extreme returns are flagged, never deleted. Real markets gap, and "
                 "removing the tails would flatter every risk metric on the other pages."),
        ]),
        section("Full metric table", [
            card(
                f"All {a.metrics.shape[1]} metrics × {a.metrics.shape[0]} holdings",
                html.Div([
                    html.Button("Download CSV", id="download-metrics-btn", className="btn"),
                    dcc.Download(id="download-metrics"),
                    data_table(a.metrics, table_id="tbl-data-metrics",
                               index_label="Ticker", page_size=20),
                ]),
                subtitle="The complete output of the metrics engine, sortable and filterable",
            ),
        ]),
    ]


def _pipeline_steps():
    steps = [
        ("Ingest", "Batched Yahoo Finance download with per-symbol retry and an on-disk "
                   "cache, so a rerun works offline."),
        ("Clean", "Align every symbol to a shared session calendar, drop duplicates, null "
                  "non-positive prices, forward-fill holidays, flag 8σ moves."),
        ("Enrich", "31 indicator columns per symbol: moving averages, Bollinger bands, RSI, "
                   "MACD, ATR, OBV, rolling volatility, drawdown and trading signals."),
        ("Measure", "38 risk and return metrics per holding, including CAPM beta and alpha, "
                    "capture ratios, historical VaR and expected shortfall."),
        ("Score", "Four factor exposures z-scored across the cross-section and blended into "
                  "a composite, plus a sector-neutral variant."),
        ("Construct", "Mean-variance optimisation under a per-name cap, backtested with "
                      "quarterly rebalancing and transaction costs."),
    ]
    return html.Ol([
        html.Li([html.Span(name, className="step-name"), html.Span(text)])
        for name, text in steps
    ], className="pipeline-steps")
