"""Compare — up to eight holdings side by side on one indexed axis."""

from __future__ import annotations

import pandas as pd

from app import figures
from app.components import card, data_table, graph, grid, note, section
from app.state import View
from app.theme import MAX_SERIES

TITLE = "Compare"
SUBTITLE = "Up to eight holdings on a common base"

COMPARE_COLUMNS = [
    "Company", "Sector", "Total Return", "CAGR", "Volatility", "Sharpe",
    "Sortino", "Max Drawdown", "Beta", "Alpha", "Up Capture", "Down Capture",
    "Information Ratio",
]


def assign_slots(selected: list[str], existing: dict[str, int]) -> dict[str, int]:
    """Give each selected ticker a stable colour slot.

    A name keeps its slot for as long as it stays selected, and removing one
    frees its slot without repainting the others — so a reader who learned
    "NVDA is orange" is never misled by a change to the selection.
    """
    slots = {t: s for t, s in existing.items() if t in selected}
    taken = set(slots.values())
    for ticker in selected:
        if ticker in slots:
            continue
        free = next((s for s in range(MAX_SERIES) if s not in taken), len(slots) % MAX_SERIES)
        slots[ticker] = free
        taken.add(free)
    return slots


def render(view: View, mode: str, selected: list[str], slots: dict[str, int]):
    selected = [t for t in selected if t in view.frames][:MAX_SERIES]
    if not selected:
        return [card("Nothing selected",
                     graph(figures.empty(mode, "Pick one or more holdings to compare")))]

    frames = {t: view.frames[t] for t in selected}
    metrics = view.metrics.loc[selected, COMPARE_COLUMNS]

    corr = view.returns[selected].corr() if len(selected) > 1 else pd.DataFrame()
    cumulative = pd.DataFrame({
        t: (1 + view.frames[t]["Return"].fillna(0)).cumprod() - 1 for t in selected
    })

    blocks = [
        section("Relative performance", [
            card(
                "Indexed to 100 at the start of the window",
                graph(figures.indexed_performance(
                    frames, slots, mode, height=440,
                    benchmark=view.benchmark["Close"])),
                subtitle="A common base keeps every series on one axis — prices at "
                         "different levels are still directly comparable",
                table=data_table((cumulative.tail(120).iloc[::-1] * 100).round(2),
                                 table_id="tbl-cmp-perf", index_label="Date", page_size=10),
            ),
        ]),
        section("Side-by-side metrics", [
            card(
                f"{len(selected)} holdings",
                data_table(metrics, table_id="tbl-cmp-metrics",
                           index_label="Ticker", page_size=8),
                subtitle="Same window, same benchmark, same risk-free rate",
            ),
        ]),
    ]

    if len(selected) > 1:
        blocks.append(section("How they move together", [
            grid([
                card(
                    "Return correlation",
                    graph(figures.heatmap(corr, mode, height=380, decimals=2,
                                          colorbar_title="ρ", scale="sequential")),
                    subtitle="Daily return correlation over the selected window",
                    table=data_table(corr.round(3), table_id="tbl-cmp-corr",
                                     index_label="Ticker", page_size=8),
                ),
                card(
                    "Risk versus return",
                    graph(figures.risk_return_scatter(view.metrics, mode,
                                                      highlight=selected, height=380)),
                    subtitle="Selected names emphasised against the full universe",
                ),
            ]),
            note("Correlations above roughly 0.7 mean two holdings are largely the same "
                 "bet — combining them adds position size, not diversification."),
        ]))

    return blocks
