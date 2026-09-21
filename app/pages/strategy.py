"""Strategy lab — rule-based backtests run across the whole universe."""

from __future__ import annotations

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
from app.state import View, analytics
from src.backtest import STRATEGIES, run_strategies, universe_strategy_scan

TITLE = "Strategy lab"
SUBTITLE = "Do the classic technical rules actually beat holding the stock?"


def render(view: View, mode: str, ticker: str):
    a = analytics()
    scan = universe_strategy_scan(view.frames, a.risk_free)
    hold = scan.loc["Buy & Hold"]
    rules = scan.drop(index="Buy & Hold")
    best_rule = rules["Mean Sharpe"].idxmax()

    tiles = tile_row([
        stat_tile(
            "Best rule versus buy and hold",
            f"{rules.loc[best_rule, 'Win Rate vs Hold']:.0%}",
            delta=f"{rules.loc[best_rule, 'Mean Sharpe'] - hold['Mean Sharpe']:+.2f} mean Sharpe",
            delta_direction=1 if rules.loc[best_rule, "Mean Sharpe"] >= hold["Mean Sharpe"] else -1,
            note=f"{best_rule} beats buy and hold on "
                 f"{rules.loc[best_rule, 'Win Rate vs Hold'] * len(view.frames):.0f} of "
                 f"{len(view.frames)} holdings",
            hero=True,
        ),
        stat_tile("Buy and hold mean Sharpe", num(hold["Mean Sharpe"]),
                  note=f"Mean CAGR {pct(hold['Mean CAGR'])}"),
        stat_tile("Backtests run", f"{len(STRATEGIES) * len(view.frames):,}",
                  note=f"{len(STRATEGIES)} rules × {len(view.frames)} holdings"),
        stat_tile("Shallowest drawdown", f"{rules['Mean Max Drawdown'].idxmax()}",
                  note=f"{pct(rules['Mean Max Drawdown'].max())} average — "
                       f"rules cut losses by staying out of the market"),
    ])

    ticker_scorecard = None
    if ticker in view.frames:
        ticker_scorecard, curves = run_strategies(view.frames[ticker], ticker, a.risk_free)
        curve_figure = figures.strategy_curves(curves, mode, height=380)
    else:
        curve_figure = figures.empty(mode, "Selected holding is outside the current filter")

    return [
        tiles,
        section("Across the whole universe", [
            card(
                "Rule scorecard",
                data_table(scan, table_id="tbl-strat-scan",
                           index_label="Strategy", page_size=5),
                subtitle=f"Every rule run on all {len(view.frames)} holdings and averaged — "
                         f"a rule that works on one name is a coincidence",
            ),
            grid([
                card(
                    "Mean Sharpe by rule",
                    graph(figures.ranked_bar(scan["Mean Sharpe"], mode,
                                             title="Mean Sharpe", suffix="",
                                             height=280, decimals=2)),
                    subtitle="Averaged across every holding",
                ),
                card(
                    "Mean maximum drawdown by rule",
                    graph(figures.diverging_bar(scan["Mean Max Drawdown"] * 100, mode,
                                                suffix="%", height=280)),
                    subtitle="Timing rules sit in cash during selloffs, so they draw down less",
                ),
            ]),
            note("Signals are generated at each day's close and traded at the next close, so "
                 "no decision uses information that was unavailable when it was made. "
                 "Turnover is charged at 5 bps per side. No parameter was tuned on this data."),
        ]),
        section(f"On a single holding · {ticker}", [
            card(
                "Growth of $1 by rule",
                graph(curve_figure),
                subtitle="Use the holding selector in the filter row to change the name",
            ),
            card(
                "Scorecard",
                data_table(ticker_scorecard, table_id="tbl-strat-single",
                           index_label="Strategy", page_size=5)
                if ticker_scorecard is not None else
                graph(figures.empty(mode, "No data")),
                subtitle="Time invested shows how much of the window each rule was "
                         "actually exposed to the market",
            ),
        ]),
    ]
