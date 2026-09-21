"""Security detail — one name, its price action, indicators and risk profile."""

from __future__ import annotations

import pandas as pd

from app import figures
from app.components import (
    badge,
    card,
    compact,
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
from src.backtest import run_strategies
from src.config import RSI_WINDOW, name_of, sector_of

TITLE = "Security detail"
SUBTITLE = "Price action, indicators and risk for a single holding"

_TREND_VARIANT = {"Uptrend": "good", "Downtrend": "critical", "Warm-up": "neutral"}
_RSI_VARIANT = {"Overbought": "warning", "Oversold": "warning",
                "Strong": "good", "Weak": "serious", "Neutral": "neutral"}


def render(view: View, mode: str, ticker: str):
    a = analytics()
    if ticker not in view.frames:
        return [card("Not in the current filter",
                     graph(figures.empty(mode, f"{ticker} is outside the selected sectors")))]

    df = view.frames[ticker]
    row = view.metrics.loc[ticker]
    signal = view.signals.loc[ticker]
    last_close = float(df["Close"].iloc[-1])

    spark = ((1 + df["Return"]).resample("ME").prod() - 1).tail(12)

    header_tiles = tile_row([
        stat_tile(
            f"{name_of(ticker)} · last close",
            f"${last_close:,.2f}",
            delta=f"{pct(row['Total Return'], signed=True)} over {view.label.lower()}",
            delta_direction=1 if row["Total Return"] >= 0 else -1,
            note=f"{sector_of(ticker)} · 52-week range "
                 f"${row['52W Low']:,.0f}–${row['52W High']:,.0f}",
            figure=figures.sparkline(spark, mode, positive=row["Total Return"] >= 0),
            hero=True,
        ),
        stat_tile("CAGR", pct(row["CAGR"], signed=True), note="Annualised growth rate"),
        stat_tile("Volatility", pct(row["Volatility"]), note="Annualised standard deviation"),
        stat_tile("Sharpe", num(row["Sharpe"]), note=f"Risk-free {a.risk_free:.2%}"),
    ])

    risk_tiles = tile_row([
        stat_tile("Sortino", num(row["Sortino"]), note="Downside-risk adjusted"),
        stat_tile("Calmar", num(row["Calmar"]), note="CAGR ÷ max drawdown"),
        stat_tile("Max drawdown", pct(row["Max Drawdown"]),
                  note=_recovery_note(row)),
        stat_tile("Beta", num(row["Beta"]), note=f"R² {row['R Squared']:.2f} vs SPY"),
        stat_tile("Alpha", pct(row["Alpha"], signed=True), note="Annualised, CAPM"),
        stat_tile("VaR 95 (1-day)", pct(row["VaR 95"]),
                  note=f"Expected shortfall {pct(row['CVaR 95'])}"),
        stat_tile("Hit rate", pct(row["Hit Rate"]), note="Share of up days"),
        stat_tile("Avg dollar volume", compact(row["Avg Dollar Volume"], prefix="$"),
                  note="Trailing year"),
    ])

    state = html_state(signal)
    scorecard, curves = run_strategies(df, ticker, a.risk_free)

    return [
        header_tiles,
        section("Risk profile", [risk_tiles]),
        section("Current technical state", [
            card(
                "Signals as of the last session",
                state,
                subtitle="Each signal is labelled in text — colour is a secondary cue only",
            ),
        ]),
        section("Price and trend", [
            card(
                f"{ticker} price with moving averages",
                graph(figures.price_with_moving_averages(df, ticker, mode, height=420)),
                subtitle="SMA-20, SMA-50 and SMA-200 on a single price axis",
                table=data_table(
                    df[["Close", "SMA_20", "SMA_50", "SMA_200"]].tail(120).iloc[::-1].round(2),
                    table_id="tbl-sec-price", index_label="Date", page_size=10),
            ),
            grid([
                card("Bollinger bands", graph(figures.bollinger(df, ticker, mode)),
                     subtitle="SMA-20 ± 2 standard deviations"),
                card("Drawdown", graph(figures.drawdown_curve(df, mode, height=340)),
                     subtitle="Percentage below the running peak"),
            ]),
        ]),
        section("Momentum indicators", [
            grid([
                card(f"RSI {RSI_WINDOW}", graph(figures.rsi(df, mode, height=300)),
                     subtitle="Wilder's smoothing; 70/30 reference bands"),
                card("MACD", graph(figures.macd(df, mode, height=300)),
                     subtitle="12/26 EMA difference with a 9-period signal line"),
            ]),
            card("Trading volume", graph(figures.volume(df, mode, height=240)),
                 subtitle="Daily shares traded against the 20-day average"),
        ]),
        section("Return distribution", [
            grid([
                card(
                    "Daily returns versus a normal distribution",
                    graph(figures.return_distribution(df["Return"], mode, height=320)),
                    subtitle=f"Skew {row['Skew']:.2f} · excess kurtosis {row['Excess Kurtosis']:.1f} · "
                             f"Jarque-Bera p {row['Jarque Bera P']:.3g}",
                ),
                card(
                    "Monthly returns by year",
                    graph(_monthly_heatmap_figure(df["Return"], mode)),
                    subtitle="Each cell is one calendar month",
                ),
            ]),
        ]),
        section("Rule-based strategies on this name", [
            card(
                "Strategy scorecard",
                data_table(scorecard, table_id="tbl-sec-strategies",
                           index_label="Strategy", page_size=5),
                subtitle="Signals generated at the close are traded at the next close; "
                         "turnover charged at 5 bps per side",
            ),
            note("A single-name backtest is a small sample. The Strategy lab page runs the same "
                 "rules across all holdings, which is the test that matters."),
        ]),
    ]


def _recovery_note(row: pd.Series) -> str:
    recovery = row.get("Recovery Days")
    if recovery is None or pd.isna(recovery):
        return f"{int(row['Drawdown Days'])} days down, not yet recovered"
    return f"{int(row['Drawdown Days'])} days down, {int(recovery)} to recover"


def _monthly_heatmap_figure(returns: pd.Series, mode: str):
    from src.eda import monthly_heatmap

    grid_df = monthly_heatmap(returns) * 100
    if grid_df.empty:
        return figures.empty(mode, "Not enough history for a monthly grid")
    return figures.heatmap(grid_df, mode, height=320, value_suffix="%",
                           decimals=1, colorbar_title="Return %", scale="returns")


def html_state(signal: pd.Series):
    from dash import html

    items = [
        ("Trend", signal["Trend"], _TREND_VARIANT.get(signal["Trend"], "neutral")),
        ("RSI zone", f"{signal['RSI Zone']} ({signal[f'RSI {RSI_WINDOW}']:.0f})",
         _RSI_VARIANT.get(signal["RSI Zone"], "neutral")),
        ("Bollinger", signal["Bollinger"],
         "warning" if signal["Bollinger"] != "In Band" else "neutral"),
        ("MACD histogram", f"{signal['MACD Hist']:+.2f}",
         "good" if signal["MACD Hist"] > 0 else "critical"),
        ("Versus SMA-200", f"{signal['% From SMA200']:+.1f}%",
         "good" if signal["% From SMA200"] > 0 else "critical"),
        ("Drawdown from peak", f"{signal['Drawdown'] * 100:.1f}%",
         "critical" if signal["Drawdown"] < -0.2 else "neutral"),
        ("Volume vs 20-day", f"{signal['Volume vs 20d']:.2f}×", "neutral"),
    ]
    return html.Div([
        html.Div([
            html.Span(label, className="signal-label"),
            badge(str(value), variant),
        ], className="signal-item")
        for label, value, variant in items
    ], className="signal-grid")
