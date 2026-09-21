"""Reusable dashboard building blocks.

Presentation rules kept in one place: stat tiles follow the
label / value / delta / trend contract, every chart card can reveal the table
view that backs it, and no value is reachable only through a tooltip.
"""

from __future__ import annotations

import pandas as pd
from dash import dash_table, dcc, html

from app.theme import tokens

GRAPH_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
}


def compact(value: float, *, prefix: str = "", suffix: str = "", decimals: int = 1) -> str:
    """1,284 · 12.9K · $4.2M — auto-compacted for stat tiles."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    magnitude = abs(value)
    for threshold, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= threshold:
            return f"{prefix}{value / threshold:,.{decimals}f}{unit}{suffix}"
    return f"{prefix}{value:,.{decimals}f}{suffix}"


def pct(value: float, *, decimals: int = 1, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:{'+' if signed else ''},.{decimals}f}%"


def num(value: float, *, decimals: int = 2, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:{'+' if signed else ''},.{decimals}f}"


def stat_tile(label: str, value: str, *, delta: str | None = None,
              delta_direction: int = 0, note: str | None = None,
              figure=None, hero: bool = False) -> html.Div:
    """label · value · optional delta · optional sparkline."""
    children = [html.Div(label, className="tile-label")]
    children.append(html.Div(value, className="tile-value tile-value--hero" if hero else "tile-value"))

    if delta is not None:
        direction = {1: "up", -1: "down"}.get(delta_direction, "flat")
        children.append(html.Div([
            html.Span(className=f"tile-arrow tile-arrow--{direction}"),
            html.Span(delta),
        ], className=f"tile-delta tile-delta--{direction}"))

    if note:
        children.append(html.Div(note, className="tile-note"))

    if figure is not None:
        children.append(dcc.Graph(figure=figure, config=GRAPH_CONFIG, className="tile-spark"))

    return html.Div(children, className="tile tile--hero" if hero else "tile")


def tile_row(tiles: list, columns: int | None = None) -> html.Div:
    style = {"gridTemplateColumns": f"repeat({columns}, minmax(0, 1fr))"} if columns else None
    return html.Div(tiles, className="tile-row", style=style)


# Columns held as decimal fractions — shown as percentages so a reader never
# has to work out whether 0.722 means 72% or 0.7%.
PERCENT_COLUMNS = {
    "Total Return", "CAGR", "Volatility", "Downside Dev", "Hit Rate",
    "Max Drawdown", "Current Drawdown", "Best Month", "Worst Month",
    "Positive Months", "Alpha", "Tracking Error", "Drawdown", "Momentum 12-1",
    "VaR 95", "VaR 99", "CVaR 95", "CVaR 99",
    "Mean CAGR", "Median CAGR", "Mean Max Drawdown", "Win Rate vs Hold",
    "Time Invested", "Max Weight", "Weight", "Risk Share", "Risk Contribution",
    "Marginal Risk", "Annualised Vol",
}
RATIO_COLUMNS = {
    "Sharpe", "Sortino", "Calmar", "Beta", "R Squared", "Correlation",
    "Up Capture", "Down Capture", "Information Ratio", "Skew",
    "Excess Kurtosis", "Composite Score", "Sector Score", "Mean Sharpe",
    "Volume vs 20d", "MACD Hist",
}
PRICE_COLUMNS = {"Start Price", "End Price", "52W High", "52W Low", "Close",
                 "SMA_20", "SMA_50", "SMA_200", "BB_Upper", "BB_Lower", "BB_Middle"}
INTEGER_COLUMNS = {"Drawdown Days", "Recovery Days", "Trades", "Holdings", "Holdings > 1%",
                   "Rank", "Sector Rank", "Raw Rows", "Clean Rows", "Duplicate Dates",
                   "Non-Positive Prices", "Calendar Gaps Filled", "Observations",
                   "Crosses (60d)", "Days", "Days beyond ±3σ"}


def _column_spec(name: str, series: pd.Series) -> dict:
    spec = {"name": name, "id": name}
    if not pd.api.types.is_numeric_dtype(series):
        return spec
    spec["type"] = "numeric"
    if name in PERCENT_COLUMNS:
        spec["format"] = {"specifier": "+,.2%" if "Return" in name or name == "Alpha" else ",.2%"}
    elif name in PRICE_COLUMNS:
        spec["format"] = {"specifier": "$,.2f"}
    elif name.endswith("(%)") or name.endswith("(bps)") or name.endswith("(bps/day)"):
        spec["format"] = {"specifier": ",.2f"}   # already scaled, do not re-scale
    elif name in INTEGER_COLUMNS:
        spec["format"] = {"specifier": ",.0f"}
    elif name in RATIO_COLUMNS:
        spec["format"] = {"specifier": ",.2f"}
    elif pd.api.types.is_float_dtype(series):
        spec["format"] = {"specifier": ",.3f"}
    else:
        spec["format"] = {"specifier": ",.0f"}
    return spec


def data_table(df: pd.DataFrame, *, table_id: str, page_size: int = 15,
               index_label: str = "", sort_by: list | None = None,
               height: str = "auto") -> dash_table.DataTable:
    """The table view that backs each chart — the WCAG-clean equivalent."""
    frame = df.reset_index()
    if index_label and frame.columns[0] != index_label:
        frame = frame.rename(columns={frame.columns[0]: index_label})
    frame.columns = [str(c) for c in frame.columns]

    # "Yes"/"No" reads faster than "true"/"false" in a results table.
    for col in frame.columns:
        if pd.api.types.is_bool_dtype(frame[col]):
            frame[col] = frame[col].map({True: "Yes", False: "No"})

    # A datetime index reads better as a plain date string.
    first = frame.columns[0]
    if pd.api.types.is_datetime64_any_dtype(frame[first]):
        frame[first] = frame[first].dt.strftime("%Y-%m-%d")

    columns = [_column_spec(col, frame[col]) for col in frame.columns]

    return dash_table.DataTable(
        id=table_id,
        data=frame.to_dict("records"),
        columns=columns,
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        sort_by=sort_by or [],
        style_as_list_view=True,
        style_table={"overflowX": "auto", "maxHeight": height},
        style_cell={
            "fontFamily": 'system-ui, -apple-system, "Segoe UI", sans-serif',
            "fontSize": "12px", "padding": "8px 12px", "textAlign": "right",
            "border": "none", "fontVariantNumeric": "tabular-nums",
            "backgroundColor": "transparent", "minWidth": "72px",
        },
        style_cell_conditional=[
            {"if": {"column_id": first}, "textAlign": "left", "fontWeight": "600"},
        ],
        style_header={
            "fontWeight": "600", "fontSize": "11px", "textTransform": "uppercase",
            "letterSpacing": "0.06em", "border": "none", "textAlign": "right",
            "backgroundColor": "transparent",
        },
        style_data={"border": "none"},
        css=[{"selector": ".dash-filter input", "rule": "text-align: left !important;"}],
    )


def card(title: str, body, *, subtitle: str | None = None, table=None,
         card_id: str | None = None, actions=None) -> html.Div:
    """A chart card with an optional, collapsible table view of the same data."""
    header = [html.Div([
        html.H3(title, className="card-title"),
        html.P(subtitle, className="card-subtitle") if subtitle else None,
    ], className="card-heading")]
    if actions:
        header.append(html.Div(actions, className="card-actions"))

    children = [html.Div(header, className="card-header"), html.Div(body, className="card-body")]

    if table is not None:
        children.append(html.Details([
            html.Summary("Table view", className="table-summary"),
            html.Div(table, className="table-wrap"),
        ], className="card-table"))

    return html.Div(children, className="card", id=card_id) if card_id else html.Div(children, className="card")


def graph(figure, *, graph_id: str | None = None) -> dcc.Graph:
    kwargs = {"figure": figure, "config": GRAPH_CONFIG, "className": "graph"}
    if graph_id:
        kwargs["id"] = graph_id
    return dcc.Graph(**kwargs)


def grid(children: list, columns: str = "1fr 1fr") -> html.Div:
    return html.Div(children, className="grid", style={"gridTemplateColumns": columns})


def section(title: str, children, *, description: str | None = None) -> html.Div:
    return html.Div([
        html.Div([
            html.H2(title, className="section-title"),
            html.P(description, className="section-description") if description else None,
        ], className="section-header"),
        html.Div(children, className="section-body"),
    ], className="section")


def insight_list(items: list[str], *, title: str = "What the data shows") -> html.Div:
    return html.Div([
        html.H3(title, className="card-title"),
        html.Ul([html.Li(text) for text in items], className="insight-list"),
    ], className="card card--insight")


def badge(text: str, variant: str = "neutral") -> html.Span:
    return html.Span(text, className=f"badge badge--{variant}")


def note(text: str) -> html.P:
    """A caveat shown with the result it qualifies, not buried in a footer."""
    return html.P(text, className="method-note")
