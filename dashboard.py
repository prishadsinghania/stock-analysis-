"""Equity analytics dashboard.

Run:  python dashboard.py     then open http://127.0.0.1:8050

Layout contract:
  * one filter row above the content scopes every chart on the page
  * light and dark are separately validated token sets, not an inversion
  * every chart card carries a table view of the data behind it
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dash
from dash import Input, Output, State, callback_context, dcc, html

from app import state
from app.pages import compare, data, exploration, overview, portfolio, risk, security
from app.state import DEFAULT_WINDOW, WINDOWS, get_view
from src.config import BENCHMARK, SECTORS

PAGES = {
    "overview": overview,
    "security": security,
    "compare": compare,
    "risk": risk,
    "portfolio": portfolio,
    "strategy": __import__("app.pages.strategy", fromlist=["strategy"]),
    "exploration": exploration,
    "data": data,
}

NAV = [
    ("Analysis", [("overview", "Market overview"), ("security", "Security detail"),
                  ("compare", "Compare")]),
    ("Risk", [("risk", "Risk & correlation"), ("exploration", "Exploration")]),
    ("Construction", [("portfolio", "Portfolio lab"), ("strategy", "Strategy lab")]),
    ("Reference", [("data", "Data quality")]),
]

ANALYTICS = state.load()
DEFAULT_TICKERS = list(ANALYTICS.factors.index[:4])

app = dash.Dash(
    __name__,
    title="Equity Analytics",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server


def sidebar() -> html.Div:
    groups = []
    for label, items in NAV:
        groups.append(html.Div(label, className="nav-group"))
        groups.extend(
            dcc.Link(title, href=f"/{key}", id=f"nav-{key}", className="nav-link")
            for key, title in items
        )
    return html.Nav([
        html.Div([
            html.Div("EA", className="brand-mark"),
            html.Div([
                html.Div("Equity Analytics", className="brand-name"),
                html.Div(f"{len(ANALYTICS.tickers)} holdings · {BENCHMARK} benchmark",
                         className="brand-meta"),
            ]),
        ], className="brand"),
        html.Div(groups, className="nav-links"),
        html.Div([
            html.Div(f"{ANALYTICS.start:%b %Y} – {ANALYTICS.end:%b %Y}", className="nav-footer-line"),
            html.Div(f"{ANALYTICS.observations:,} observations", className="nav-footer-line"),
        ], className="nav-footer"),
    ], className="sidebar")


def filter_bar() -> html.Div:
    """One row above everything it scopes — never a filter inside a chart card."""
    return html.Div([
        html.Div([
            html.Label("Window", className="filter-label"),
            html.Div([
                html.Button(label, id={"type": "window-btn", "index": label},
                            className="segment", n_clicks=0)
                for label in WINDOWS
            ], className="segmented", id="window-group"),
        ], className="filter-field"),

        html.Div([
            html.Label("Sectors", className="filter-label"),
            dcc.Dropdown(
                id="sector-filter", options=[{"label": s, "value": s} for s in SECTORS],
                value=[], multi=True, placeholder="All sectors", clearable=True,
                className="dropdown dropdown--multi",
            ),
        ], className="filter-field filter-field--grow"),

        html.Div([
            html.Label("Holding", className="filter-label"),
            dcc.Dropdown(
                id="ticker-select",
                options=_ticker_options(ANALYTICS.tickers),
                value=DEFAULT_TICKERS[0], clearable=False, className="dropdown",
            ),
        ], className="filter-field"),

        html.Div([
            html.Label("Compare (max 8)", className="filter-label"),
            dcc.Dropdown(
                id="compare-select",
                # Ticker-only labels here: eight chips of "NVDA · NVIDIA" would
                # truncate mid-word, and the symbol is the unambiguous key.
                options=[{"label": t, "value": t} for t in sorted(ANALYTICS.tickers)],
                value=DEFAULT_TICKERS, multi=True, className="dropdown dropdown--multi",
            ),
        ], className="filter-field filter-field--grow"),

        html.Div([
            html.Label("Theme", className="filter-label"),
            html.Button("", id="theme-toggle", className="theme-toggle", n_clicks=0),
        ], className="filter-field"),
    ], className="filter-bar")


def _ticker_options(tickers):
    from src.config import name_of

    return [{"label": f"{t} · {name_of(t)}", "value": t} for t in sorted(tickers)]


app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="theme-store", data="dark", storage_type="local"),
    dcc.Store(id="window-store", data=DEFAULT_WINDOW, storage_type="session"),
    dcc.Store(id="slot-store", data={}, storage_type="session"),
    html.Div([
        sidebar(),
        html.Main([
            html.Header([
                html.Div([
                    html.H1(id="page-title", className="page-title"),
                    html.P(id="page-subtitle", className="page-subtitle"),
                ]),
                html.Div(id="page-context", className="page-context"),
            ], className="page-header"),
            filter_bar(),
            dcc.Loading(
                html.Div(id="page-content", className="page-content"),
                type="default", color="#3987e5",
                # Hold the previous render rather than flashing a skeleton.
                delay_show=250, overlay_style={"visibility": "visible", "opacity": 0.55},
            ),
        ], className="main"),
    ], className="shell", id="shell"),
], id="root", **{"data-theme": "dark"})


# ── Theme ────────────────────────────────────────────────────────────────────
@app.callback(
    Output("theme-store", "data"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme(_clicks, current):
    return "light" if current == "dark" else "dark"


app.clientside_callback(
    """
    function(theme) {
        document.documentElement.setAttribute('data-theme', theme || 'dark');
        return theme === 'light' ? 'Dark' : 'Light';
    }
    """,
    Output("theme-toggle", "children"),
    Input("theme-store", "data"),
)


# ── Window selector ──────────────────────────────────────────────────────────
@app.callback(
    Output("window-store", "data"),
    Input({"type": "window-btn", "index": dash.ALL}, "n_clicks"),
    State("window-store", "data"),
    prevent_initial_call=True,
)
def pick_window(_clicks, current):
    triggered = callback_context.triggered_id
    if isinstance(triggered, dict) and "index" in triggered:
        return triggered["index"]
    return current


app.clientside_callback(
    """
    function(active) {
        const buttons = document.querySelectorAll('#window-group .segment');
        buttons.forEach(b => b.classList.toggle('segment--active', b.textContent === active));
        return window.dash_clientside.no_update;
    }
    """,
    Output("window-group", "className"),
    Input("window-store", "data"),
)


# ── Colour-slot assignment ───────────────────────────────────────────────────
@app.callback(
    Output("slot-store", "data"),
    Input("compare-select", "value"),
    State("slot-store", "data"),
)
def update_slots(selected, existing):
    """A holding keeps its colour for as long as it stays selected."""
    return compare.assign_slots(list(selected or []), dict(existing or {}))


# ── Navigation highlight ─────────────────────────────────────────────────────
@app.callback(
    [Output(f"nav-{key}", "className") for _, items in NAV for key, _ in items],
    Input("url", "pathname"),
)
def highlight_nav(pathname):
    active = (pathname or "/").strip("/") or "overview"
    return [
        "nav-link nav-link--active" if key == active else "nav-link"
        for _, items in NAV for key, _ in items
    ]


# ── Page render ──────────────────────────────────────────────────────────────
@app.callback(
    Output("page-content", "children"),
    Output("page-title", "children"),
    Output("page-subtitle", "children"),
    Output("page-context", "children"),
    Input("url", "pathname"),
    Input("theme-store", "data"),
    Input("window-store", "data"),
    Input("sector-filter", "value"),
    Input("ticker-select", "value"),
    Input("compare-select", "value"),
    Input("slot-store", "data"),
)
def render_page(pathname, theme, window, sectors, ticker, compare_tickers, slots):
    key = (pathname or "/").strip("/") or "overview"
    module = PAGES.get(key, overview)
    mode = theme or "dark"

    view = get_view(window, sectors or None)
    if not view.frames:
        body = [html.Div("No holdings match the current filter.", className="card")]
        return body, module.TITLE, module.SUBTITLE, ""

    if key == "security":
        body = module.render(view, mode, _resolve(ticker, view))
    elif key == "compare":
        body = module.render(view, mode, list(compare_tickers or []), dict(slots or {}))
    elif key == "strategy":
        body = module.render(view, mode, _resolve(ticker, view))
    else:
        body = module.render(view, mode)

    context = [
        html.Span(f"{len(view.tickers)} holdings"),
        html.Span(f"{view.start:%b %Y} – {view.end:%b %Y}"),
        html.Span(f"{view.sessions:,} sessions"),
    ]
    return body, module.TITLE, module.SUBTITLE, context


def _resolve(ticker: str, view) -> str:
    """Fall back to the top-ranked in-scope holding if the pick is filtered out."""
    if ticker in view.frames:
        return ticker
    return view.factors.index[0] if len(view.factors) else view.tickers[0]


# ── Downloads ────────────────────────────────────────────────────────────────
@app.callback(
    Output("download-metrics", "data"),
    Input("download-metrics-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_metrics(_clicks):
    return dcc.send_data_frame(ANALYTICS.metrics.to_csv, "equity_metrics.csv")


if __name__ == "__main__":
    print(f"\n  Dashboard ready → http://127.0.0.1:8050\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
