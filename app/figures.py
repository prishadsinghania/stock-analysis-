"""Plotly figure builders.

Every builder takes the theme mode so light and dark render from their own
validated token set. Shared conventions:

  * one y-axis per chart, always (an indexed base replaces any dual axis)
  * a legend whenever two or more series are plotted; none for a single series
  * direct labels used selectively — endpoints and extremes only
  * a crosshair tooltip on time series, a per-mark tooltip everywhere else
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.theme import (
    AREA_ALPHA,
    LINE_WIDTH,
    LINE_WIDTH_THIN,
    MARKER_SIZE,
    SEQUENTIAL,
    diverging,
    diverging_poles,
    diverging_returns,
    layout,
    rgba,
    series_color,
    tokens,
)
from src.config import MA_200, MA_LONG, MA_SHORT, RSI_WINDOW


def empty(mode: str, message: str = "No data for this selection", height: int = 320) -> go.Figure:
    t = tokens(mode)
    fig = go.Figure()
    fig.update_layout(**layout(mode, height=height, showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(size=13, color=t["text_muted"]))
    return fig


def _endpoint_label(fig: go.Figure, x, y, text: str, color: str, mode: str) -> None:
    """Direct-label the last point of a line — the one label worth showing."""
    t = tokens(mode)
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers",
        marker=dict(size=MARKER_SIZE, color=color,
                    line=dict(width=2, color=t["surface"])),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_annotation(
        x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=10,
        font=dict(size=11, color=t["text_secondary"]),
        bgcolor=rgba(t["surface"], 0.75), borderpad=2,
    )


# ── Time series ──────────────────────────────────────────────────────────────
def price_history(df: pd.DataFrame, ticker: str, mode: str, height: int = 320) -> go.Figure:
    """Single-series close price — no legend needed, the title names it."""
    t = tokens(mode)
    color = series_color(mode, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name=ticker, mode="lines",
        line=dict(color=color, width=LINE_WIDTH, shape="linear"),
        fill="tozeroy", fillcolor=rgba(color, AREA_ALPHA),
        hovertemplate="%{x|%d %b %Y}<br><b>$%{y:,.2f}</b><extra></extra>",
    ))
    fig.update_layout(**layout(mode, height=height, showlegend=False, hovermode="x unified"))
    fig.update_yaxes(title_text="Close (USD)", tickprefix="$", tickformat=",.0f")
    if len(df):
        _endpoint_label(fig, df.index[-1], df["Close"].iloc[-1],
                        f"${df['Close'].iloc[-1]:,.2f}", color, mode)
    return fig


def price_with_moving_averages(df: pd.DataFrame, ticker: str, mode: str,
                               height: int = 400) -> go.Figure:
    fig = go.Figure()
    series = [
        (ticker, df["Close"], 0, LINE_WIDTH),
        (f"SMA {MA_SHORT}", df[f"SMA_{MA_SHORT}"], 1, LINE_WIDTH_THIN),
        (f"SMA {MA_LONG}", df[f"SMA_{MA_LONG}"], 2, LINE_WIDTH_THIN),
        (f"SMA {MA_200}", df[f"SMA_{MA_200}"], 3, LINE_WIDTH_THIN),
    ]
    for name, values, slot, width in series:
        fig.add_trace(go.Scatter(
            x=df.index, y=values, name=name, mode="lines",
            line=dict(color=series_color(mode, slot), width=width),
            hovertemplate=f"{name}: $%{{y:,.2f}}<extra></extra>",
        ))
    fig.update_layout(**layout(mode, height=height, hovermode="x unified"))
    fig.update_yaxes(title_text="Price (USD)", tickprefix="$", tickformat=",.0f")
    return fig


def bollinger(df: pd.DataFrame, ticker: str, mode: str, height: int = 340) -> go.Figure:
    t = tokens(mode)
    band = series_color(mode, 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="Upper band",
                             line=dict(color=band, width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="Lower band",
                             line=dict(color=band, width=1),
                             fill="tonexty", fillcolor=rgba(band, AREA_ALPHA),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name=ticker,
        line=dict(color=series_color(mode, 0), width=LINE_WIDTH),
        hovertemplate="%{x|%d %b %Y}<br>Close <b>$%{y:,.2f}</b><extra></extra>",
    ))
    fig.update_layout(**layout(mode, height=height, hovermode="x unified"))
    fig.update_yaxes(title_text="Price (USD)", tickprefix="$", tickformat=",.0f")
    return fig


def rsi(df: pd.DataFrame, mode: str, height: int = 260) -> go.Figure:
    t = tokens(mode)
    color = series_color(mode, 0)
    col = f"RSI_{RSI_WINDOW}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df[col], name=f"RSI {RSI_WINDOW}",
        line=dict(color=color, width=LINE_WIDTH_THIN),
        hovertemplate="%{x|%d %b %Y}<br>RSI <b>%{y:.1f}</b><extra></extra>",
    ))
    # Threshold bands: reference regions, labelled in text, not colour alone.
    for y0, y1, label, anchor in ((70, 100, "Overbought 70+", 70), (0, 30, "Oversold 30−", 30)):
        fig.add_hrect(y0=y0, y1=y1, fillcolor=t["text_muted"], opacity=0.07, line_width=0)
        fig.add_hline(y=anchor, line=dict(color=t["axis"], width=1))
        fig.add_annotation(x=df.index[0] if len(df) else 0, y=anchor, text=label,
                           showarrow=False, xanchor="left", yshift=9 if anchor == 70 else -9,
                           font=dict(size=10, color=t["text_muted"]))
    fig.update_layout(**layout(mode, height=height, showlegend=False, hovermode="x unified"))
    fig.update_yaxes(range=[0, 100], title_text="RSI", dtick=25)
    return fig


def macd(df: pd.DataFrame, mode: str, height: int = 240) -> go.Figure:
    t = tokens(mode)
    hist = df["MACD_Hist"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=hist, name="Histogram",
        marker_color=np.where(hist >= 0, diverging_poles(mode)[1], diverging_poles(mode)[0]),
        marker_line_width=0, opacity=0.55,
        hovertemplate="%{x|%d %b %Y}<br>Histogram <b>%{y:.2f}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                             line=dict(color=series_color(mode, 0), width=LINE_WIDTH_THIN),
                             hovertemplate="MACD %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                             line=dict(color=series_color(mode, 1), width=LINE_WIDTH_THIN),
                             hovertemplate="Signal %{y:.2f}<extra></extra>"))
    fig.update_layout(**layout(mode, height=height, hovermode="x unified", bargap=0))
    fig.update_yaxes(title_text="MACD")
    return fig


def volume(df: pd.DataFrame, mode: str, height: int = 200) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=series_color(mode, 0), marker_line_width=0, opacity=0.45,
        hovertemplate="%{x|%d %b %Y}<br>Volume <b>%{y:,.0f}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Volume_SMA_20"], name="20-day average",
        line=dict(color=series_color(mode, 1), width=LINE_WIDTH_THIN),
        hovertemplate="20-day avg %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**layout(mode, height=height, hovermode="x unified", bargap=0))
    fig.update_yaxes(title_text="Shares", tickformat=".2s")
    return fig


def drawdown_curve(df: pd.DataFrame, mode: str, height: int = 240) -> go.Figure:
    t = tokens(mode)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Drawdown"], name="Drawdown",
        line=dict(color=t["negative"], width=LINE_WIDTH_THIN),
        fill="tozeroy", fillcolor=rgba(t["negative"], AREA_ALPHA),
        hovertemplate="%{x|%d %b %Y}<br>Drawdown <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(**layout(mode, height=height, showlegend=False, hovermode="x unified"))
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%")
    return fig


def return_distribution(returns: pd.Series, mode: str, height: int = 280) -> go.Figure:
    """Histogram with the normal curve overlaid — the fat tails are the point."""
    t = tokens(mode)
    r = returns.dropna() * 100
    color = series_color(mode, 0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, nbinsx=90, name="Observed", marker_color=color,
        marker_line_width=0, opacity=0.75, histnorm="probability density",
        hovertemplate="Return %{x:.2f}%<br>Density %{y:.3f}<extra></extra>",
    ))
    if len(r) > 2 and r.std() > 0:
        grid = np.linspace(r.min(), r.max(), 240)
        density = (1 / (r.std() * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((grid - r.mean()) / r.std()) ** 2)
        fig.add_trace(go.Scatter(
            x=grid, y=density, name="Normal fit",
            line=dict(color=t["text_secondary"], width=LINE_WIDTH_THIN),
            hovertemplate="Normal density %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(**layout(mode, height=height))
    fig.update_xaxes(title_text="Daily return (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Density")
    return fig


# ── Multi-series comparison ──────────────────────────────────────────────────
def indexed_performance(frames: dict[str, pd.DataFrame], slots: dict[str, int],
                        mode: str, height: int = 400,
                        benchmark: pd.Series | None = None) -> go.Figure:
    """Every series rebased to 100 — one axis, so the comparison is honest."""
    t = tokens(mode)
    fig = go.Figure()

    if benchmark is not None and len(benchmark):
        base = benchmark / benchmark.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=base.index, y=base, name="SPY (benchmark)",
            line=dict(color=t["text_muted"], width=LINE_WIDTH_THIN, dash="dot"),
            hovertemplate="SPY <b>%{y:,.0f}</b><extra></extra>",
        ))

    for ticker, df in frames.items():
        close = df["Close"].dropna()
        if close.empty:
            continue
        indexed = close / close.iloc[0] * 100
        color = series_color(mode, slots.get(ticker, 0))
        fig.add_trace(go.Scatter(
            x=indexed.index, y=indexed, name=ticker,
            line=dict(color=color, width=LINE_WIDTH),
            hovertemplate=f"{ticker} <b>%{{y:,.0f}}</b><extra></extra>",
        ))

    fig.add_hline(y=100, line=dict(color=t["axis"], width=1))
    fig.update_layout(**layout(mode, height=height, hovermode="x unified"))
    fig.update_yaxes(title_text="Indexed to 100 at start", tickformat=",.0f")
    return fig


def strategy_curves(curves: dict[str, pd.Series], mode: str, height: int = 380) -> go.Figure:
    """Growth of $1 for each portfolio construction method."""
    t = tokens(mode)
    fig = go.Figure()
    for slot, (name, series) in enumerate(curves.items()):
        growth = (1 + series.fillna(0)).cumprod()
        is_benchmark = "Benchmark" in name
        color = t["text_muted"] if is_benchmark else series_color(mode, slot)
        fig.add_trace(go.Scatter(
            x=growth.index, y=growth, name=name,
            line=dict(color=color, width=LINE_WIDTH_THIN if is_benchmark else LINE_WIDTH,
                      dash="dot" if is_benchmark else "solid"),
            hovertemplate=f"{name} <b>$%{{y:,.2f}}</b><extra></extra>",
        ))
    fig.update_layout(**layout(mode, height=height, hovermode="x unified"))
    fig.update_yaxes(title_text="Growth of $1", tickprefix="$", tickformat=",.1f")
    return fig


# ── Cross-section ────────────────────────────────────────────────────────────
def risk_return_scatter(metrics: pd.DataFrame, mode: str, highlight: list[str] | None = None,
                        height: int = 440) -> go.Figure:
    """60 names on one axis pair. One series colour; emphasis marks the selection."""
    t = tokens(mode)
    highlight = set(highlight or [])
    accent = series_color(mode, 0)

    base = metrics[~metrics.index.isin(highlight)]
    picked = metrics[metrics.index.isin(highlight)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base["Volatility"] * 100, y=base["CAGR"] * 100, mode="markers",
        name="Universe",
        marker=dict(size=MARKER_SIZE, color=rgba(accent, 0.45),
                    line=dict(width=2, color=t["surface"])),
        customdata=np.stack([base.index, base["Sector"], base["Sharpe"]], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]}"
                       "<br>Volatility %{x:.1f}%<br>CAGR %{y:.1f}%"
                       "<br>Sharpe %{customdata[2]:.2f}<extra></extra>"),
    ))

    if len(picked):
        fig.add_trace(go.Scatter(
            x=picked["Volatility"] * 100, y=picked["CAGR"] * 100,
            mode="markers+text", name="Selected",
            text=picked.index, textposition="top center",
            textfont=dict(size=11, color=t["text_primary"]),
            marker=dict(size=MARKER_SIZE + 5, color=accent,
                        line=dict(width=2, color=t["surface"])),
            customdata=np.stack([picked.index, picked["Sector"], picked["Sharpe"]], axis=-1),
            hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]}"
                           "<br>Volatility %{x:.1f}%<br>CAGR %{y:.1f}%"
                           "<br>Sharpe %{customdata[2]:.2f}<extra></extra>"),
        ))

    fig.update_layout(**layout(mode, height=height, showlegend=len(picked) > 0))
    fig.update_xaxes(title_text="Annualised volatility (%)", ticksuffix="%")
    fig.update_yaxes(title_text="CAGR (%)", ticksuffix="%")
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1))
    return fig


def efficient_frontier_chart(frontier: pd.DataFrame, cloud: pd.DataFrame,
                             points: dict[str, dict], mode: str, height: int = 440) -> go.Figure:
    """Random portfolios shaded by Sharpe, with the frontier and named solutions."""
    t = tokens(mode)
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=cloud["Volatility"] * 100, y=cloud["Return"] * 100, mode="markers",
        name="Random portfolios",
        marker=dict(size=4, color=cloud["Sharpe"], colorscale=[[0, "#cde2fb"], [1, "#0d366b"]],
                    showscale=True, opacity=0.38,
                    colorbar=dict(title=dict(text="Sharpe", font=dict(size=11, color=t["text_secondary"])),
                                  thickness=10, len=0.55, outlinewidth=0,
                                  tickfont=dict(size=10, color=t["text_muted"]))),
        hovertemplate="Vol %{x:.1f}%<br>Return %{y:.1f}%<extra></extra>",
    ))

    if len(frontier):
        fig.add_trace(go.Scatter(
            x=frontier["Volatility"] * 100, y=frontier["Return"] * 100,
            mode="lines", name="Efficient frontier",
            line=dict(color=t["text_primary"], width=LINE_WIDTH),
            hovertemplate="Frontier — vol %{x:.1f}%, return %{y:.1f}%<extra></extra>",
        ))

    # Stagger the labels: the four solutions can sit close together inside the
    # cloud, and stacked text detached from its marker reads as noise.
    positions = ["top center", "bottom center", "middle left", "middle right"]
    for slot, (name, stats) in enumerate(points.items()):
        fig.add_trace(go.Scatter(
            x=[stats["volatility"] * 100], y=[stats["return"] * 100],
            mode="markers+text", name=name, text=[name],
            textposition=positions[slot % len(positions)],
            textfont=dict(size=11, color=t["text_primary"]),
            marker=dict(size=MARKER_SIZE + 4, color=series_color(mode, slot),
                        line=dict(width=2.5, color=t["surface"])),
            cliponaxis=False,
            hovertemplate=(f"<b>{name}</b><br>Return %{{y:.1f}}%<br>Vol %{{x:.1f}}%"
                           f"<br>Sharpe {stats['sharpe']:.2f}<extra></extra>"),
        ))

    fig.update_layout(**layout(mode, height=height))
    fig.update_xaxes(title_text="Annualised volatility (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Expected annual return (%)", ticksuffix="%")
    return fig


def ranked_bar(series: pd.Series, mode: str, *, title: str = "", suffix: str = "%",
               height: int = 360, decimals: int = 1) -> go.Figure:
    """Nominal categories, so one colour for every bar — length is the encoding."""
    t = tokens(mode)
    data = series.dropna().sort_values()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data.values, y=list(data.index), orientation="h",
        marker_color=series_color(mode, 0), marker_line_width=0,
        width=0.68,
        text=[f"{v:,.{decimals}f}{suffix}" for v in data.values],
        textposition="outside", outsidetextfont=dict(size=11, color=t["text_secondary"]),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:,." + str(decimals) + "f}" + suffix + "<extra></extra>",
    ))
    fig.update_layout(**layout(mode, height=height, showlegend=False,
                               margin=dict(l=8, r=56, t=8, b=8), bargap=0.3))
    lo, hi = float(min(data.min(), 0)), float(max(data.max(), 0))
    pad = max(abs(lo), abs(hi)) * 0.16 or 1.0
    fig.update_xaxes(title_text=title, ticksuffix=suffix, showgrid=True,
                     range=[lo - (pad if lo < 0 else 0), hi + pad])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11))
    return fig


def diverging_bar(series: pd.Series, mode: str, *, suffix: str = "%",
                  height: int = 360, decimals: int = 1) -> go.Figure:
    """Signed values — direction is the message, so the two poles differ."""
    t = tokens(mode)
    data = series.dropna().sort_values()
    # Diverging poles, not green/red: red-green is the pairing most often
    # confused, and the sign is already carried by the +/- in every label.
    loss, gain = diverging_poles(mode)
    colors = [gain if v >= 0 else loss for v in data.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data.values, y=list(data.index), orientation="h",
        marker_color=colors, marker_line_width=0, width=0.68,
        text=[f"{v:+,.{decimals}f}{suffix}" for v in data.values],
        textposition="outside", outsidetextfont=dict(size=11, color=t["text_secondary"]),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:+,." + str(decimals) + "f}" + suffix + "<extra></extra>",
    ))
    fig.update_layout(**layout(mode, height=height, showlegend=False,
                               margin=dict(l=8, r=56, t=8, b=8), bargap=0.3))
    fig.add_vline(x=0, line=dict(color=t["axis"], width=1))
    # Outside labels need room on whichever side the bar grows, or they collide
    # with the category ticks. Measure the extent and pad both ends.
    lo, hi = float(min(data.min(), 0)), float(max(data.max(), 0))
    pad = max(abs(lo), abs(hi)) * 0.22 or 1.0
    fig.update_xaxes(ticksuffix=suffix, range=[lo - pad, hi + pad])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11))
    return fig


def heatmap(matrix: pd.DataFrame, mode: str, *, height: int = 440, zmid: float = 0.0,
            value_suffix: str = "", show_text: bool = True, decimals: int = 2,
            colorbar_title: str = "", scale: str = "magnitude") -> go.Figure:
    """Diverging heatmap — two poles around a neutral midpoint.

    ``scale="returns"`` flips the ramp so losses read red and gains blue.
    """
    t = tokens(mode)
    if scale == "sequential":
        # Magnitude, not polarity: one hue, light to dark. Using a diverging
        # ramp on all-positive values throws away half its range.
        colorscale = SEQUENTIAL
        zmid = None
    elif scale == "returns":
        colorscale = diverging_returns(mode)
    else:
        colorscale = diverging(mode)

    # Blank cells stay blank — a "nan" label is worse than no label.
    labels = np.where(np.isnan(matrix.values.astype(float)), None, matrix.values) \
        if show_text else None

    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=list(matrix.columns), y=list(matrix.index),
        colorscale=colorscale, zmid=zmid,
        xgap=2, ygap=2,   # surface gap, not a border, separates the cells
        text=labels,
        texttemplate=f"%{{text:.{decimals}f}}{value_suffix}" if show_text else None,
        textfont=dict(size=10),
        hovertemplate=("%{y} × %{x}<br><b>%{z:." + str(decimals) + "f}"
                       + value_suffix + "</b><extra></extra>"),
        colorbar=dict(title=dict(text=colorbar_title, font=dict(size=11, color=t["text_secondary"])),
                      thickness=10, len=0.6, outlinewidth=0,
                      tickfont=dict(size=10, color=t["text_muted"])),
    ))
    fig.update_layout(**layout(mode, height=height, showlegend=False,
                               margin=dict(l=8, r=8, t=8, b=8)))
    fig.update_xaxes(showgrid=False, side="bottom", tickfont=dict(size=10))
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(size=10))
    return fig


def sector_dispersion(metrics: pd.DataFrame, mode: str, height: int = 380) -> go.Figure:
    """Distribution of member CAGRs within each sector — one series, so one colour."""
    t = tokens(mode)
    order = metrics.groupby("Sector")["CAGR"].median().sort_values().index
    color = series_color(mode, 0)
    fig = go.Figure()
    for sector in order:
        values = metrics.loc[metrics["Sector"] == sector, "CAGR"] * 100
        fig.add_trace(go.Box(
            x=values, name=sector, orientation="h",
            marker=dict(color=color, size=6, opacity=0.7),
            line=dict(color=color, width=LINE_WIDTH_THIN),
            fillcolor=rgba(color, AREA_ALPHA),
            boxpoints="all", jitter=0.5, pointpos=0, showlegend=False,
            hovertemplate="<b>%{text}</b><br>CAGR %{x:.1f}%<extra></extra>",
            text=metrics.loc[metrics["Sector"] == sector].index,
        ))
    fig.update_layout(**layout(mode, height=height, showlegend=False,
                               margin=dict(l=8, r=16, t=8, b=8)))
    fig.add_vline(x=0, line=dict(color=t["axis"], width=1))
    fig.update_xaxes(title_text="CAGR (%)", ticksuffix="%")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11))
    return fig


def rolling_line(series: pd.Series, mode: str, *, label: str, y_title: str,
                 height: int = 260, suffix: str = "", reference: float | None = None) -> go.Figure:
    t = tokens(mode)
    color = series_color(mode, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, name=label,
        line=dict(color=color, width=LINE_WIDTH),
        fill="tozeroy", fillcolor=rgba(color, AREA_ALPHA),
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.2f}" + suffix + "</b><extra></extra>",
    ))
    if reference is not None:
        fig.add_hline(y=reference, line=dict(color=t["axis"], width=1))
    fig.update_layout(**layout(mode, height=height, showlegend=False, hovermode="x unified"))
    fig.update_yaxes(title_text=y_title, ticksuffix=suffix)
    return fig


def dispersion_band(dispersion: pd.DataFrame, mode: str, height: int = 300) -> go.Figure:
    """Median with a P10–P90 band — the room stock selection has to work in."""
    t = tokens(mode)
    color = series_color(mode, 0)
    fig = go.Figure()
    # The band is the fill; drawing its edges as lines would read as two more
    # series competing with the median.
    fig.add_trace(go.Scatter(x=dispersion.index, y=dispersion["P90"] * 100,
                             name="10th–90th percentile band", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=dispersion.index, y=dispersion["P10"] * 100,
                             name="10th–90th percentile band", line=dict(width=0),
                             fill="tonexty", fillcolor=rgba(color, 0.18),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=dispersion.index, y=dispersion["Median"] * 100, name="Median stock",
        line=dict(color=color, width=LINE_WIDTH),
        hovertemplate="%{x|%b %Y}<br>Median <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1))
    fig.update_layout(**layout(mode, height=height, hovermode="x unified"))
    fig.update_yaxes(title_text="Rolling 1-month return (%)", ticksuffix="%")
    return fig


def weight_comparison(weights: pd.DataFrame, mode: str, top_n: int = 15,
                      height: int = 400) -> go.Figure:
    """Grouped bars: how each construction method allocates. Legend + gaps."""
    t = tokens(mode)
    ranked = weights.loc[weights.max(axis=1).sort_values(ascending=False).index[:top_n]]
    fig = go.Figure()
    for slot, method in enumerate(ranked.columns):
        fig.add_trace(go.Bar(
            x=list(ranked.index), y=ranked[method] * 100, name=method,
            marker_color=series_color(mode, slot), marker_line_width=0,
            hovertemplate=f"<b>%{{x}}</b><br>{method} <b>%{{y:.2f}}%</b><extra></extra>",
        ))
    fig.update_layout(**layout(mode, height=height, barmode="group",
                               bargap=0.28, bargroupgap=0.08))
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(title_text="Weight (%)", ticksuffix="%")
    return fig


def sparkline(series: pd.Series, mode: str, positive: bool) -> go.Figure:
    """12-point trend for a stat tile — no axes, no legend, pure shape."""
    t = tokens(mode)
    color = t["positive"] if positive else t["negative"]
    fig = go.Figure(go.Scatter(
        x=list(range(len(series))), y=series.values, mode="lines",
        line=dict(color=color, width=1.5), fill="tozeroy",
        fillcolor=rgba(color, 0.12), hoverinfo="skip",
    ))
    fig.update_layout(
        height=36, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig
