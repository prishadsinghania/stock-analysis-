"""Static chart export for the CLI pipeline and the README.

Uses the same validated palette as the dashboard so the exported PNGs and the
live app are visually one system. Charts are written to ``outputs/charts/``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import CHART_DPI, OUTPUT_DIR
from src.eda import cluster_order, monthly_seasonality, regime_matrix

# Light-mode tokens — exported charts land in READMEs and slide decks.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
POSITIVE = "#006300"
NEGATIVE = "#d03b3b"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Magnitude ramp: one hue, light to dark, for all-positive scales.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
)

# Polarity ramp for returns: losses red, gains blue. Red/green is the pairing
# most often confused under colour-vision deficiency, so it is not used.
DIVERGING_RETURNS = LinearSegmentedColormap.from_list(
    "red_blue", ["#8f1f1f", "#e34948", "#f0efec", "#3987e5", "#0d366b"]
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",     # solid hairlines; dashed grids read as thresholds
    "legend.frameon": False,
})


def _title_block(ax, title: str, subtitle: str = "") -> None:
    """Title and subtitle placed in figure space so they never collide."""
    # Offsets are in points, converted to axes fraction, so the gap is the same
    # regardless of how tall the axes happen to be.
    height_points = ax.get_figure().get_size_inches()[1] * 72
    line = 17 / height_points
    if subtitle:
        ax.text(0, 1 + 2.1 * line, title, transform=ax.transAxes,
                color=TEXT_PRIMARY, fontsize=13, fontweight="600", va="bottom")
        ax.text(0, 1 + 0.75 * line, subtitle, transform=ax.transAxes,
                color=TEXT_SECONDARY, fontsize=10, va="bottom")
    else:
        ax.text(0, 1 + 0.75 * line, title, transform=ax.transAxes,
                color=TEXT_PRIMARY, fontsize=13, fontweight="600", va="bottom")


def _canvas(title: str, subtitle: str = "", figsize=(11, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    _title_block(ax, title, subtitle)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def _save(fig, filename: str) -> str:
    path = f"{OUTPUT_DIR}/{filename}"
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    {filename}")
    return path


def chart_sector_performance(metrics: pd.DataFrame, sectors: pd.DataFrame) -> str:
    """One series, so one colour — bar length is the encoding."""
    data = (sectors["CAGR"] * 100).sort_values()
    fig, ax = _canvas("Annualised return by sector",
                      "Equal-weighted mean CAGR of each sector's holdings",
                      figsize=(10, 6))
    ax.barh(data.index, data.values, color=CATEGORICAL[0], height=0.62)
    for name, value in data.items():
        ax.text(value + 0.25, name, f"{value:.1f}%", va="center",
                fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlabel("CAGR (%)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.9)
    ax.set_xlim(0, data.max() * 1.16)
    return _save(fig, "01_sector_performance.png")


def chart_risk_return(metrics: pd.DataFrame) -> str:
    """Every holding on one axis pair; only the extremes are labelled."""
    fig, ax = _canvas("Risk versus return across the universe",
                      f"{len(metrics)} US large caps — annualised, over the full window",
                      figsize=(10, 6.5))
    x = metrics["Volatility"] * 100
    y = metrics["CAGR"] * 100
    ax.scatter(x, y, s=46, color=CATEGORICAL[0], alpha=0.5,
               edgecolors=SURFACE, linewidths=1.6, zorder=3)

    # Label only the extremes — a name on every point would be unreadable.
    notable = set(metrics.nlargest(4, "CAGR").index) | set(metrics.nsmallest(3, "CAGR").index) \
        | set(metrics.nlargest(2, "Volatility").index) | set(metrics.nlargest(3, "Sharpe").index)
    for ticker in notable:
        ax.annotate(ticker, (x[ticker], y[ticker]), xytext=(7, 5),
                    textcoords="offset points", fontsize=9, color=TEXT_PRIMARY)

    ax.axhline(0, color=AXIS, linewidth=0.9)
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("CAGR (%)")
    ax.grid(alpha=0.9)
    # Labels sit to the upper-right of their point, so the extremes need room.
    ax.set_xlim(x.min() - 1.5, x.max() + 6.5)
    ax.set_ylim(y.min() - 4, y.max() + 7)
    return _save(fig, "02_risk_return.png")


def chart_correlation(returns: pd.DataFrame) -> str:
    """Diverging scale around zero, clustered so the blocks are visible."""
    corr = returns.corr()
    order = cluster_order(corr)
    corr = corr.loc[order, order]

    fig, ax = plt.subplots(figsize=(11, 9.5))
    # Correlations here are all positive, so this is a magnitude scale; a
    # diverging ramp centred on zero would throw away half its range.
    image = ax.imshow(corr.values, cmap=SEQUENTIAL,
                      vmin=max(0.0, float(corr.values.min())), vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
    _title_block(ax, "Return correlation, hierarchically clustered",
                 "Blocks of holdings that move together sit next to each other")
    bar = fig.colorbar(image, ax=ax, shrink=0.55, pad=0.02)
    bar.set_label("Correlation", color=TEXT_SECONDARY, fontsize=9)
    bar.outline.set_visible(False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save(fig, "03_correlation_matrix.png")


def chart_strategy_curves(curves: dict[str, pd.Series]) -> str:
    fig, ax = _canvas("Growth of $1 by portfolio construction method",
                      "Quarterly rebalancing, 5 bps charged on turnover",
                      figsize=(11, 5.8))
    for slot, (name, series) in enumerate(curves.items()):
        growth = (1 + series.fillna(0)).cumprod()
        benchmark = "Benchmark" in name
        ax.plot(growth.index, growth.values,
                color=TEXT_MUTED if benchmark else CATEGORICAL[slot],
                linewidth=1.4 if benchmark else 2.0,
                linestyle=":" if benchmark else "-", label=name)
        ax.annotate(f"${growth.iloc[-1]:,.2f}", (growth.index[-1], growth.iloc[-1]),
                    xytext=(6, -3), textcoords="offset points",
                    fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=9, labelcolor=TEXT_SECONDARY, ncol=2)
    ax.margins(x=0.06)
    return _save(fig, "04_portfolio_curves.png")


def chart_regime_matrix(returns: pd.DataFrame) -> str:
    grid = regime_matrix(returns) * 100
    fig, ax = plt.subplots(figsize=(10, 5.6))
    limit = np.nanmax(np.abs(grid.values))
    image = ax.imshow(grid.values, cmap=DIVERGING_RETURNS,
                      vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(range(len(grid.columns)), grid.columns, rotation=22, ha="right", fontsize=9)
    ax.set_yticks(range(len(grid.index)), grid.index, fontsize=9)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.values[i, j]
            if np.isnan(value):
                continue
            # Pick ink by cell luminance so the label always clears contrast.
            ink = "#ffffff" if abs(value) > limit * 0.55 else TEXT_PRIMARY
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8, color=ink)

    _title_block(ax, "Sector total return by market regime",
                 "Percent, over six documented market episodes")
    bar = fig.colorbar(image, ax=ax, shrink=0.7, pad=0.02)
    bar.set_label("Total return (%)", color=TEXT_SECONDARY, fontsize=9)
    bar.outline.set_visible(False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save(fig, "05_regime_matrix.png")


def chart_return_distribution(returns: pd.DataFrame) -> str:
    """The fat-tail finding, made visible."""
    equal_weighted = returns.mean(axis=1) * 100
    fig, ax = _canvas("Daily returns are not normally distributed",
                      "Equal-weighted universe against a fitted normal curve",
                      figsize=(10, 5.2))
    ax.hist(equal_weighted, bins=110, color=CATEGORICAL[0], alpha=0.8, density=True)
    grid = np.linspace(equal_weighted.min(), equal_weighted.max(), 400)
    mu, sigma = equal_weighted.mean(), equal_weighted.std()
    density = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((grid - mu) / sigma) ** 2)
    ax.plot(grid, density, color=TEXT_PRIMARY, linewidth=1.6, label="Normal fit")
    ax.set_xlabel("Daily return (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, labelcolor=TEXT_SECONDARY)
    return _save(fig, "06_return_distribution.png")


def chart_seasonality(returns: pd.DataFrame) -> str:
    monthly = monthly_seasonality(returns).mean(axis=1) * 100
    fig, ax = _canvas("Average return by calendar month",
                      "Equal-weighted across holdings, averaged over every year in the window",
                      figsize=(10, 4.6))
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in monthly.values]
    ax.bar(monthly.index, monthly.values, color=colors, width=0.62)
    ax.axhline(0, color=AXIS, linewidth=0.9)
    for month, value in monthly.items():
        ax.text(month, value + (0.12 if value >= 0 else -0.12), f"{value:+.1f}",
                ha="center", va="bottom" if value >= 0 else "top",
                fontsize=8, color=TEXT_SECONDARY)
    ax.set_ylabel("Mean monthly return (%)")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.18)
    return _save(fig, "07_seasonality.png")


def chart_top_holdings(metrics: pd.DataFrame, n: int = 15) -> str:
    top = metrics.nlargest(n, "CAGR")["CAGR"].sort_values() * 100
    fig, ax = _canvas(f"Top {n} holdings by annualised return",
                      "CAGR over the full analysis window", figsize=(10, 6.4))
    ax.barh(top.index, top.values, color=CATEGORICAL[0], height=0.62)
    for name, value in top.items():
        ax.text(value + 0.6, name, f"{value:.1f}%", va="center",
                fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlabel("CAGR (%)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.9)
    ax.set_xlim(0, top.max() * 1.14)
    return _save(fig, "08_top_holdings.png")


def generate_all_charts(bundle) -> list[str]:
    print("\n" + "─" * 72)
    print("  7 · CHART EXPORT")
    print("─" * 72)
    paths = [
        chart_sector_performance(bundle.metrics, bundle.sectors),
        chart_risk_return(bundle.metrics),
        chart_correlation(bundle.returns),
        chart_strategy_curves(bundle.portfolio_curves),
        chart_regime_matrix(bundle.returns),
        chart_return_distribution(bundle.returns),
        chart_seasonality(bundle.returns),
        chart_top_holdings(bundle.metrics),
    ]
    print(f"  {len(paths)} charts → {OUTPUT_DIR}")
    return paths
