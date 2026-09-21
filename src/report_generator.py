"""Report export: a readable text brief, plus CSV and Excel deliverables."""

from __future__ import annotations

import datetime as dt
import os

import pandas as pd

from src.config import BENCHMARK, DATA_DIR, END_DATE, REPORT_DIR, START_DATE

WIDTH = 78


def _rule(char: str = "─") -> str:
    return char * WIDTH


def _pct(value: float, decimals: int = 1, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:{'+' if signed else ''}.{decimals}f}%"


def build_text_report(bundle) -> str:
    metrics = bundle.metrics
    sectors = bundle.sectors
    factors = bundle.factors
    portfolios = bundle.portfolios
    scan = bundle.strategy_scan

    bench = portfolios.loc["Benchmark (SPY)"] if "Benchmark (SPY)" in portfolios.index else None
    strategies = portfolios.drop(index="Benchmark (SPY)", errors="ignore")

    lines: list[str] = [
        _rule("═"),
        "  EQUITY ANALYTICS — SUMMARY REPORT",
        _rule("═"),
        f"  Generated   : {dt.datetime.now():%Y-%m-%d %H:%M}",
        f"  Window      : {START_DATE} → {END_DATE}  ({bundle.sessions:,} sessions)",
        f"  Universe    : {len(metrics)} US large caps across "
        f"{metrics['Sector'].nunique()} GICS sectors",
        f"  Benchmark   : {BENCHMARK}",
        f"  Risk-free   : {bundle.risk_free:.2%} (mean 13-week T-bill over the window)",
        f"  Observations: {bundle.observations:,} daily price records",
        "",
        _rule("═"),
        "  UNIVERSE SUMMARY",
        _rule("═"),
        f"  Median CAGR            : {_pct(metrics['CAGR'].median())}",
        f"  Median volatility      : {_pct(metrics['Volatility'].median())}",
        f"  Median Sharpe          : {metrics['Sharpe'].median():.2f}",
        f"  Median max drawdown    : {_pct(metrics['Max Drawdown'].median())}",
        f"  {'Median beta vs ' + BENCHMARK:<23}: {metrics['Beta'].median():.2f}",
        f"  {'Names beating ' + BENCHMARK:<23}: "
        f"{int((metrics['CAGR'] > (bench['CAGR'] if bench is not None else 0)).sum())} of {len(metrics)}",
        "",
        _rule("═"),
        "  TOP 10 BY COMPOSITE FACTOR SCORE",
        _rule("═"),
        f"  {'Rank':<5}{'Ticker':<8}{'Sector':<26}{'Score':>7}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}",
        _rule(),
    ]

    for ticker in factors.index[:10]:
        row = factors.loc[ticker]
        met = metrics.loc[ticker]
        lines.append(
            f"  {int(row['Rank']):<5}{ticker:<8}{row['Sector'][:24]:<26}"
            f"{row['Composite Score']:>7.2f}{_pct(met['CAGR']):>9}"
            f"{met['Sharpe']:>8.2f}{_pct(met['Max Drawdown']):>9}"
        )

    lines += [
        "",
        _rule("═"),
        "  SECTOR ROLL-UP (equal-weighted)",
        _rule("═"),
        f"  {'Sector':<26}{'N':>4}{'CAGR':>9}{'Vol':>9}{'Sharpe':>8}{'Beta':>7}{'MaxDD':>9}",
        _rule(),
    ]
    for sector, row in sectors.iterrows():
        lines.append(
            f"  {sector[:24]:<26}{int(row['Holdings']):>4}{_pct(row['CAGR']):>9}"
            f"{_pct(row['Volatility']):>9}{row['Sharpe']:>8.2f}"
            f"{row['Beta']:>7.2f}{_pct(row['Max Drawdown']):>9}"
        )

    lines += [
        "",
        _rule("═"),
        "  PORTFOLIO CONSTRUCTION (quarterly rebalancing, 5 bps on turnover)",
        _rule("═"),
        f"  {'Strategy':<22}{'CAGR':>9}{'Vol':>9}{'Sharpe':>8}{'MaxDD':>9}{'Beta':>7}{'Alpha':>9}",
        _rule(),
    ]
    for name, row in portfolios.iterrows():
        lines.append(
            f"  {name[:20]:<22}{_pct(row['CAGR']):>9}{_pct(row['Volatility']):>9}"
            f"{row['Sharpe']:>8.2f}{_pct(row['Max Drawdown']):>9}"
            f"{row['Beta']:>7.2f}{_pct(row['Alpha'], signed=True):>9}"
        )
    lines += [
        "",
        "  Note: the maximum-Sharpe weights are fitted on the same window they are",
        "  measured on, so that row is in-sample and optimistic. Equal weight and",
        "  inverse volatility use no return forecast and are the fair comparison.",
    ]

    lines += [
        "",
        _rule("═"),
        f"  RULE BACKTESTS ({len(scan)} rules × {len(bundle.frames)} holdings)",
        _rule("═"),
        f"  {'Rule':<24}{'MeanCAGR':>10}{'MeanSharpe':>12}{'MeanMaxDD':>11}{'Beats Hold':>12}",
        _rule(),
    ]
    for name, row in scan.iterrows():
        lines.append(
            f"  {name[:22]:<24}{_pct(row['Mean CAGR']):>10}{row['Mean Sharpe']:>12.2f}"
            f"{_pct(row['Mean Max Drawdown']):>11}{_pct(row['Win Rate vs Hold'], 0):>12}"
        )

    lines += ["", _rule("═"), "  FINDINGS", _rule("═")]
    for i, headline in enumerate(bundle.headlines, 1):
        wrapped = _wrap(headline, WIDTH - 6)
        lines.append(f"  {i}. {wrapped[0]}")
        lines.extend(f"     {part}" for part in wrapped[1:])

    lines += [
        "",
        _rule("═"),
        "  DATA QUALITY",
        _rule("═"),
        f"  Calendar coverage      : {bundle.quality['Coverage (%)'].mean():.2f}%",
        f"  Gaps forward-filled    : {int(bundle.quality['Calendar Gaps Filled'].sum()):,}",
        f"  Duplicate dates removed: {int(bundle.quality['Duplicate Dates'].sum()):,}",
        f"  Non-positive prices    : {int(bundle.quality['Non-Positive Prices'].sum()):,}",
        f"  Moves beyond 8σ        : {int(bundle.quality['Return Outliers (>8σ)'].sum()):,} "
        f"(flagged, not removed)",
        _rule("═"),
    ]
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def export_reports(bundle) -> dict[str, str]:
    """Write the text brief, the CSV set and a multi-sheet Excel workbook."""
    print("\n" + "─" * 72)
    print("  8 · REPORTS")
    print("─" * 72)
    paths: dict[str, str] = {}

    text = build_text_report(bundle)
    text_path = os.path.join(REPORT_DIR, "summary_report.txt")
    with open(text_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    paths["text"] = text_path
    print(f"    summary_report.txt")

    csv_targets = {
        "metrics.csv": bundle.metrics,
        "factor_scores.csv": bundle.factors,
        "sector_summary.csv": bundle.sectors,
        "current_signals.csv": bundle.signals,
        "portfolio_scorecard.csv": bundle.portfolios,
        "portfolio_weights.csv": bundle.weights,
        "strategy_scan.csv": bundle.strategy_scan,
        "data_quality.csv": bundle.quality,
    }
    for filename, frame in csv_targets.items():
        path = os.path.join(REPORT_DIR, filename)
        frame.to_csv(path)
        paths[filename] = path
        print(f"    {filename}")

    excel_path = os.path.join(REPORT_DIR, "equity_analytics.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            bundle.metrics.to_excel(writer, sheet_name="Metrics")
            bundle.factors.to_excel(writer, sheet_name="Factor Scores")
            bundle.sectors.to_excel(writer, sheet_name="Sectors")
            bundle.signals.to_excel(writer, sheet_name="Signals")
            bundle.portfolios.to_excel(writer, sheet_name="Portfolios")
            bundle.weights.to_excel(writer, sheet_name="Weights")
            bundle.strategy_scan.to_excel(writer, sheet_name="Strategy Scan")
            bundle.quality.to_excel(writer, sheet_name="Data Quality")
        paths["excel"] = excel_path
        print(f"    equity_analytics.xlsx  (8 sheets)")
    except Exception as exc:
        print(f"    Excel export skipped ({type(exc).__name__})")

    enriched_dir = os.path.join(DATA_DIR, "enriched")
    os.makedirs(enriched_dir, exist_ok=True)
    for ticker, frame in bundle.frames.items():
        frame.to_csv(os.path.join(enriched_dir, f"{ticker}.csv"))
    print(f"    {len(bundle.frames)} enriched per-symbol CSVs → data/enriched/")

    return paths
