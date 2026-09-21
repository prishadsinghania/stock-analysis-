"""Central configuration: universe, date window, paths and analysis parameters."""

from __future__ import annotations

import datetime as _dt
import os

# ── Universe ─────────────────────────────────────────────────────────────────
# 56 US large caps across all 11 GICS sectors. Sector labels drive the
# sector-level aggregation, the factor screen and the portfolio constraints.
UNIVERSE: dict[str, dict[str, str]] = {
    # Information Technology
    "AAPL":  {"name": "Apple",                  "sector": "Information Technology"},
    "MSFT":  {"name": "Microsoft",              "sector": "Information Technology"},
    "NVDA":  {"name": "NVIDIA",                 "sector": "Information Technology"},
    "AVGO":  {"name": "Broadcom",               "sector": "Information Technology"},
    "ORCL":  {"name": "Oracle",                 "sector": "Information Technology"},
    "CRM":   {"name": "Salesforce",             "sector": "Information Technology"},
    "AMD":   {"name": "Advanced Micro Devices", "sector": "Information Technology"},
    "ADBE":  {"name": "Adobe",                  "sector": "Information Technology"},
    "CSCO":  {"name": "Cisco Systems",          "sector": "Information Technology"},
    "INTC":  {"name": "Intel",                  "sector": "Information Technology"},
    "TXN":   {"name": "Texas Instruments",      "sector": "Information Technology"},
    "QCOM":  {"name": "Qualcomm",               "sector": "Information Technology"},

    # Communication Services
    "GOOGL": {"name": "Alphabet",               "sector": "Communication Services"},
    "META":  {"name": "Meta Platforms",         "sector": "Communication Services"},
    "NFLX":  {"name": "Netflix",                "sector": "Communication Services"},
    "DIS":   {"name": "Walt Disney",            "sector": "Communication Services"},
    "CMCSA": {"name": "Comcast",                "sector": "Communication Services"},
    "T":     {"name": "AT&T",                   "sector": "Communication Services"},

    # Consumer Discretionary
    "AMZN":  {"name": "Amazon",                 "sector": "Consumer Discretionary"},
    "TSLA":  {"name": "Tesla",                  "sector": "Consumer Discretionary"},
    "HD":    {"name": "Home Depot",             "sector": "Consumer Discretionary"},
    "MCD":   {"name": "McDonald's",             "sector": "Consumer Discretionary"},
    "NKE":   {"name": "Nike",                   "sector": "Consumer Discretionary"},
    "SBUX":  {"name": "Starbucks",              "sector": "Consumer Discretionary"},
    "LOW":   {"name": "Lowe's",                 "sector": "Consumer Discretionary"},

    # Consumer Staples
    "WMT":   {"name": "Walmart",                "sector": "Consumer Staples"},
    "PG":    {"name": "Procter & Gamble",       "sector": "Consumer Staples"},
    "KO":    {"name": "Coca-Cola",              "sector": "Consumer Staples"},
    "PEP":   {"name": "PepsiCo",                "sector": "Consumer Staples"},
    "COST":  {"name": "Costco",                 "sector": "Consumer Staples"},

    # Financials
    "BRK-B": {"name": "Berkshire Hathaway",     "sector": "Financials"},
    "JPM":   {"name": "JPMorgan Chase",         "sector": "Financials"},
    "V":     {"name": "Visa",                   "sector": "Financials"},
    "MA":    {"name": "Mastercard",             "sector": "Financials"},
    "BAC":   {"name": "Bank of America",        "sector": "Financials"},
    "GS":    {"name": "Goldman Sachs",          "sector": "Financials"},
    "AXP":   {"name": "American Express",       "sector": "Financials"},

    # Health Care
    "UNH":   {"name": "UnitedHealth",           "sector": "Health Care"},
    "JNJ":   {"name": "Johnson & Johnson",      "sector": "Health Care"},
    "LLY":   {"name": "Eli Lilly",              "sector": "Health Care"},
    "ABBV":  {"name": "AbbVie",                 "sector": "Health Care"},
    "MRK":   {"name": "Merck",                  "sector": "Health Care"},
    "PFE":   {"name": "Pfizer",                 "sector": "Health Care"},
    "TMO":   {"name": "Thermo Fisher",          "sector": "Health Care"},

    # Industrials
    "CAT":   {"name": "Caterpillar",            "sector": "Industrials"},
    "BA":    {"name": "Boeing",                 "sector": "Industrials"},
    "HON":   {"name": "Honeywell",              "sector": "Industrials"},
    "UPS":   {"name": "United Parcel Service",  "sector": "Industrials"},
    "GE":    {"name": "GE Aerospace",           "sector": "Industrials"},

    # Energy
    "XOM":   {"name": "Exxon Mobil",            "sector": "Energy"},
    "CVX":   {"name": "Chevron",                "sector": "Energy"},
    "COP":   {"name": "ConocoPhillips",         "sector": "Energy"},
    "SLB":   {"name": "SLB",                    "sector": "Energy"},

    # Utilities
    "NEE":   {"name": "NextEra Energy",         "sector": "Utilities"},
    "DUK":   {"name": "Duke Energy",            "sector": "Utilities"},
    "SO":    {"name": "Southern Company",       "sector": "Utilities"},

    # Real Estate
    "AMT":   {"name": "American Tower",         "sector": "Real Estate"},
    "PLD":   {"name": "Prologis",               "sector": "Real Estate"},

    # Materials
    "LIN":   {"name": "Linde",                  "sector": "Materials"},
    "SHW":   {"name": "Sherwin-Williams",       "sector": "Materials"},
}

TICKERS: list[str] = list(UNIVERSE)
SECTORS: list[str] = sorted({m["sector"] for m in UNIVERSE.values()})

def sector_of(ticker: str) -> str:
    return UNIVERSE.get(ticker, {}).get("sector", "Unclassified")

def name_of(ticker: str) -> str:
    return UNIVERSE.get(ticker, {}).get("name", ticker)

def tickers_in(sector: str) -> list[str]:
    return [t for t, m in UNIVERSE.items() if m["sector"] == sector]

# ── Benchmark & risk-free proxy ──────────────────────────────────────────────
BENCHMARK = "SPY"            # S&P 500 ETF — beta, alpha, capture ratios
RISK_FREE_TICKER = "^IRX"    # 13-week T-bill yield (annualised %, quoted daily)
RISK_FREE_FALLBACK = 0.03    # used when the T-bill series is unavailable

# ── Analysis window ──────────────────────────────────────────────────────────
START_DATE = "2019-01-01"
END_DATE = _dt.date.today().isoformat()

TRADING_DAYS = 252

# ── Indicator parameters ─────────────────────────────────────────────────────
MA_SHORT, MA_LONG, MA_200 = 20, 50, 200
RSI_WINDOW = 14
BB_WINDOW, BB_STD = 20, 2.0
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_WINDOW = 14
VOL_WINDOW = 30              # rolling realised-volatility window (days)

# ── Factor screen ────────────────────────────────────────────────────────────
MOMENTUM_LOOKBACK = 252      # 12-month total return …
MOMENTUM_SKIP = 21           # … skipping the most recent month (12-1 momentum)
FACTOR_WEIGHTS = {           # composite score weights (must sum to 1.0)
    "momentum": 0.30,
    "risk_adjusted": 0.30,
    "low_volatility": 0.20,
    "drawdown_resilience": 0.20,
}

# ── Backtest ─────────────────────────────────────────────────────────────────
COST_BPS = 5.0               # round-trip transaction cost assumption, basis points

# ── Value at Risk ────────────────────────────────────────────────────────────
VAR_LEVELS = (0.95, 0.99)

# ── Paths ────────────────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "charts")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

for _dir in (DATA_DIR, CACHE_DIR, OUTPUT_DIR, REPORT_DIR):
    os.makedirs(_dir, exist_ok=True)

# ── Static chart rendering ───────────────────────────────────────────────────
CHART_DPI = 150
CHART_FIGSIZE = (14, 6)
