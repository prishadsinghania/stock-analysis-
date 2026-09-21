"""Command-line entry point for the full analysis pipeline.

    python main.py                 run the pipeline, export charts and reports
    python main.py --refresh       ignore the cache and re-download prices
    python main.py --no-charts     skip chart rendering

Outputs land in reports/ (text, CSV, Excel) and outputs/charts/ (PNG).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import OUTPUT_DIR, REPORT_DIR
from src.pipeline import build_analytics
from src.report_generator import export_reports
from src.visualizer import generate_all_charts


def main() -> int:
    parser = argparse.ArgumentParser(description="Equity analytics pipeline")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cached run and re-download prices")
    parser.add_argument("--no-charts", action="store_true",
                        help="skip PNG chart rendering")
    parser.add_argument("--no-reports", action="store_true",
                        help="skip report export")
    args = parser.parse_args()

    bundle = build_analytics(refresh=args.refresh)

    if not args.no_charts:
        generate_all_charts(bundle)
    if not args.no_reports:
        export_reports(bundle)

    print("\n" + "═" * 72)
    print("  DONE")
    print(f"  {len(bundle.tickers)} holdings · {bundle.observations:,} price observations · "
          f"{bundle.metrics.shape[1]} metrics each")
    if not args.no_reports:
        print(f"  Reports → {REPORT_DIR}")
    if not args.no_charts:
        print(f"  Charts  → {OUTPUT_DIR}")
    print("  Dashboard → python dashboard.py")
    print("═" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
