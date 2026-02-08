#!/usr/bin/env python3
"""Generate vnstock snapshots for a single ticker."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(
    "/Users/quangluu/Documents/Projects/InvestingAgent/buffet_style_investing_agent/phase_0/snapshots"
)
DEFAULT_ZIP_PATH = Path(
    "/Users/quangluu/Documents/Projects/InvestingAgent/buffet_style_investing_agent/phase_0/vnstock-main.zip"
)


def ensure_vnstock_importable(zip_path: Path, extract_base: Path) -> None:
    """Import vnstock from current environment or from local source zip."""
    try:
        import vnstock  # noqa: F401
        return
    except ImportError:
        pass

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Could not import 'vnstock' and zip file does not exist: {zip_path}"
        )

    extract_base.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_base)

    source_root = extract_base / "vnstock-main"
    if not source_root.exists():
        raise FileNotFoundError(
            f"Extracted source root not found at: {source_root}"
        )

    sys.path.insert(0, str(source_root))
    try:
        import vnstock  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Unable to import vnstock even after extracting local source zip."
        ) from exc


def df_to_json_obj(df: Any) -> Any:
    """Convert DataFrame-like result into JSON-serializable object."""
    if df is None:
        return {}
    if hasattr(df, "to_dict"):
        records = df.to_dict(orient="records")
        if len(records) == 1:
            return records[0]
        return records
    if isinstance(df, dict):
        return df
    return {"value": str(df)}


def save_csv(df: Any, path: Path) -> None:
    if df is None or not hasattr(df, "to_csv"):
        raise ValueError(f"Expected DataFrame-like object for CSV output: {path.name}")
    df.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate vnstock snapshots.")
    parser.add_argument("--ticker", default="FPT", help="Ticker symbol (default: FPT)")
    parser.add_argument(
        "--source",
        default="vci",
        help="vnstock source/provider (default: KBS)",
    )
    parser.add_argument(
        "--start",
        default="2000-01-01",
        help="Start date for price history in YYYY-MM-DD (default: 2000-01-01)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output snapshots directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--vnstock-zip",
        type=Path,
        default=DEFAULT_ZIP_PATH,
        help=f"Path to vnstock source zip (default: {DEFAULT_ZIP_PATH})",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper().strip()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    ensure_vnstock_importable(args.vnstock_zip, script_dir / ".vendor")

    from vnstock import Vnstock

    stock = Vnstock().stock(symbol=ticker, source=args.source)

    price_df = stock.quote.history(start=args.start, end=datetime.now().strftime("%Y-%m-%d"), interval="1D")
    income_df = stock.finance.income_statement(period="year",lang='en')
    balance_df = stock.finance.balance_sheet(period="year",lang='en')
    cashflow_df = stock.finance.cash_flow(period="year",lang='en')

    try:
        profile_df = stock.company.profile()
    except Exception:
        profile_df = stock.company.overview()

    save_csv(price_df, output_dir / f"{ticker}_price.csv")
    save_csv(income_df, output_dir / f"{ticker}_income_statement.csv")
    save_csv(balance_df, output_dir / f"{ticker}_balance_sheet.csv")
    save_csv(cashflow_df, output_dir / f"{ticker}_cashflow.csv")

    profile_obj = df_to_json_obj(profile_df)
    with (output_dir / f"{ticker}_profile.json").open("w", encoding="utf-8") as f:
        json.dump(profile_obj, f, ensure_ascii=False, indent=2)

    print(f"Done. Snapshots saved to: {output_dir}")


if __name__ == "__main__":
    main()
