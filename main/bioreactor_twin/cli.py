"""Inspect sources or calibrate the offline digital twin."""

import argparse
import json
from pathlib import Path

from bioreactor_twin.data.audit import audit_workbook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    audit = subparsers.add_parser("audit", help="Audit the original workbook without modifying it")
    audit.add_argument("workbook", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    calibrate = subparsers.add_parser("fit", help="Fit runs 7/8; withhold nominal runs 1/2")
    calibrate.add_argument("workbook", type=Path)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--starts", type=int, default=8)
    calibrate.add_argument("--max-nfev", type=int, default=350)
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit_workbook(args.workbook)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Audited {result['row_count']} rows in {len(result['runs'])} runs.")
        print(f"Report: {args.output}")
        return 0
    if args.command == "fit":
        from bioreactor_twin.workflow import run_workflow

        run_workflow(args.workbook, args.output, args.starts, args.max_nfev)
        return 0
    print("Offline simulator and calibration are available; use audit or fit.")
    parser.print_help()
    return 0
