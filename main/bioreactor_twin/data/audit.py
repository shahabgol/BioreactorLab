"""Reproducible read-only audit of the supplied experimental workbook.

Report suspicious records; do not repair, impute, deduplicate or normalize the source.
"""

import hashlib
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

HEADERS = (
    "Vessel",
    "Feed_Elapsed_Time (h)",
    "DCC (cells/mL)",
    "TCC (cells/mL)",
    "VCC (cells/mL)",
    "GLC (g/L)",
    "Feed_Vsampled mL",
    "Feed_A mL",
    "Feed_B mL",
    "Vessel_Volume_Calculated_mL",
    "Feed_C mL",
    "pH measured",
    "pH temp corrected",
    "GLN (g/L)",
    "GLU (g/L)",
    "LAC (g/L)",
    "NH3 (g/L)",
)
RUN, TIME, DCD, TCD, VCD, GLC, SAMPLE, A, B, VOLUME, C, PH, PH_CORR, GLN, GLU, LAC, NH3 = HEADERS
MEASUREMENTS = (DCD, TCD, VCD, GLC, GLN, GLU, LAC, NH3)
EVENTS = (SAMPLE, A, B, C)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate structural fields and report biological/ledger diagnostics by run."""
    if not records:
        raise ValueError("Workbook contains no experimental records")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for record in records:
        for name in HEADERS:
            value = record.get(name)
            if value is None and name not in (RUN, TIME, VOLUME, *EVENTS):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Expected a numeric value for {name}: {value!r}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid nonnegative finite value for {name}: {value!r}")
        if int(record[RUN]) != record[RUN] or int(record[TIME]) != record[TIME]:
            raise ValueError("Vessel and elapsed hour must be integers")
        if record[VOLUME] <= 0:
            raise ValueError("Vessel volume must be positive")
        key = (record[RUN], record[TIME])
        if key in seen:
            raise ValueError(f"Duplicate run/time key: {key}")
        seen.add(key)
        grouped[int(record[RUN])].append(record)

    runs = []
    for run, original in sorted(grouped.items()):
        rows = sorted(original, key=lambda r: r[TIME])
        observed = [r for r in rows if any(r[k] is not None for k in MEASUREMENTS)]
        volume_residuals = []
        nonhourly = []
        for previous, current in pairwise(rows):
            if current[TIME] - previous[TIME] != 1:
                nonhourly.append([previous[TIME], current[TIME]])
            residual = (
                current[VOLUME]
                - previous[VOLUME]
                + current[SAMPLE]
                - current[A]
                - current[B]
                - current[C]
            )
            volume_residuals.append((current[TIME], residual))
        panels: dict[tuple, list[int | float]] = defaultdict(list)
        for row in observed:
            panels[tuple(row[k] for k in MEASUREMENTS)].append(row[TIME])
        duplicates = [times for times in panels.values() if len(times) > 1]
        closure = [
            abs(r[TCD] - r[VCD] - r[DCD])
            for r in observed
            if all(r[k] is not None for k in (TCD, VCD, DCD))
        ]
        runs.append(
            {
                "run_id": run,
                "row_count": len(rows),
                "time_range_h": [rows[0][TIME], rows[-1][TIME]],
                "source_was_time_sorted": rows == original,
                "nonhourly_intervals": nonhourly,
                "measurement_panels": len(observed),
                "measurement_times_h": [r[TIME] for r in observed],
                "measurement_gaps_h": [b[TIME] - a[TIME] for a, b in pairwise(observed)],
                "missing_measurements_in_panels": {
                    k: sum(r[k] is None for r in observed) for k in MEASUREMENTS
                },
                "identical_measurement_panels_at_h": duplicates,
                "zero_measurements": {
                    k: [r[TIME] for r in observed if r[k] == 0] for k in MEASUREMENTS
                },
                "max_abs_cell_identity_error_cells_per_mL": max(closure, default=None),
                "vcd_exceeds_tcd_at_h": [
                    r[TIME]
                    for r in observed
                    if r[VCD] is not None and r[TCD] is not None and r[VCD] > r[TCD]
                ],
                "initial_reported_volume_mL": rows[0][VOLUME],
                "initial_row_feed_C_mL": rows[0][C],
                "final_reported_volume_mL": rows[-1][VOLUME],
                "total_events_mL_including_initial_row": {
                    k: sum(r[k] for r in rows) for k in EVENTS
                },
                "max_abs_volume_ledger_error_after_initial_row_mL": max(
                    (abs(error) for _, error in volume_residuals), default=0
                ),
                "volume_ledger_discrepancies_over_1e-6_mL": [
                    {"time_h": t, "residual_mL": error}
                    for t, error in volume_residuals
                    if abs(error) > 1e-6
                ],
                "measurement_and_feed_same_hour": [
                    r[TIME] for r in observed if r[A] + r[B] + r[C] > 0
                ],
            }
        )
    return {"row_count": len(records), "runs": runs}


def audit_workbook(path: Path) -> dict[str, Any]:
    """Read one explicitly selected workbook; preserve source values and cell provenance."""
    from openpyxl import load_workbook

    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        if workbook.sheetnames != ["SampleData"]:
            raise ValueError(f"Unexpected sheets: {workbook.sheetnames}")
        sheet = workbook["SampleData"]
        iterator = sheet.iter_rows()
        headers = tuple(cell.value for cell in next(iterator))
        if headers != HEADERS:
            raise ValueError(f"Unexpected source columns: {headers}")
        records = []
        for excel_row, cells in enumerate(iterator, start=2):
            if any(cell.data_type in ("f", "e") for cell in cells):
                raise ValueError(f"Formula or Excel error at row {excel_row}; inspect explicitly")
            values = tuple(cell.value for cell in cells)
            if not any(value is not None for value in values):
                continue
            record = dict(zip(headers, values, strict=True))
            record["source_excel_row"] = excel_row
            records.append(record)
        summary = summarize_records(records)
        return {
            "schema_version": 1,
            "source_name": path.name,
            "source_sha256": checksum,
            "sheet": sheet.title,
            "headers": list(headers),
            "policy": "No imputation, deduplication, unit conversion, or source edits performed.",
            **summary,
        }
    finally:
        workbook.close()
