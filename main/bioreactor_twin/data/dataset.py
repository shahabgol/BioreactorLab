"""Immutable observations and hourly sample/feed events, with source provenance."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from bioreactor_twin.data.audit import HEADERS, summarize_records
from bioreactor_twin.data.feeds import FEED_CONCENTRATIONS

NAMES = ("VCD", "TCD", "GLC", "GLU", "LAC")
MW = np.array([180.156, 147.130, 90.078])
FEED_A = np.array([FEED_CONCENTRATIONS["A"]["GLC"], 0.0, 0.0]) * 1000 / MW
FEED_B = np.array([FEED_CONCENTRATIONS["B"][name] for name in ("GLC", "GLU")] + [0.0]) * 1000 / MW


@dataclass(frozen=True)
class Run:
    """Amounts use L, billion cells and mmol; assay outputs use million/mL and g/L."""

    number: int
    events: np.ndarray  # sample, A, B, C in L, indexed by elapsed integer hour
    volumes: np.ndarray  # post-event volume in L
    times: np.ndarray
    observed: np.ndarray  # columns NAMES
    source_rows: tuple[int, ...]
    strength: float = 1.0
    additional: dict = field(default_factory=dict)
    replicate_group: str | None = None
    age_offset_h: float = 0.0  # culture age at scenario hour zero


def make_run(number, records):
    """Remove the flagged repeated assay only; retain every process event."""
    events = np.array([[r[6], r[7], r[8], r[10]] for _, r in records]) / 1000
    volumes = np.array([r[9] for _, r in records]) / 1000
    panels = [(i, r) for i, r in records if r[4] is not None and (number, r[1]) != (7, 42)]
    observed = np.array([[r[4] / 1e6, r[3] / 1e6, r[5], r[14], r[15]] for _, r in panels])
    return Run(
        number,
        events,
        volumes,
        np.array([r[1] for _, r in panels], dtype=int),
        observed,
        tuple(i for i, _ in panels),
        0.5 if number == 8 else 1.0,
        {name: np.array([r[col] for _, r in panels]) for name, col in (("GLN", 13), ("AMM", 16))},
    )


def load_runs(path: Path) -> dict[int, Run]:
    """Validate original data before constructing normalized immutable run objects."""
    book = load_workbook(path, data_only=False, read_only=True)
    try:
        rows = list(book["SampleData"].values)
        if tuple(rows[0]) != HEADERS:
            raise ValueError("Unexpected workbook columns")
        summarize_records([dict(zip(HEADERS, r, strict=True)) for r in rows[1:]])
        result = {}
        for number in sorted({r[0] for r in rows[1:]}):
            records = [(i, r) for i, r in enumerate(rows[1:], 2) if r[0] == number]
            if [r[1] for _, r in records] != list(range(len(records))):
                raise ValueError("An hourly ledger starting at zero is required")
            result[number] = make_run(number, records)
        return result
    finally:
        book.close()


def condition_key(run):
    return (
        run.replicate_group
        if run.replicate_group is not None
        else ("Nominal replicates 1 / 2" if run.number in (1, 2) else f"Condition {run.number}")
    )


def condition_size(run, runs):
    return sum(condition_key(other) == condition_key(run) for other in runs)
