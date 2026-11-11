"""Reproducible offline workflow; reusable model services have no UI dependency."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from bioreactor_twin.calibration.fit import KEYS, fit
from bioreactor_twin.data.dataset import NAMES, load_runs
from bioreactor_twin.simulation.engine import simulate
from bioreactor_twin.validation.plots import plot_output
from bioreactor_twin.validation.uncertainty import metrics, repeatability


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_workflow(workbook: Path, output: Path, starts=8, max_nfev=350):
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(workbook)
    parameters, diagnostics = fit([runs[7], runs[8]], starts=starts, max_nfev=max_nfev)
    write_json(output / "parameters.json", asdict(parameters))
    write_json(output / "calibration.json", diagnostics)
    return export_results(runs, parameters, output)


def export_results(runs, parameters, output):
    """Evaluate a frozen model; uncertainty diagnostics cannot change its parameters."""
    predictions = {n: simulate(r, parameters) for n, r in runs.items()}
    # Holdout assays first enter here, after parameter estimation has completed.
    uncertainty = repeatability(runs[1], runs[2], predictions)
    write_json(output / "repeatability.json", uncertainty)
    scores = {n: metrics(r, predictions[n]) for n, r in runs.items()}
    write_json(output / "metrics.json", {"outputs": NAMES, "runs": scores})
    refinement = {}
    for number, run in runs.items():
        reference = simulate(run, parameters, reference=True)
        refinement[number] = {}
        for step in (1.0, 0.5, 0.25):
            result = simulate(run, parameters, step=step)
            error = np.abs(result - reference) / (1 + np.abs(reference))
            refinement[number][str(step)] = float(np.nanmax(error))
        frame = pd.DataFrame(predictions[number], columns=NAMES)
        frame.index.name = "time_h"
        frame.to_csv(output / f"run_{number}_hourly.csv")
    write_json(output / "numerical_checks.json", refinement)
    for column, name in enumerate(NAMES):
        plot_output(
            runs,
            predictions,
            np.array(uncertainty["sd"]),
            column,
            output / f"{name.lower()}_comparison.png",
        )
    table = pd.DataFrame({"parameter": KEYS, "value": [getattr(parameters, key) for key in KEYS]})
    table.to_csv(output / "calibrated_parameters.csv", index=False)
    return parameters, scores
