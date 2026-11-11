"""Calibrate the mechanistic nitrogen extension with a frozen core."""

import numpy as np
from scipy.optimize import least_squares

from bioreactor_twin.calibration.fit import weighted_loss
from bioreactor_twin.data.dataset import condition_size
from bioreactor_twin.simulation.nitrogen import (
    SPECIES,
    Nitrogen,
    nitrogen_trace,
    nitrogen_values,
    observations,
)


def measured_nitrogen(run):
    """Legacy datasets may lack either additional assay channel."""
    return set(SPECIES) <= run.additional.keys()


def fit_nitrogen(runs, core, max_nfev=250):
    runs = [run for run in runs if measured_nitrogen(run)]
    if not runs:
        return None, None
    prepared = []
    weights = []
    for run in runs:
        actual = observations(run)[1:]
        if not np.all(np.isfinite(actual)) or np.min(actual) < 0:
            raise ValueError(f"Invalid glutamine or ammonia assays in Run {run.number}.")
        scale = np.hypot([0.005, 0.002], 0.1 * actual)
        prepared.append((run.times[1:], nitrogen_trace(run, core), actual, scale))
        weights.extend([1 / len(actual) / condition_size(run, runs)] * actual.size)

    def residual(vector):
        parameters = Nitrogen(*vector)
        return np.concatenate(
            [
                ((nitrogen_values(trace, parameters)[times] - actual) / scale).ravel()
                for times, trace, actual, scale in prepared
            ]
        )

    result = least_squares(
        residual,
        [0.1, 0.002, 0.1],
        bounds=(0.0, np.inf),
        loss=weighted_loss(np.array(weights)),
        max_nfev=max_nfev,
    )
    return Nitrogen(*result.x), {
        "success": bool(result.success),
        "cost": float(result.cost),
        "evaluations": int(result.nfev),
        "training_runs": [run.number for run in runs],
    }


def nitrogen_scores(run, values):
    if values.shape[1] == 5:
        return {}
    errors = values[run.times[1:], 5:] - observations(run)[1:]
    return dict(zip(SPECIES, np.sqrt(np.mean(errors**2, axis=0)).tolist(), strict=True))
