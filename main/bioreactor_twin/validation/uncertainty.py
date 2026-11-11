"""Descriptive, model-adjusted repeatability from two nominal replicate trajectories."""

import numpy as np

from bioreactor_twin.calibration.fit import FLOORS


def repeatability(first, second, predictions):
    """Match actual common assay hours; remove deterministic feed/initial-state effects.

    Differences of model residuals retain replicate variability without interpreting
    time-interpolated assays as data. Two runs do not identify a coverage guarantee.
    """
    common = np.intersect1d(first.times[1:], second.times[1:])
    a = np.searchsorted(first.times, common)
    b = np.searchsorted(second.times, common)
    error_a = first.observed[a] - predictions[first.number][common]
    error_b = second.observed[b] - predictions[second.number][common]
    difference = error_a - error_b
    sd = np.sqrt(np.mean(difference**2, axis=0) / 2)
    return {
        "common_times_h": common.tolist(),
        "sd": sd.tolist(),
        "paired_residual_differences": difference.tolist(),
        "interpretation": "Descriptive repeatability; two batches; not validated 95% coverage",
    }


def envelope(prediction, repeatability_sd, multiplier=1.96):
    width = multiplier * np.sqrt(FLOORS**2 + repeatability_sd**2)
    return np.maximum(0, prediction - width), prediction + width


def metrics(run, prediction):
    error = prediction[run.times[1:]] - run.observed[1:]
    return {
        "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "mae": np.mean(np.abs(error), axis=0).tolist(),
        "bias": np.mean(error, axis=0).tolist(),
    }
