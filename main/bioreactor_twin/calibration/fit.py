"""Bounded multi-start trajectory fitting; holdout runs never enter the objective."""

from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from bioreactor_twin.data.dataset import condition_size
from bioreactor_twin.models.kinetics import Parameters
from bioreactor_twin.simulation.engine import simulate

KEYS = tuple(asdict(Parameters()))[:11]
LOWER = np.array([0.01, 2.0, 18, 0.00001, 0.0001, 0.05, 0.001, 0.0001, 0.00001, 0.01, 0.001])
UPPER = np.array([0.15, 60, 80, 0.01, 0.08, 15, 0.8, 2, 0.05, 2, 0.8])
FLOORS = np.array([0.15, 0.15, 0.10, 0.02, 0.05])


def unpack(vector, **fixed):
    return Parameters(**dict(zip(KEYS, np.exp(vector), strict=True)), **fixed)


def residuals(vector, runs, fixed):
    """Standardize retained assays, exclude first panel, balance each output/run."""
    parts = []
    for run in runs:
        scale = np.sqrt(FLOORS**2 + (0.10 * np.abs(run.observed[1:])) ** 2)
        try:
            predicted = simulate(run, unpack(vector, **fixed))[run.times[1:]]
            values = (predicted - run.observed[1:]) / scale
        except (ValueError, OverflowError, FloatingPointError):
            values = np.full_like(scale, 1e4)
        parts.append(values.ravel())
    return np.concatenate(parts)


def weighted_loss(weights):
    """Apply experiment weights outside soft-L1, preserving its residual cutoff."""

    def loss(squared):
        root = np.sqrt(1 + squared)
        return weights * np.array([2 * (root - 1), 1 / root, -0.5 / root**3])

    return loss


def fit(runs, starts=8, seed=20261110, max_nfev=350, grouped=False, on_progress=None, **fixed):
    """Fit log parameters; identical training groups each have eleven scored panels."""
    if not grouped and len({len(r.times) for r in runs}) != 1:
        raise ValueError("This balanced fit requires equal retained panel counts")
    rng = np.random.default_rng(seed)
    center = np.log([getattr(Parameters(), key) for key in KEYS])
    best = None
    attempts = []
    weights = (
        np.concatenate(
            [
                np.full(
                    (len(r.times) - 1) * 5,
                    1 / (len(r.times) - 1) / condition_size(r, runs),
                )
                for r in runs
            ]
        )
        if grouped
        else None
    )
    for index in range(starts):
        initial = center if index == 0 else center + rng.normal(0, 0.45, len(KEYS))
        initial = np.clip(initial, np.log(LOWER) + 1e-6, np.log(UPPER) - 1e-6)
        began = perf_counter()
        result = least_squares(
            residuals,
            initial,
            args=(runs, fixed),
            bounds=(np.log(LOWER), np.log(UPPER)),
            loss=weighted_loss(weights) if grouped else "soft_l1",
            max_nfev=max_nfev,
            ftol=2e-6,
            xtol=2e-6,
            gtol=2e-6,
        )
        record = {
            "start": index,
            "cost": float(result.cost),
            "success": bool(result.success),
            "nfev": result.nfev,
            "seconds": perf_counter() - began,
        }
        attempts.append(record)
        if on_progress is not None:
            on_progress(record)
        print(record, flush=True)
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        raise ValueError("At least one calibration start is required")
    parameters = unpack(best.x, **fixed)
    for run in runs:
        simulate(run, parameters)  # reject an optimum consisting of numerical penalties
    singular = np.linalg.svd(best.jac, compute_uv=False)
    return parameters, {
        "attempts": attempts,
        "cost": float(best.cost),
        "success": bool(best.success),
        "jacobian_singular_values": singular.tolist(),
        "fixed": {k: v for k, v in asdict(parameters).items() if k not in KEYS},
    }
