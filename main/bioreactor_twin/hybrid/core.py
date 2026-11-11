"""Unchanged biological core; lactate closure is excluded from core calibration."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from bioreactor_twin.calibration.fit import FLOORS, KEYS, LOWER, UPPER, weighted_loss
from bioreactor_twin.data.dataset import condition_size
from bioreactor_twin.models.kinetics import Parameters, derivative, rates
from bioreactor_twin.simulation.engine import (
    event,
    initialize,
    observe,
    refine_step,
    validate_state,
)


def rhs(state, parameters):
    values = derivative(state, parameters)
    return (*values[:5], 0.0)


def features(state, parameters, time):
    volume, viable, dead, glucose, glutamate, _ = state
    total = viable + dead
    return [
        viable / volume,
        viable / total if total else 0.0,
        glucose / volume * 0.180156,
        glutamate / volume * 0.147130,
        rates(state, parameters)[0],
        time,
    ]


def step(state, parameters, time, duration, record=True):
    """RK4 core advance plus identical quadrature nodes for the rate balances."""
    s = np.asarray(state)
    k1 = np.array(rhs(s, parameters))
    second = s + duration * k1 / 2
    validate_state(second)
    k2 = np.array(rhs(second, parameters))
    third = s + duration * k2 / 2
    validate_state(third)
    k3 = np.array(rhs(third, parameters))
    fourth = s + duration * k3
    validate_state(fourth)
    k4 = np.array(rhs(fourth, parameters))
    result = s + duration / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    validate_state(result)
    if not record:
        return result, [], []
    nodes = [
        features(x, parameters, t)
        for x, t in (
            (s, time),
            (second, time + duration / 2),
            (third, time + duration / 2),
            (fourth, time + duration),
        )
    ]
    exposure = duration / 6 * np.array([s[1], 2 * second[1], 2 * third[1], fourth[1]])
    return result, nodes, exposure


@dataclass
class Trace:
    values: np.ndarray
    nodes: np.ndarray
    exposure: np.ndarray
    hours: np.ndarray
    volume: np.ndarray
    retention: np.ndarray
    run: object


def trajectory(run, parameters, duration=1.0, record=True):
    if duration not in (1.0, 0.5, 0.25):
        raise ValueError("Use one-hour, half-hour or quarter-hour integration.")
    state = initialize(run)
    start = int(run.times[0])
    size = len(run.events)
    values = np.full((size, 5), np.nan)
    volume = np.full(size, np.nan)
    retention = np.ones(size)
    nodes, exposure, hours = [], [], []
    for hour in range(start, size - 1):
        values[hour] = observe(state)
        volume[hour] = state[0]
        retention[hour] = 1 - run.events[hour, 0] / state[0]
        state = event(state, run.events[hour], run.strength)

        def record_step(current, time, dt, hour=hour):
            result, z, weight = step(current, parameters, time, dt, record)
            nodes.extend(z)
            exposure.extend(weight)
            hours.extend([hour] * len(weight))
            return result

        for sub in range(round(1 / duration)):
            time = hour + run.age_offset_h + sub * duration
            state = refine_step(state, time, duration, record_step)
    values[-1] = observe(state)
    volume[-1] = state[0]
    retention[-1] = 1 - run.events[-1, 0] / state[0]
    event(state, run.events[-1], run.strength)
    return Trace(
        values, np.array(nodes), np.array(exposure), np.array(hours), volume, retention, run
    )


def fit_core(runs, starts=1, evaluations=100):
    """Fit only nine core coefficients and four core assay channels."""
    runs = list(runs)
    center = np.log([getattr(Parameters(), k) for k in KEYS[:9]])
    weights = np.concatenate(
        [
            np.full(
                (len(r.times) - 1) * 4,
                1 / (len(r.times) - 1) / condition_size(r, runs),
            )
            for r in runs
        ]
    )

    def unpack(x):
        return Parameters(**dict(zip(KEYS[:9], np.exp(x), strict=True)))

    def residual(x):
        parts = []
        for r in runs:
            obs = r.observed[1:, :4]
            scale = np.sqrt(FLOORS[:4] ** 2 + (0.1 * abs(obs)) ** 2)
            try:
                # Avoid computing unused feature/quadrature records in the optimizer.
                pred = trajectory(r, unpack(x), record=False).values[r.times[1:], :4]
                parts.append(((pred - obs) / scale).ravel())
            except (ValueError, OverflowError, FloatingPointError):
                parts.append(np.full(obs.size, 1e4))
        return np.concatenate(parts)

    best = None
    attempts = []
    rng = np.random.default_rng(20261110)
    for index in range(starts):
        initial = np.clip(
            center + (rng.normal(0, 0.45, 9) if index else 0),
            np.log(LOWER[:9]) + 1e-6,
            np.log(UPPER[:9]) - 1e-6,
        )
        result = least_squares(
            residual,
            initial,
            bounds=(np.log(LOWER[:9]), np.log(UPPER[:9])),
            loss=weighted_loss(weights),
            max_nfev=evaluations,
            ftol=2e-6,
            xtol=2e-6,
            gtol=2e-6,
        )
        attempts.append(
            dict(cost=float(result.cost), success=bool(result.success), nfev=result.nfev)
        )
        if best is None or result.cost < best.cost:
            best = result
    parameters = unpack(best.x)
    for r in runs:
        trajectory(r, parameters)
    return parameters, dict(success=bool(best.success), attempts=attempts, cost=float(best.cost))
