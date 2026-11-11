"""Fixed hourly RK4 with exact pre-assay, sampling, then feeding event maps."""

from math import isfinite

import numpy as np
from scipy.integrate import solve_ivp

from bioreactor_twin.data.dataset import FEED_A, FEED_B, MW, Run
from bioreactor_twin.models.kinetics import Parameters, derivative


class InvalidState(ValueError):
    """An integration stage left the physical domain."""


def validate_state(state):
    if not all(isfinite(x) for x in state) or state[0] <= 0 or min(state[1:]) < -1e-10:
        raise InvalidState("Nonphysical integration state")


def observe(state):
    """Convert integrated amounts into assay units; TCD is always derived."""
    volume, viable, dead, *metabolites = state
    return np.r_[
        viable / volume, (viable + dead) / volume, np.array(metabolites) / volume * MW / 1000
    ]


def initialize(run: Run):
    """First assay initializes pre-event amounts and is excluded from scoring."""
    hour = run.times[0]
    sample, a, b, c = run.events[hour]
    volume = run.volumes[hour] + sample - a - b - c
    viable, total, *metabolites = run.observed[0]
    return np.r_[
        volume,
        viable * volume,
        (total - viable) * volume,
        np.array(metabolites) * 1000 / MW * volume,
    ]


def event(state, volumes, strength=1.0):
    """Well-mixed sample removes amounts; cell-free feed changes volume and solutes."""
    sample, a, b, c = volumes
    if min(volumes) < 0 or sample >= state[0]:
        raise ValueError("Invalid sample/feed volume")
    result = np.asarray(state, dtype=float).copy() * (1 - sample / state[0])
    result[0] += a + b + c
    result[3:] += a * FEED_A + b * strength * FEED_B
    validate_state(result)
    return result


def shifted(state, slope, step):
    return tuple(x + step * dx for x, dx in zip(state, slope, strict=True))


def rk4(state, p: Parameters, step=1.0):
    """One classical RK4 step; reject physical-domain violations at stages/end."""
    state = tuple(map(float, state))
    validate_state(state)
    k1 = derivative(state, p)
    second = shifted(state, k1, step / 2)
    validate_state(second)
    k2 = derivative(second, p)
    third = shifted(state, k2, step / 2)
    validate_state(third)
    k3 = derivative(third, p)
    fourth = shifted(state, k3, step)
    validate_state(fourth)
    k4 = derivative(fourth, p)
    result = tuple(
        x + step / 6 * (a + 2 * b + 2 * c + d)
        for x, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
    )
    validate_state(result)
    return result


def subdivisions(step):
    if not isfinite(step) or step <= 0:
        raise ValueError("Step must be positive and finite")
    count = round(1 / step)
    if not np.isclose(step * count, 1, rtol=0, atol=1e-12):
        raise ValueError("Step must divide one hour exactly")
    return count


def refine_step(state, time, duration, stepper, retries=7):
    """Retry an invalid RK4 stage inside the same event interval."""
    validate_state(state)
    try:
        result = stepper(state, time, duration)
        # The stage validator already rejects material negatives.
        return np.maximum(result, 0) if min(result[1:]) < 0 else result
    except InvalidState:
        if not retries:
            raise
        half = duration / 2
        middle = refine_step(state, time, half, stepper, retries - 1)
        return refine_step(middle, time + half, half, stepper, retries - 1)


def advance_rk4(state, p, step=1.0):
    for _ in range(subdivisions(step)):
        state = refine_step(state, 0, step, lambda x, _t, dt: rk4(x, p, dt))
    return state


def simulate(run: Run, p: Parameters, step=1.0, reference=False):
    """Return hourly pre-event observations, without resetting to later assays."""
    subdivisions(step)
    state = initialize(run)
    outputs = np.full((len(run.events), 5), np.nan)
    for hour in range(run.times[0], len(run.events) - 1):
        outputs[hour] = observe(state)
        state = event(state, run.events[hour], run.strength)
        if reference:
            solution = solve_ivp(
                lambda t, y: derivative(y, p),
                (0, 1),
                state,
                method="DOP853",
                rtol=1e-10,
                atol=1e-12,
                max_step=0.125,
            )
            if not solution.success:
                raise ValueError(solution.message)
            state = solution.y[:, -1]
            validate_state(state)
        else:
            state = advance_rk4(state, p, step)
    outputs[-1] = observe(state)
    event(state, run.events[-1], run.strength)
    return outputs
