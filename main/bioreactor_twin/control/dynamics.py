"""Hourly amount dynamics shared by plant, forecasts and constrained controllers."""

from dataclasses import dataclass, replace

import numpy as np

from bioreactor_twin.data.dataset import FEED_A, MW
from bioreactor_twin.models.kinetics import rates
from bioreactor_twin.simulation.engine import advance_rk4, event, initialize, observe


@dataclass(frozen=True)
class Settings:
    horizon: int = 48
    interval: int = 6
    assay_interval: int = 24
    delay: int = 0
    window: int = 72
    disturbance: int = 72
    multiplier: float = 0.5
    max_dose_ml: float = 2.0
    max_volume_ml: float = 300.0
    max_glucose: float = 15.0
    noise: float = 0.0
    seed: int = 42

    def validate(self):
        integers = (
            self.horizon,
            self.interval,
            self.assay_interval,
            self.delay,
            self.window,
            self.disturbance,
            self.seed,
        )
        if any(not np.isfinite(v) or int(v) != v or v < 0 for v in integers):
            raise ValueError("Times and seed must be nonnegative whole numbers.")
        if min(self.horizon, self.interval, self.assay_interval, self.window) < 1:
            raise ValueError("Horizon, decision/assay intervals and window must be positive.")
        if self.horizon > 96 or self.horizon < self.interval or self.window < self.assay_interval:
            raise ValueError(
                "Use decision interval <= horizon <= 96 h and window >= assay interval."
            )
        values = (
            self.multiplier,
            self.max_dose_ml,
            self.max_volume_ml,
            self.max_glucose,
            self.noise,
        )
        if not np.all(np.isfinite(values)) or min(values) < 0 or self.max_volume_ml <= 0:
            raise ValueError("Limits/noise must be finite and nonnegative; volume positive.")
        if not 0.1 <= self.multiplier <= 1.5 or self.max_glucose <= 0:
            raise ValueError("Growth multiplier must be 0.1–1.5 and glucose limit positive.")


def advance(state, p, volumes, strength=1.0, factor=1.0, step=1.0):
    result = event(state, volumes, strength)
    effective = replace(p, mu_max=p.mu_max * factor)
    return np.array(advance_rk4(result, effective, step))


def forecast(state, p, events, strength, factor=1.0):
    values = [observe(state)]
    for amounts in events:
        state = advance(state, p, amounts, strength, factor)
        values.append(observe(state))
    return np.array(values)


def reachability(state, p, net_growth):
    volume, viable, dead, _, glutamate, _ = state
    density = (viable + dead) / volume / p.capacity
    maximum = p.mu_max * (glutamate / volume) / (p.k_e + glutamate / volume) * max(0, 1 - density)
    death = rates(state, p)[1]
    required = net_growth + death
    phi = required / maximum if maximum > 0 else None
    feasible = 0 <= required < maximum or (maximum == 0 and required == 0)
    return dict(
        phi=phi,
        glucose_mM=p.k_g * phi / (1 - phi) if feasible and phi is not None else None,
        feasible=bool(feasible),
    )


def target_profile(rows, duration, initial):
    values = np.array([[float(r["hour"]), float(r["VCD"])] for r in rows])
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or len(values) < 2
        or not np.all(np.isfinite(values))
        or np.any(values[:, 1] <= 0)
        or np.any(np.diff(values[:, 0]) <= 0)
        or values[0, 0] != 0
        or values[-1, 0] < duration
    ):
        raise ValueError(
            "Target needs increasing hours from zero through the duration, with positive VCD."
        )
    if not np.isclose(values[0, 1], initial):
        raise ValueError("The VCD target at hour zero must match the initial VCD.")
    return np.interp(np.arange(duration + 1), values[:, 0], values[:, 1])


def reference_target(run, p, settings):
    """Replay an admissible nominal feed sequence to construct a tracking target."""
    state = initialize(run)
    target = [observe(state)[0]]
    feed_glucose = FEED_A[0] * MW[0] / 1000
    for hour, inputs in enumerate(run.events[:-1]):
        amounts = inputs.copy()
        amounts[1] = 0
        post = event(state, amounts, run.strength)
        glucose = observe(post)[2]
        room = settings.max_volume_ml / 1000 - post[0]
        if room < -1e-10 or glucose > settings.max_glucose + 1e-10:
            raise ValueError(f"Fixed feeds exceed the control limits at hour {hour}.")
        if hour % settings.interval == 0:
            dose = min(inputs[1], settings.max_dose_ml / 1000, room)
            if feed_glucose > settings.max_glucose:
                dose = min(
                    dose,
                    post[0]
                    * (settings.max_glucose - glucose)
                    / (feed_glucose - settings.max_glucose),
                )
            amounts[1] = max(dose, 0)
        state = advance(state, p, amounts, run.strength)
        target.append(observe(state)[0])
    return np.array(target)
