"""Constrained moving-window state and growth-parameter estimation."""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import least_squares

from bioreactor_twin.calibration.fit import FLOORS
from bioreactor_twin.control.dynamics import advance
from bioreactor_twin.simulation.engine import observe


@dataclass
class Observer:
    state: np.ndarray
    parameters: object
    settings: object
    adaptive: bool = True
    factor: float = 1.0
    history: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    maintenance: float = 1.0

    def effective_parameters(self):
        return replace(
            self.parameters,
            glucose_maintenance=self.parameters.glucose_maintenance * self.maintenance,
        )

    def update(self, now, assays, events, strength):
        """Only delivered assays may enter this call; return the estimate at now."""
        latest = max(assays)
        start = max(t for t in self.history if t <= max(0, latest - self.settings.window))
        anchor, prior, *maintenance_history = self.history[start]
        prior_maintenance = next(iter(maintenance_history), 1.0)
        times = sorted(t for t in assays if start < t <= latest)
        if not times:
            return
        # One growth multiplier per observed interval, not an unconstrained hourly sequence.
        cuts = np.array([start, *times])
        size = len(times) if self.adaptive else 0
        initial = np.r_[
            np.zeros(5), np.full(size, np.log(prior)), np.full(size, np.log(prior_maintenance))
        ]
        lower = np.r_[np.full(5, -0.7), np.full(size, np.log(0.1)), np.full(size, np.log(0.25))]
        upper = np.r_[np.full(5, 0.7), np.full(size, np.log(1.5)), np.full(size, np.log(3.0))]

        def replay(vector):
            state = anchor.copy()
            state[1:] = np.maximum(state[1:], 1e-12) * np.exp(vector[:5])
            predicted = {start: observe(state)}
            for hour in range(start, now):
                index = min(np.searchsorted(cuts[1:], hour, side="right"), len(times) - 1)
                factor = np.exp(vector[5 + index]) if self.adaptive else 1.0
                maintenance = np.exp(vector[5 + size + index]) if self.adaptive else 1.0
                p = replace(
                    self.parameters,
                    glucose_maintenance=self.parameters.glucose_maintenance * maintenance,
                )
                state = advance(state, p, events[hour], strength, factor)
                predicted[hour + 1] = observe(state)
            return state, predicted

        def residual(vector):
            try:
                _, predicted = replay(vector)
                errors = [
                    (predicted[t][:4] - assays[t][:4])
                    / np.sqrt(FLOORS[:4] ** 2 + (0.1 * abs(assays[t][:4])) ** 2)
                    for t in times
                ]
                return np.r_[
                    np.concatenate(errors),
                    vector[:5] / 0.2,
                    np.diff(np.r_[np.log(prior), vector[5 : 5 + size]]) / 0.7,
                    np.diff(np.r_[np.log(prior_maintenance), vector[5 + size :]]) / 0.5,
                ]
            except (ValueError, OverflowError):
                return np.full(len(times) * 4 + 5 + 2 * size, 1e5)

        fit = least_squares(residual, initial, bounds=(lower, upper), max_nfev=80, loss="soft_l1")
        accepted = False
        sensitivity = None
        if fit.success and np.isfinite(fit.cost):
            try:
                candidate, _ = replay(fit.x)
            except (ValueError, OverflowError):
                pass
            else:
                self.state = candidate
                accepted = True
                if self.adaptive:
                    self.factor = float(np.exp(fit.x[4 + size]))
                    self.maintenance = float(np.exp(fit.x[-1]))
                    sensitivity = float(np.linalg.norm(fit.jac[: len(times) * 4, 4 + size]))
        self.diagnostics.append(
            dict(
                hour=now,
                last_assay=latest,
                factor=self.factor,
                maintenance=self.maintenance,
                converged=bool(fit.success),
                accepted=accepted,
                sensitivity=sensitivity,
                cost=float(fit.cost) if np.isfinite(fit.cost) else None,
            )
        )
