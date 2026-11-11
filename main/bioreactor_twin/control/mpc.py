"""Finite-horizon glucose bolus control with explicit physical constraints."""

import numpy as np
from scipy.optimize import minimize

from bioreactor_twin.control.dynamics import advance
from bioreactor_twin.simulation.engine import observe


def choose_dose(state, p, events, strength, target, settings, factor=1.0):
    count = (len(events) + settings.interval - 1) // settings.interval
    if not count:
        return 0.0, dict(success=True, message="No remaining horizon")
    schedule = np.arange(0, len(events), settings.interval)

    def plan(doses):
        proposed = events.copy()
        proposed[:, 1] = 0
        proposed[schedule, 1] = doses / 1000
        return proposed

    def roll(doses):
        current = state.copy()
        outputs = []
        constraints = []
        for amounts in plan(doses):
            # Check immediately after feed as well as at the next pre-event state.
            from bioreactor_twin.simulation.engine import event

            post = event(current, amounts, strength)
            constraints.extend(
                [settings.max_volume_ml - post[0] * 1000, settings.max_glucose - observe(post)[2]]
            )
            current = advance(current, p, amounts, strength, factor)
            outputs.append(observe(current)[0])
        return np.array(outputs), np.array(constraints)

    def objective(doses):
        try:
            outputs, _ = roll(doses)
            return np.mean(((outputs - target) / max(float(max(target)), 1.0)) ** 2) + 0.0001 * (
                np.sum(doses**2) + np.sum(np.diff(doses) ** 2)
            )
        except (ValueError, OverflowError):
            return 1e6

    def constraints(doses):
        try:
            return roll(doses)[1]
        except (ValueError, OverflowError):
            return np.full(2 * len(events), -1e6)

    fit = minimize(
        objective,
        np.full(count, min(0.1, settings.max_dose_ml)),
        method="SLSQP",
        bounds=[(0, settings.max_dose_ml)] * count,
        constraints=[dict(type="ineq", fun=constraints)],
        options=dict(maxiter=25, ftol=1e-7),
    )
    accepted = bool(fit.success and np.min(constraints(fit.x)) >= -1e-7)
    doses = fit.x if accepted else np.zeros(count)
    return float(doses[0]), dict(
        success=accepted, message=str(fit.message), planned_events=plan(doses).tolist()
    )
