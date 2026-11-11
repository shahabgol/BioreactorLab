"""One-way mechanistic extracellular glutamine and ammonia balances."""

from dataclasses import dataclass
from math import exp, expm1, isfinite, log

import numpy as np
from scipy.special import wrightomega

from bioreactor_twin.data.feeds import FEED_CONCENTRATIONS
from bioreactor_twin.hybrid.core import trajectory
from bioreactor_twin.simulation.engine import simulate

SPECIES = ("GLN", "AMM")
GRAMS_PER_MMOL = np.array([0.146145, 0.017031])  # AMM provisionally as NH3 equivalents
FEED_B_GLN = FEED_CONCENTRATIONS["B"]["GLN"] / GRAMS_PER_MMOL[0]


@dataclass(frozen=True)
class Nitrogen:
    glutamine_growth: float
    glutamine_uptake: float
    ammonia_growth: float
    k_glutamine: float = 0.25  # mM, fixed screening value
    glutamine_degradation: float = 0.0  # 1/h; enable only with supporting metadata

    def __post_init__(self):
        for value in vars(self).values():
            if not isfinite(value) or value < 0:
                raise ValueError("Nitrogen parameters must be finite and nonnegative.")
        if self.k_glutamine == 0:
            raise ValueError("Glutamine saturation must be positive.")


def observations(run):
    return np.column_stack([run.additional[name] for name in SPECIES])


def nitrogen_trace(run, core, duration=1.0):
    """Reuse the core RK4 nodes, including refined steps and event volumes."""
    trace = trajectory(run, core, duration)
    initial = observations(run)[0]
    if not np.all(np.isfinite(initial)) or np.min(initial) < 0:
        raise ValueError("Enter finite, nonnegative initial glutamine and ammonia.")
    # Each accepted RK4 step contributes four consecutive quadrature nodes.
    hours = trace.hours[::4].astype(int)
    weights = trace.exposure.reshape(-1, 4)
    growth = (weights * trace.nodes[:, 4].reshape(-1, 4)).sum(axis=1)
    step = trace.nodes[3::4, 5] - trace.nodes[::4, 5]
    return trace, initial, hours, weights.sum(axis=1), growth, step


def uptake(amount, capacity, exposure):
    """Exact flow for dM/dE = -u M/(K V + M), with E the cell exposure."""
    if amount == 0 or exposure == 0:
        return amount
    ratio = amount / capacity
    return capacity * float(wrightomega(log(ratio) + ratio - exposure / capacity))


def nitrogen_values(prepared, p):
    trace, initial, hours, exposure, growth, step = prepared
    run = trace.run
    output = np.full((len(run.events), 2), np.nan)
    start = run.times[0]
    glutamine, ammonia = initial * trace.volume[start] / GRAMS_PER_MMOL
    output[start] = initial
    for i, hour in enumerate(hours):
        if i == 0 or hours[i - 1] != hour:
            glutamine = glutamine * trace.retention[hour] + (
                run.events[hour, 2] * run.strength * FEED_B_GLN
            )
            ammonia *= trace.retention[hour]
        # Symmetric source/uptake splitting; each subflow preserves positivity.
        source = growth[i] * p.glutamine_growth / 2
        decay = p.glutamine_degradation * step[i] / 2
        remaining = exp(-decay)
        decayed = -expm1(-decay)
        ammonia += glutamine * decayed
        glutamine = (
            uptake(
                glutamine * remaining + source,
                p.k_glutamine * trace.volume[hour + 1],
                p.glutamine_uptake * exposure[i],
            )
            + source
        )
        ammonia += glutamine * decayed + growth[i] * p.ammonia_growth
        glutamine *= remaining
        if i == len(hours) - 1 or hours[i + 1] != hour:
            output[hour + 1] = np.array([glutamine, ammonia]) * (
                GRAMS_PER_MMOL / trace.volume[hour + 1]
            )
    return output


def simulate_mechanistic(run, core, nitrogen=None):
    values = simulate(run, core)
    if nitrogen is None or not set(SPECIES) <= run.additional.keys():
        return values
    return np.column_stack((values, nitrogen_values(nitrogen_trace(run, core), nitrogen)))
