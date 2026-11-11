"""Linear observation operators: rates -> amounts -> pre-event concentrations."""

import numpy as np

from bioreactor_twin.data.feeds import FEED_CONCENTRATIONS
from bioreactor_twin.hybrid.species import assays


def integrate(trace, basis):
    """Apply RK4 cell exposure and sampling survival to every basis column."""
    basis = np.asarray(basis)
    if basis.ndim == 1:
        basis = basis[:, None]
    increments = np.zeros((len(trace.volume), basis.shape[1]))
    np.add.at(increments, trace.hours, trace.exposure[:, None] * basis)
    amount = np.zeros(basis.shape[1])
    result = np.full_like(increments, np.nan)
    start = int(trace.run.times[0])
    for hour in range(start, len(result)):
        result[hour] = amount / trace.volume[hour]
        amount = amount * trace.retention[hour] + increments[hour]
    return result


def boundary(trace, species, initial=None):
    run = trace.run
    start = int(run.times[0])
    amount = float(assays(run, species)[0] if initial is None else initial) * trace.volume[start]
    result = np.full(len(trace.volume), np.nan)
    feed = FEED_CONCENTRATIONS["B"]["GLN"] * run.strength if species == "GLN" else 0.0
    for hour in range(start, len(result)):
        result[hour] = amount / trace.volume[hour]
        amount = amount * trace.retention[hour] + run.events[hour, 2] * feed
    return result
