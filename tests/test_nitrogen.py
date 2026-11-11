"""Mechanistic nitrogen conservation, independent solver and workflow checks."""

from dataclasses import asdict, replace

import numpy as np
import pytest
from bioreactor_twin.calibration.nitrogen import fit_nitrogen, nitrogen_scores
from bioreactor_twin.calibration.validation import train_validated
from bioreactor_twin.data.feeds import FEED_CONCENTRATIONS
from bioreactor_twin.models.kinetics import Parameters, derivative, rates
from bioreactor_twin.simulation.engine import initialize, simulate
from bioreactor_twin.simulation.nitrogen import (
    GRAMS_PER_MMOL,
    Nitrogen,
    nitrogen_trace,
    nitrogen_values,
    simulate_mechanistic,
)
from bioreactor_twin.web.charts import simulation_figure
from bioreactor_twin.web.services import replicate_analysis
from scipy.integrate import solve_ivp
from test_numerics import synthetic_run


def with_nitrogen(run):
    return replace(run, additional={name: np.full(len(run.times), 0.01) for name in ("GLN", "AMM")})


def test_nitrogen_matches_independent_ode_and_preserves_core():
    run = with_nitrogen(synthetic_run())
    run.events[:] = 0
    core = Parameters()
    p = Nitrogen(0.4, 0.02, 0.05, glutamine_degradation=0.01)
    initial = np.r_[initialize(run), 0.01 * run.volumes[0] / GRAMS_PER_MMOL]

    def rhs(_t, state):
        volume, viable = state[:2]
        glutamine = state[6]
        mu = rates(state[:6], core)[0]
        chemical = p.glutamine_degradation * glutamine
        uptake = p.glutamine_uptake * glutamine / (p.k_glutamine * volume + glutamine)
        return (
            *derivative(state[:6], core),
            viable * (p.glutamine_growth * mu - uptake) - chemical,
            viable * p.ammonia_growth * mu + chemical,
        )

    truth = solve_ivp(rhs, (0, 4), initial, t_eval=run.times, rtol=1e-11, atol=1e-13)
    expected = truth.y[6:].T * GRAMS_PER_MMOL / run.volumes[0]
    model = simulate_mechanistic(run, core, p)
    np.testing.assert_array_equal(model[:, :5], simulate(run, core))
    refined = nitrogen_values(nitrogen_trace(run, core, 0.5), p)
    assert np.max(abs(refined - expected)) < np.max(abs(model[:, 5:] - expected)) / 3
    np.testing.assert_allclose(model[:, 5:], expected, atol=1e-6)


def test_nitrogen_feed_and_sampling_are_pre_event_and_mass_conservative():
    run = with_nitrogen(synthetic_run())
    run.events[:] = 0
    run.events[0] = [0.01, 0.002, 0.008, 0.004]
    run = replace(run, volumes=np.full(5, 0.204), strength=0.5)
    p = Nitrogen(0, 0, 0)
    value = simulate_mechanistic(run, Parameters(), p)[:, 5:]
    np.testing.assert_array_equal(value[0], [0.01, 0.01])
    feed = FEED_CONCENTRATIONS["B"]["GLN"]
    expected = (0.19 * 0.01 + np.array([0.008 * 0.5 * feed, 0])) / 0.204
    np.testing.assert_allclose(value[1:], np.tile(expected, (4, 1)))
    depleted = simulate_mechanistic(run, Parameters(), replace(p, glutamine_uptake=1e4))
    assert np.min(depleted[:, 5:]) >= 0
    assert depleted[1, 5] == pytest.approx(0, abs=1e-12)


def test_nitrogen_training_ignores_holdout_and_exports_rates():
    runs = {n: with_nitrogen(synthetic_run(n)) for n in (1, 2, 7, 8)}

    def fitter(*args, **kwargs):
        return Parameters(), {"success": True}

    def train():
        return train_validated(runs, "train_test", (1, 2), 1, 50, fitter, lambda _: None)

    original = train()
    assert original["validation"]["folds"][0]["diagnostics"]["nitrogen"]["training_runs"] == [7, 8]
    assert original["validation"]["folds"][0]["nitrogen"] == asdict(original["nitrogen"])
    runs[1].additional["GLN"][1:] = 0.5
    changed = train()
    assert changed["nitrogen"] == original["nitrogen"]
    assert changed["scores"][0]["GLN"] > original["scores"][0]["GLN"]
    info = replicate_analysis(runs, (1, 2), Parameters(), original["nitrogen"])
    assert info["adjusted_outputs"][-2:] == ["GLN", "AMM"]


def test_nitrogen_handles_legacy_and_bad_initial_assays():
    run = synthetic_run()
    core = Parameters()
    p = Nitrogen(0, 0, 0)
    assert fit_nitrogen([run], core) == (None, None)
    legacy = simulate_mechanistic(run, core, p)
    assert nitrogen_scores(run, legacy) == {}
    full = simulate_mechanistic(with_nitrogen(run), core, p)
    assert len(simulation_figure(full, legacy, "GLN").data) == 1
    with pytest.raises(ValueError, match="nonnegative"):
        Nitrogen(-1, 0, 0)
    with pytest.raises(ValueError, match="saturation"):
        replace(p, k_glutamine=0)
    run = with_nitrogen(run)
    run.additional["GLN"][1] = np.nan
    with pytest.raises(ValueError, match="assays"):
        fit_nitrogen([run], core)
    run.additional["GLN"][0] = -1
    with pytest.raises(ValueError, match="initial"):
        simulate_mechanistic(run, core, p)
